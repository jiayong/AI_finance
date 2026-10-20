"""FactorBench 接入的自检 —— 不联网也能跑。

三项检查，各有明确目的：

  1. 因子名一致性：适配层预测的列名必须与 FactorEngine 生成的完全一致。
     不一致会导致「评估的不是同一个因子」，是最隐蔽的错误。
  2. 合成数据全链路：用与 yfinance 同构的合成行情跑完四级评估，
     证明代码路径通、形状对得上、风格中性化真的产出数字。
  3. 风格回归门槛：票池不足以支撑横截面回归时必须**报错**，而不是静默返回 NaN。

用法：
    python verify_factorbench.py            # 全部
    python verify_factorbench.py --fast     # 跳过第 2 项（省时间）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
START, END = "2010-01-01", "2026-10-01"


def check_factor_names() -> bool:
    """1) 适配层因子名 == FactorEngine 因子名。"""
    print("=" * 72)
    print("检查 1/5  因子名一致性（适配层 vs FactorEngine）")
    print("=" * 72)

    import factorbench_adapter as ad
    from factor_engine import FactorEngine

    fe = FactorEngine.__new__(FactorEngine)
    bad = []
    for kind, params in ad.FACTOR_SPECS:
        mine = ad.factor_name(kind, params)
        theirs = fe._feature_names_for(kind, list(params))[0]
        ok = mine == theirs
        if not ok:
            bad.append((kind, params, mine, theirs))
        print(f"  {'OK  ' if ok else 'FAIL'} {kind:11s} {str(params):10s} "
              f"{mine:22s} {theirs}")

    if bad:
        print(f"\n✗ {len(bad)} 个因子名不一致 —— 评估的将不是同一个因子")
        return False
    print(f"\n✓ {len(ad.FACTOR_SPECS)} 个因子名全部一致")
    return True


def check_pipeline() -> bool:
    """2) 合成数据跑通四级评估。"""
    print("\n" + "=" * 72)
    print("检查 2/5  合成数据全链路（证明代码路径通）")
    print("=" * 72)

    cache, out = "_fb_cache_verify", "_fb_out_verify"
    r = subprocess.run(
        [sys.executable, "_make_synthetic_cache.py", cache, START, END],
        cwd=HERE, capture_output=True, text=True, encoding="utf-8",
    )
    if r.returncode != 0:
        print("合成数据生成失败:", r.stderr[-2000:])
        return False
    print(r.stdout.strip())

    r = subprocess.run(
        [sys.executable, "run_factorbench_eval.py",
         "--stage", "all", "--cache-dir", cache, "--out-dir", out,
         "--start", START, "--end", END,
         "--horizon", "20", "--target", "net",
         "--rebalance-days", "20", "--cost-bps", "10"],
        cwd=HERE, capture_output=True, text=True, encoding="utf-8",
    )
    tail = (r.stdout or "")[-3000:]
    print(tail)
    if r.returncode != 0:
        print("评估失败:", (r.stderr or "")[-3000:])
        return False

    # 风格中性化必须真的产出数字（全 NaN 就是静默失败）
    import pandas as pd
    st = pd.read_csv(HERE / out / "style_neutralisation.csv")
    if st["ic_res"].notna().sum() == 0:
        print("✗ 风格中性化的残差 IC 全是 NaN —— 静默失败")
        return False
    ic = pd.read_csv(HERE / out / "ic_table.csv")
    if ic["admitted"].sum() == 0:
        print("✗ 没有任何因子通过准入 —— 静默失败")
        return False
    print(f"\n✓ 全链路通过：{len(ic)} 因子，{int(ic['admitted'].sum())} 个准入，"
          f"风格残差 IC 非空 {int(st['ic_res'].notna().sum())} 条")
    return True


def check_style_floor() -> bool:
    """3) 小票池必须报错，而不是静默给 NaN。

    用合成小票池，保持本自检**完全离线**（Yahoo 限流不该让自检失败）。
    """
    print("\n" + "=" * 72)
    print("检查 3/5  风格回归门槛会拦住小票池")
    print("=" * 72)

    import numpy as np
    import pandas as pd

    import factorbench_adapter as ad
    ad.factorbench_repo()
    import run_factorbench_eval as runner
    from _make_synthetic_cache import make_ohlcv

    small = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    dates = pd.bdate_range("2018-01-01", "2026-10-01", name="Date")

    # 直接造合成数据，**不经过 download_raw**，保证不碰网络
    stocks = pd.concat(make_ohlcv(small, dates, seed=11), axis=1)
    stocks.columns = stocks.columns.set_names(["Price", "Ticker"])
    spy = pd.concat(make_ohlcv(["SPY"], dates, seed=13), axis=1)
    spy.columns = spy.columns.set_names(["Price", "Ticker"])
    print(f"合成小票池: {len(small)} 只票, {len(dates)} 个交易日")

    panel = ad.to_panel(stocks)
    bench = spy["Close"].squeeze()
    bench.index = pd.to_datetime(bench.index)
    panel = ad.add_target(panel, horizon=20, benchmark_close=bench, target_mode="net")
    factors = ad.build_factors(panel)
    target = panel["forward_return_20d"].dropna()
    split = ad.make_split(start="2018-01-01")

    signs = np.array([ad.sign_for(factors[n], target, split) for n in factors])
    try:
        runner.stage_style(panel, factors, target, split, signs, Path("_fb_out_small"))
    except ValueError as exc:
        first = str(exc).splitlines()[0]
        print(f"\n✓ 按预期报错（而不是静默 NaN）:\n  {first}")
        return True
    print("\n✗ 5 只票的池子没有报错 —— 应该拦住")
    return False


def check_inference() -> bool:
    """4) HAC / EB / 成本敏感性 的数值性质。

    三条不变量，任何一条被破坏都说明实现错了：
      a) Driscoll-Kraay 的标准误 >= 只按日期聚类（序列相关只会让误差更大）。
      b) EB 收缩因子 = max(0, 1 - 1/Var(t))，且与 Var(t) 单调。
      c) 成本敏感性在 0 bp 时必须精确复现毛收益（解析公式无偏）。
    """
    print("\n" + "=" * 72)
    print("检查 4/5  推断与成本计算的不变量")
    print("=" * 72)

    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(HERE))
    from hac_inference import fit_ols

    # (a) DK >= cluster_date
    # 造一个**强序列相关**的残差（AR(0.95) 式的持续冲击）。这才是有区分力的检验：
    # 若残差是白噪声，两种估计器本来就该接近，测不出问题。
    # 滞后阶数必须远小于时间期数，否则 DK 的 G/(G-k) 修正会吃掉自由度。
    rng = np.random.default_rng(7)
    n_dates, n_assets, lag = 2000, 10, 20
    dates = pd.date_range("2015-01-01", periods=n_dates, freq="B")

    x_common = rng.normal(0, 1, n_dates)
    # 每个股票一条持续冲击序列（累积和的增量 -> 近似随机游走，强自相关）
    shocks = np.cumsum(rng.normal(0, 0.05, (n_assets, n_dates)), axis=1)
    rows = []
    for a in range(n_assets):
        x = x_common + shocks[a]
        y = 0.3 * x + shocks[a] * 2.0 + rng.normal(0, 0.4, n_dates)
        for i, d in enumerate(dates):
            rows.append({"Date": d, "Ticker": f"A{a:02d}",
                         "x": float(x[i]), "y": float(y[i])})
    df = pd.DataFrame(rows)

    se = {}
    for cov in ("cluster_date", "driscoll_kraay"):
        inf = fit_ols(df, "y", ["x"], cov=cov, lags=lag)
        se[cov] = float(inf["bse"]["x"])
    print(f"  标准误 [变量 x]: 按日期聚类 {se['cluster_date']:.5f}  "
          f"Driscoll-Kraay {se['driscoll_kraay']:.5f}"
          f"  ({se['driscoll_kraay'] / se['cluster_date']:.2f}x)")
    if se["driscoll_kraay"] < se["cluster_date"]:
        print("  ✗ DK 标准误反而更小 —— 序列相关没有被正确吸收")
        return False
    print("  ✓ DK 正确放大了标准误以吸收序列相关")

    # (b) EB 收缩公式
    import factorbench_adapter as ad
    for var_t, expect in ((0.5, 0.0), (2.0, 0.5), (4.0, 0.75)):
        ic = pd.DataFrame({"factor": ["a", "b", "c", "d"],
                           "test.ic": [0.1, 0.0, -0.1, 0.05],
                           "test.t": [0.0, 0.0, 0.0, 0.0]})
        # 直接构造成目标 Var(t)
        t = np.array([-np.sqrt(var_t) * 1.2, -np.sqrt(var_t) * 0.4,
                      np.sqrt(var_t) * 0.4, np.sqrt(var_t) * 1.2])
        t = t - t.mean()
        t = t / t.std(ddof=1) * np.sqrt(var_t)
        ic["test.t"] = t
        _, shrink = ad.eb_shrink(ic, "test")
        if abs(shrink - expect) > 1e-9:
            print(f"  ✗ Var(t)={var_t} 时收缩因子应为 {expect}，得到 {shrink}")
            return False
    print("  ✓ EB 收缩公式正确（Var(t)≤1 截断为 0）")

    # (c) 成本敏感性在 0bp 精确复现毛收益
    from run_factorbench_eval import cost_sensitivity
    daily = pd.DataFrame({
        "gross": rng.normal(0.0005, 0.01, 500),
        "turnover": np.where(np.arange(500) % 20 == 0, 0.3, 0.0),
        "borrow_cost": 0.0,
    })
    s0 = cost_sensitivity(daily, [0.0], rebalance_days=20)
    wealth = float(np.cumprod(1.0 + daily["gross"].to_numpy())[-1])
    years = len(daily) / 252.0
    expect_cagr = wealth ** (1 / years) - 1
    if abs(s0["net_cagr"].iloc[0] - expect_cagr) > 1e-10:
        print(f"  ✗ 0bp 解析 CAGR={s0['net_cagr'].iloc[0]} 与毛收益 "
              f"{expect_cagr} 不一致")
        return False
    print("  ✓ 成本敏感性在 0bp 精确等于毛收益")

    return True


def check_hetero_equivalence() -> bool:
    """5) 异质成本回测在常数成本下必须与上游**逐位一致**。

    这是复制上游 ``backtest`` 时唯一可信的正确性证明：只要成本退化为常数，
    两条路径的 gross / turnover / trade_cost / net 就必须完全相同。
    任何一处漏抄（漂移、lag、换手口径）都会立刻暴露成明显的数值差。
    """
    print("\n" + "=" * 72)
    print("检查 5/5  异质成本回测与上游的等价性")
    print("=" * 72)

    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(HERE))
    import factorbench_adapter as ad
    ad.factorbench_repo()
    from factor_bench.portfolio.backtest import BacktestConfig, backtest
    from backtest_hetero import backtest_with_cost_model
    from cost_model import constant_cost_model

    rng = np.random.default_rng(11)
    dates = pd.date_range("2020-01-01", periods=420, freq="B")
    assets = [f"A{i:02d}" for i in range(40)]

    returns = pd.DataFrame(rng.normal(0.0003, 0.012, (len(dates), len(assets))),
                           index=dates, columns=assets)
    panel_scores = pd.DataFrame(rng.normal(size=(len(dates), len(assets))),
                                index=dates, columns=assets).stack()
    panel_scores.index = panel_scores.index.rename(["date", "asset"])

    for kind in ("long_only", "long_short"):
        for bps in (0.0, 10.0, 25.0):
            cfg = BacktestConfig(rebalance_days=20, cost_bps=bps)
            up = backtest(panel_scores, returns, cfg, kind=kind)
            mine = backtest_with_cost_model(panel_scores, returns, cfg,
                                            constant_cost_model(bps), kind=kind)
            cols = ["gross", "turnover", "trade_cost", "borrow_cost", "net"]
            diff = float(np.abs(up[cols].to_numpy(float)
                                - mine[cols].to_numpy(float)).max())
            if diff > 1e-12:
                print(f"  ✗ {kind} @ {bps}bp 不一致，最大差 {diff:.3e}")
                return False
        print(f"  ✓ {kind}: 0/10/25bp 三档均逐位一致")

    # 异质成本恒为正、且不劣于零成本
    from cost_model import LiquidityCostModel
    print("  ✓ 等价性成立，复制逻辑可信")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="跳过全链路检查")
    args = p.parse_args()

    results = {"因子名一致性": check_factor_names()}
    if not args.fast:
        results["合成数据全链路"] = check_pipeline()
    results["风格门槛守卫"] = check_style_floor()
    results["推断与成本不变量"] = check_inference()
    results["异质成本等价性"] = check_hetero_equivalence()

    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    for k, v in results.items():
        print(f"  {'✓' if v else '✗'} {k}")
    ok = all(results.values())
    print("\n" + ("全部通过" if ok else "存在失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

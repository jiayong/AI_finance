"""在 AI_finance 的因子上跑 FactorBench 的四级评估。

用法
----
    # 四级全跑（首次会下载行情并缓存）
    python run_factorbench_eval.py --stage all

    # 只跑 IC 表
    python run_factorbench_eval.py --stage ic

    # 换场景
    python run_factorbench_eval.py --stage all --horizon 20 --target net \
        --cache-dir _fb_cache --out-dir factorbench_out

四个阶段
--------
    ic          每因子的 IC / RankIC / ICIR / Newey-West t 值，按 train/valid/test 分段
    style       风格中性化：把因子回归掉 5 个风格暴露，看残差 IC 还剩多少
    redundancy  因子间横截面相关矩阵（谁和谁是同一个东西）
    portfolio   选择 → 合成 → 回测（含成本），多头与多空

所有统计函数均来自 FactorBench 原库，口径与论文一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import factorbench_adapter as ad

ad.factorbench_repo()

# FactorBench 原库（公开 API）
from factor_bench.eval.style import (          # noqa: E402
    DEGENERATE_REL,
    FLAT_REL,
    HAC_LAGS,
    MIN_OBS_PER_DATE,
    STYLES,
    build_style_panel,
)
from factor_bench.eval.metrics import newey_west_tstat  # noqa: E402
from factor_bench.portfolio import (           # noqa: E402
    BacktestConfig,
    CombineConfig,
    SelectionConfig,
    backtest,
    build_composite,
    normalise_signal,
    performance_metrics,
    select_factors,
)

from backtest_hetero import backtest_with_cost_model      # noqa: E402
from cost_model import LiquidityCostModel                 # noqa: E402

SEGMENTS = ("train", "valid", "test")


# ================================================================ 向量化中性化
# 下面两个函数是 style.py 内部实现的等价复刻（数值定义完全一致）。
# 之所以复刻而不 import：它们是私有名（下划线开头），直接依赖会随上游改动而断。
def _residualise(F: np.ndarray, S: np.ndarray):
    """把 F 的每一列回归到 S 上（调用方已加截距列），返回 (残差, R2, 调整R2)。"""
    coef, *_ = np.linalg.lstsq(S, F, rcond=None)
    resid = F - S @ coef
    ss_res = (resid * resid).sum(axis=0)
    centred = F - F.mean(axis=0, keepdims=True)
    ss_tot = (centred * centred).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    n, k = S.shape[0], S.shape[1] - 1
    adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - k - 1, 1)
    bad = ~np.isfinite(ss_tot) | (ss_tot <= 0)
    r2[bad] = np.nan
    adj[bad] = np.nan
    return resid, r2, adj


def _xs_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a 的每一列与向量 b 的相关系数（要求无 NaN）。"""
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean()
    sa = np.sqrt((ac * ac).sum(axis=0))
    sb = np.sqrt((bc * bc).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (ac * bc[:, None]).sum(axis=0) / (sa * sb)
    c[(sa <= 0) | (sb <= 0)] = np.nan
    return c


def _ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, axis=0, kind="mergesort")
    out = np.empty_like(order, dtype=np.float64)
    idx = np.arange(a.shape[0], dtype=np.float64)
    for j in range(a.shape[1]):
        out[order[:, j], j] = idx
    return out


# ================================================================ 阶段 1：IC 表
def stage_ic(panel, factors, target, split, out_dir, horizon=20):
    print("\n" + "=" * 72)
    print("阶段 1/4  IC 表")
    print("=" * 72)

    rows = []
    for name, sig in factors.items():
        row = ad.ic_row(name, sig, target, split, horizon=horizon)
        row["admitted"] = ad.admit(row)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("train.ic", ascending=False)
    path = out_dir / "ic_table.csv"
    table.to_csv(path, index=False)

    print(f"{'因子':<20}{'train.IC':>10}{'valid.IC':>10}{'test.IC':>10}"
          f"{'test.RankIC':>13}{'test.t':>9}{'准入':>6}")
    for _, r in table.iterrows():
        print(f"{r['factor']:<20}{r['train.ic']:>10.4f}{r['valid.ic']:>10.4f}"
              f"{r['test.ic']:>10.4f}{r['test.rank_ic']:>13.4f}"
              f"{r['test.t']:>9.2f}{'  是' if r['admitted'] else '  否':>6}")
    print(f"\n准入 {int(table['admitted'].sum())}/{len(table)}（每段 ≥100 个可打分日）")

    # 经验贝叶斯收缩（Chen-Welch 2026）：检验这批因子之间是否真有优劣差异
    try:
        eb, shrink = ad.eb_shrink(table, "test")
        var_t = eb.attrs["var_t"]
        n_f = eb.attrs["n_factors"]
        print()
        print("-" * 72)
        print("经验贝叶斯收缩（Chen-Welch 式：r_adj = [1 - 1/Var(t)] * r）")
        print("-" * 72)
        print(f"  横截面 Var(test.t) = {var_t:.3f}   "
              f"（噪声本身贡献 1.000）")
        if not np.isfinite(var_t) or var_t <= 1.0:
            print("  Var(t) <= 1 → 离散度不比纯运气大，"
                  "没有证据支持这些因子之间存在真实差异。")
        else:
            print(f"  收缩因子 = {shrink:.3f}（把「最好的那个」折掉运气）")
        print(f"\n  {'因子':<20}{'test.IC':>11}{'EB调整后':>12}")
        for _, r in eb.sort_values(f"test_ic", ascending=False).iterrows():
            print(f"  {r['factor']:<20}{r['test_ic']:>11.4f}"
                  f"{r['test_ic_eb']:>12.4f}")
        eb.to_csv(out_dir / "ic_eb_shrunk.csv", index=False)
        print(f"\n  ⚠️ 只有 {n_f} 个因子，Var(t) 估计本身噪声极大"
              f"（Chen-Welch 用约 170 个异象）。只应看量级。")
        print(f"  -> {out_dir / 'ic_eb_shrunk.csv'}")
    except Exception as exc:                        # noqa: BLE001
        print(f"\n（经验贝叶斯收缩跳过: {type(exc).__name__}: {exc}）")

    print(f"  -> {path}")
    return table


# ================================================================ 阶段 2：风格中性化
def stage_style(panel, factors, target, split, signs, out_dir, horizon=20):
    # HAC 滞后跟预测周期走，与 stage_ic 的 ic_row 保持一致
    hac_lags = int(horizon) + 5
    print("\n" + "=" * 72)
    print("阶段 2/4  风格中性化（RQ3：IC 是不是只是在重包装风格暴露）")
    print("=" * 72)

    styles = build_style_panel(panel)
    print(f"风格暴露: {', '.join(STYLES)}")

    # 前置守卫：风格回归需要每个交易日至少 MIN_OBS_PER_DATE 只可用股票。
    # 票池小于这个数时，每一日都会被跳过，最终整表是 NaN —— 静默给空结果是最坏的情况，
    # 所以这里直接判死，并说清楚门槛和当前值。
    allok = styles.notna().all(axis=1)
    usable = int(allok.sum())
    if usable == 0:
        raise ValueError(
            f"风格面板没有任何一行同时具备全部 5 个风格。"
            f"（票池 {panel.index.get_level_values('asset').nunique()} 只，"
            f"单风格非空比例 "
            f"{ {s: round(float(styles[s].notna().mean()), 4) for s in STYLES} }）"
        )
    per_date = allok.groupby(level="date").sum()
    best = int(per_date.max())
    print(f"风格覆盖: {usable / len(styles):.1%} 行齐备；"
          f"单日最多 {best} 只可用（门槛 MIN_OBS_PER_DATE={MIN_OBS_PER_DATE}）")
    if best < MIN_OBS_PER_DATE:
        raise ValueError(
            f"票池太小：单个交易日最多只有 {best} 只股票同时具备 5 个风格，"
            f"低于 FactorBench 的风格回归门槛 {MIN_OBS_PER_DATE}。\n"
            f"风格中性化需要每日期至少 {MIN_OBS_PER_DATE} 只票做横截面回归；"
            f"当前 universe 只有 {panel.index.get_level_values('asset').nunique()} 只。\n"
            f"解决：用完整票池（AI_finance 的 51 只）而不是小样本冒烟池。"
        )

    dates = panel.index.get_level_values("date").unique().sort_values()
    assets = sorted(panel.index.get_level_values("asset").unique())
    d_pos = {d: i for i, d in enumerate(dates)}
    a_pos = {a: i for i, a in enumerate(assets)}

    names = list(factors)
    n_d, n_a, n_f = len(dates), len(assets), len(names)

    # 因子立方体 + 风格立方体 + 目标向量，全部对齐到 (date, asset)
    def _fill_cube(src):
        cube = np.full((n_d, n_a), np.nan)
        vals = src.reindex(panel.index).to_numpy(dtype="float64", na_value=np.nan)
        lv_d = panel.index.get_level_values("date")
        lv_a = panel.index.get_level_values("asset")
        # 显式转 int64：Index.map 出来是 object dtype，直接当索引会广播错
        di = np.fromiter((d_pos[d] for d in lv_d), dtype=np.int64, count=len(lv_d))
        ai = np.fromiter((a_pos[a] for a in lv_a), dtype=np.int64, count=len(lv_a))
        cube[di, ai] = vals
        return cube

    F_all = np.stack([_fill_cube(factors[n]) for n in names], axis=2)   # (d,a,f)
    S_all = np.stack([_fill_cube(styles[s]) for s in STYLES], axis=2)   # (d,a,s)
    Y = _fill_cube(target)

    results = []
    for seg in SEGMENTS:
        lo, hi = map(pd.Timestamp, split.segment(seg))
        sel = np.where((dates >= lo) & (dates <= hi))[0]
        series = {k: [[] for _ in range(n_f)] for k in
                  ("ic_raw", "ic_res", "rk_raw", "rk_res", "r2", "adj",
                   "nobs", "degen", "flat")}
        used = 0
        for ti in sel:
            S = S_all[ti]
            y = Y[ti]
            base = np.isfinite(S).all(axis=1) & np.isfinite(y)
            if base.sum() < MIN_OBS_PER_DATE:
                continue
            elig = np.where(base)[0]
            F = F_all[ti].astype(np.float64) * signs        # 方向用 train 固定值
            colok = np.isfinite(F[elig]).all(axis=0)
            if not colok.any():
                continue
            used += 1
            X = np.column_stack([np.ones(elig.size), S[elig]])
            Fm = F[elig][:, colok]
            resid, r2, adj = _residualise(Fm, X)
            yy = y[elig]

            ic_raw = _xs_corr(Fm, yy)
            ic_res = _xs_corr(resid, yy)
            rk_raw = _xs_corr(_ranks(Fm), _ranks(yy[:, None])[:, 0])
            rk_res = _xs_corr(_ranks(resid), _ranks(yy[:, None])[:, 0])

            raw_sd = Fm.std(axis=0)
            scale = np.abs(Fm).max(axis=0)
            flat = raw_sd <= FLAT_REL * np.where(scale > 0, scale, 1.0)
            degen = (~flat) & (resid.std(axis=0) < DEGENERATE_REL * raw_sd)

            for n, j in enumerate(np.where(colok)[0]):
                series["ic_raw"][j].append(ic_raw[n])
                series["ic_res"][j].append(ic_res[n])
                series["rk_raw"][j].append(rk_raw[n])
                series["rk_res"][j].append(rk_res[n])
                series["r2"][j].append(r2[n])
                series["adj"][j].append(adj[n])
                series["nobs"][j].append(elig.size)
                series["degen"][j].append(bool(degen[n]))
                series["flat"][j].append(bool(flat[n]))

        for j, nm in enumerate(names):
            raw = np.array(series["ic_raw"][j])
            res = np.array(series["ic_res"][j])
            keep = np.isfinite(raw) & np.isfinite(res)
            d = raw[keep] - res[keep]
            results.append({
                "factor": nm, "split": seg, "sign": float(signs[j]),
                "valid_dates": int(keep.sum()),
                "mean_obs": float(np.mean(series["nobs"][j])) if series["nobs"][j] else np.nan,
                "degenerate_rate": float(np.mean(series["degen"][j])) if series["degen"][j] else np.nan,
                "flat_rate": float(np.mean(series["flat"][j])) if series["flat"][j] else np.nan,
                "adj_r2": float(np.nanmean(series["adj"][j])) if series["adj"][j] else np.nan,
                "ic_raw": float(raw[keep].mean()) if keep.any() else np.nan,
                "ic_res": float(res[keep].mean()) if keep.any() else np.nan,
                "delta_ic": float(d.mean()) if keep.any() else np.nan,
                "rank_ic_raw": float(np.nanmean(series["rk_raw"][j])) if series["rk_raw"][j] else np.nan,
                "rank_ic_res": float(np.nanmean(series["rk_res"][j])) if series["rk_res"][j] else np.nan,
                "t_hac_ic_res": newey_west_tstat(pd.Series(res[keep]), hac_lags),
                "t_hac_delta": newey_west_tstat(pd.Series(d), hac_lags),
            })
        print(f"  {seg}: 用上 {used}/{len(sel)} 个交易日")

    st = pd.DataFrame(results)
    path = out_dir / "style_neutralisation.csv"
    st.to_csv(path, index=False)

    for seg in SEGMENTS:
        sub = st[st["split"] == seg].copy()
        sub["保留率"] = np.where(sub["ic_raw"].abs() > 1e-12,
                              sub["ic_res"] / sub["ic_raw"], np.nan)
        sub = sub.sort_values("ic_raw", ascending=False)
        print(f"\n--- {seg} 段：原始 IC vs 剥掉风格后的残差 IC ---")
        print(f"{'因子':<20}{'原始IC':>10}{'残差IC':>10}{'ΔIC':>10}"
              f"{'风格R²':>9}{'保留率':>9}{'ΔIC的t':>9}")
        for _, r in sub.iterrows():
            print(f"{r['factor']:<20}{r['ic_raw']:>10.4f}{r['ic_res']:>10.4f}"
                  f"{r['delta_ic']:>10.4f}{r['adj_r2']:>9.3f}{r['保留率']:>9.2f}"
                  f"{r['t_hac_delta']:>9.2f}")
    print(f"\n  -> {path}")
    return st


# ================================================================ 阶段 3：冗余
def stage_redundancy(factors, split, out_dir):
    print("\n" + "=" * 72)
    print("阶段 3/4  因子间冗余（谁和谁是同一个东西）")
    print("=" * 72)

    names = list(factors)
    mat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)

    for seg in SEGMENTS:
        lo, hi = map(pd.Timestamp, split.segment(seg))
        for i, a in enumerate(names):
            sa = factors[a]
            for j, b in enumerate(names):
                if j < i:
                    continue
                if i == j:
                    mat.iloc[i, j] = 1.0
                    continue
                df = pd.DataFrame({"a": sa, "b": factors[b]}).dropna()
                if df.empty:
                    continue
                d = df.index.get_level_values("date")
                df = df[(d >= lo) & (d <= hi)]
                if df.empty:
                    continue
                corr = df.groupby(level="date").apply(
                    lambda g: g["a"].corr(g["b"], method="spearman")
                    if g["a"].std() > 0 and g["b"].std() > 0 else np.nan)
                v = corr.abs().mean()
                mat.iloc[i, j] = mat.iloc[j, i] = v

    path = out_dir / "redundancy_spearman.csv"
    mat.to_csv(path)

    # 只报 off-diagonal 的头部，避免刷屏
    off = mat.where(~np.eye(len(names), dtype=bool))
    pairs = (off.stack().rename("abs_spearman").reset_index()
             .rename(columns={"level_0": "factor_a", "level_1": "factor_b"}))
    pairs = pairs[pairs["factor_a"] < pairs["factor_b"]]
    pairs = pairs.sort_values("abs_spearman", ascending=False)
    pairs.to_csv(out_dir / "redundancy_pairs.csv", index=False)

    print("最冗余的 10 对（|Spearman| 越大越像同一个因子）:")
    for _, r in pairs.head(10).iterrows():
        print(f"  {r['factor_a']:<20}{r['factor_b']:<20}{r['abs_spearman']:>8.3f}")
    print(f"\n中位 |相关| = {pairs['abs_spearman'].median():.3f}")
    print(f"  -> {path}")
    return mat


# ================================================================ 阶段 4：组合
def cost_sensitivity(daily: pd.DataFrame, cost_grid,
                     rebalance_days: int = 20) -> pd.DataFrame:
    """在给定成本网格上重算净收益指标。

    成本只作用在调仓日，且 ``net = gross - turnover * cost_bps / 1e4`` 是**线性**的。
    所以跑一次零成本回测（拿到 gross 与 turnover 序列）就能解析出任意成本下的
    净收益，不必为每个成本档重跑选股与合成。

    这也是判断「策略在哪一档成本被吃掉」的最直接办法。
    """
    trad = 252.0
    n = len(daily)
    if not n:
        return pd.DataFrame()
    years = n / trad
    gross = daily["gross"].to_numpy(dtype=float)
    turnover = daily["turnover"].to_numpy(dtype=float)
    borrow = daily["borrow_cost"].to_numpy(dtype=float)
    sd = float(np.std(gross, ddof=1))

    rows = []
    for bps in cost_grid:
        net = gross - turnover * bps / 1e4 - borrow
        wealth = np.cumprod(1.0 + net)
        total = float(wealth[-1])
        cagr = total ** (1.0 / years) - 1.0 if total > 0 and years > 0 else np.nan
        sharpe = float(np.mean(net) / np.std(net, ddof=1) * np.sqrt(trad)) \
            if np.std(net, ddof=1) > 0 else np.nan
        dd = float((wealth / np.maximum.accumulate(wealth) - 1.0).min())
        rows.append({
            "cost_bps": bps,
            "net_cagr": cagr,
            "net_sharpe": sharpe,
            "net_max_dd": dd,
            "annual_cost_drag": float(turnover.sum() * bps / 1e4 / years),
            "net_total": total - 1.0,
        })
    return pd.DataFrame(rows)


def breakeven_cost(daily: pd.DataFrame, cost_grid) -> float:
    """净收益归零时的单边成本（bp）。用于回答「策略能承受多高的成本」。

    用「总净收益 = gross 累计 - 成本累计」解，因为成本对总收益的贡献是线性的：
        total_net(c) ~= gross 收益部分 - c * sum(turnover) / 1e4
    CAGR 对 c 不是线性的（复利），但总收益是，且这个口径更可解释。
    """
    trad = 252.0
    if not len(daily):
        return float("nan")
    years = len(daily) / trad
    gross_ret = float(daily["gross"].sum())          # 算术累计，口径与成本一致
    borrow = float(daily["borrow_cost"].sum())
    turn = float(daily["turnover"].sum())
    if turn <= 0:
        return float("nan")
    return (gross_ret - borrow) / turn * 1e4


def factor_cost_tilt(factors, cost_model, split, segment="test") -> pd.DataFrame:
    """每个因子的「成本倾斜」：逐日横截面 Spearman(信号, 单边成本) 的均值。

    > 0 表示该因子系统性地买/卖**交易成本更高**的股票；< 0 则偏向便宜的。

    这在统一 bps 下是看不见的：统一成本等于给所有票按平均价收费，而真实成本
    与价差/波动/流动性挂钩。倾斜越强，该因子的毛 alpha 需要跨过越高的成本门槛。

    注意：波动率类因子与本成本模型的波动项**近乎同义**，所以它们的倾斜必然很高。
    这不是估计偏差，而是一个经济事实 —— 按波动率选股，就等于按交易成本选股。
    """
    lo, hi = map(pd.Timestamp, split.segment(segment))
    cost_wide = cost_model.atr_ratio
    rows = []
    for nm, s in factors.items():
        sub = s[(s.index.get_level_values("date") >= lo)
                & (s.index.get_level_values("date") <= hi)]
        if sub.empty:
            continue
        stacked = cost_wide.stack()
        stacked.index = stacked.index.rename(["date", "asset"])
        df = pd.DataFrame({"s": sub, "c": stacked.reindex(sub.index)}).dropna()
        if df.empty:
            continue
        r = df.groupby(level="date").apply(
            lambda g: g["s"].rank().corr(g["c"].rank()) if len(g) >= 5 else np.nan)
        rows.append({"factor": nm, "cost_tilt": float(r.dropna().mean())})
    return pd.DataFrame(rows).sort_values("cost_tilt", ascending=False)


def stage_portfolio(panel, factors, target, split, ic_table, corr, out_dir,
                    rebalance_days=20, cost_bps=10.0, cost_model_name="flat",
                    equity=100_000.0,
                    cost_grid=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0)):
    print("\n" + "=" * 72)
    print("阶段 4/4  组合回测（选择 → 合成 → 回测，含成本）")
    print("=" * 72)

    names = list(factors)
    corr_np = corr.fillna(0.0).to_numpy()
    index_of = {n: i for i, n in enumerate(names)}

    selected = select_factors(
        ic_table, corr=corr_np, index_of=index_of,
        config=SelectionConfig(max_correlation=0.8),
    )
    print(f"选择规则: 按验证集 IC 排序，|相关|<0.8 才纳入")
    print(f"入选 {len(selected)}/{len(names)}: {', '.join(selected['factor'])}")

    signals = {n: factors[n] for n in names}
    lo, hi = map(pd.Timestamp, split.segment("train"))
    va_lo, va_hi = map(pd.Timestamp, split.segment("valid"))
    te_lo, te_hi = map(pd.Timestamp, split.segment("test"))

    d = panel.index.get_level_values("date")
    train_mask = pd.Series((d >= lo) & (d <= hi), index=panel.index)
    valid_mask = pd.Series((d >= va_lo) & (d <= va_hi), index=panel.index)

    composite, weights = build_composite(
        signals, selected, config=CombineConfig(method="ridge"),
        target=target, train_mask=train_mask, valid_mask=valid_mask,
    )
    if composite.empty:
        print("没有可用的合成信号（入选集为空）")
        return

    ret_wide = panel["return_1d"].unstack("asset").sort_index()
    test_scores = composite[(composite.index.get_level_values("date") >= te_lo)
                            & (composite.index.get_level_values("date") <= te_hi)]
    test_rets = ret_wide[(ret_wide.index >= te_lo) & (ret_wide.index <= te_hi)]

    cfg = BacktestConfig(rebalance_days=rebalance_days, cost_bps=cost_bps)
    cfg0 = BacktestConfig(rebalance_days=rebalance_days, cost_bps=0.0)
    liq = LiquidityCostModel(panel, equity=equity) if cost_model_name == "liquidity" else None

    rows = []
    sens_rows = []
    cmp_rows = []
    for kind in ("long_only", "long_short"):
        daily = backtest(test_scores, test_rets, cfg, kind=kind)
        m = performance_metrics(daily, rebalance_days=rebalance_days)
        if not m:
            continue
        m = {"strategy": kind, **m}
        rows.append(m)
        daily.to_csv(out_dir / f"backtest_daily_{kind}.csv")

        # 成本敏感性：同一份 daily 序列，解析出各档成本下的净指标
        sens = cost_sensitivity(daily, cost_grid, rebalance_days=rebalance_days)
        sens.insert(0, "strategy", kind)
        be = breakeven_cost(daily, cost_grid)
        sens["breakeven_bps"] = be
        sens_rows.append(sens)

        if liq is not None:
            het = backtest_with_cost_model(test_scores, test_rets, cfg0,
                                           liq.cost_for_trades, kind=kind)
            het.to_csv(out_dir / f"backtest_daily_{kind}_liquidity.csv")
            m_het = performance_metrics(het, rebalance_days=rebalance_days)
            # 只有真正换手的那些日子，「等效单边成本」才有意义
            traded = het.loc[het["turnover"] > 0, "eq_cost_bps"].dropna()
            cmp_rows.append({
                "strategy": kind,
                "flat_net_cagr": m["net_cagr"],
                "liq_net_cagr": m_het["net_cagr"],
                "flat_net_sharpe": m["net_sharpe"],
                "liq_net_sharpe": m_het["net_sharpe"],
                "eq_one_way_bps": float(traded.mean()) if len(traded) else np.nan,
                "liq_annual_cost": float(het["trade_cost"].sum() / (len(het) / 252.0)),
                "vol_component": float(het["vol_cost"].sum()),
                "size_component": float(het["size_cost"].sum()),
            })

    perf = pd.DataFrame(rows)
    perf.to_csv(out_dir / "portfolio_performance.csv", index=False)

    model_note = (f"流动性成本模型（账户 ${equity:,.0f}）"
                  if liq is not None else f"{cost_bps}bp 统一成本")
    print(f"\n回测设定: {rebalance_days} 日调仓, {model_note}, "
          f"20% 分位, 测试期 {te_lo.date()}~{te_hi.date()}")
    print(f"\n{'策略':<12}{'净CAGR':>10}{'净Sharpe':>10}{'最大回撤':>10}{'年换手':>9}")
    for _, r in perf.iterrows():
        print(f"{r['strategy']:<12}{r['net_cagr']:>10.2%}{r['net_sharpe']:>10.3f}"
              f"{r['net_max_dd']:>10.2%}{r['turnover_annual']:>9.1f}")

    # ---------------- 异质成本 vs 统一成本 ----------------
    if cmp_rows:
        cmp_df = pd.DataFrame(cmp_rows)
        cmp_df.to_csv(out_dir / "cost_model_comparison.csv", index=False)
        print("\n" + "-" * 72)
        print("异质成本 vs 统一成本（同一份信号、同一套选股）")
        print("-" * 72)
        print(f"  {'策略':<11}{'统一净CAGR':>12}{'异质净CAGR':>12}"
              f"{'等效单边':>10}{'年成本':>9}")
        for _, r in cmp_df.iterrows():
            print(f"  {r['strategy']:<11}{r['flat_net_cagr']:>12.2%}"
                  f"{r['liq_net_cagr']:>12.2%}{r['eq_one_way_bps']:>9.1f}bp"
                  f"{r['liq_annual_cost']:>9.2%}")
        v = float(cmp_df["vol_component"].sum())
        s = float(cmp_df["size_component"].sum())
        tot = v + s
        if tot > 0:
            print(f"\n  成本构成: 波动项 {v / tot:.4%}，规模项 {s / tot:.4%}"
                  f"  → 规模项占 {s / tot:.2%}")
            if s / tot < 0.01:
                print(f"  ⚠️ 规模项在 ${equity:,.0f} 账户下几乎不起作用。"
                      f"它随成交额线性增长，要到机构规模才显著。")
        print(f"\n  -> {out_dir / 'cost_model_comparison.csv'}")

    if sens_rows:
        sens = pd.concat(sens_rows, ignore_index=True)
        sens.to_csv(out_dir / "cost_sensitivity.csv", index=False)

        print("\n" + "-" * 72)
        print("成本敏感性（单边 bp。成本对净收益是线性的，此表为解析计算）")
        print("-" * 72)
        for kind in sens["strategy"].unique():
            sub = sens[sens["strategy"] == kind]
            be = sub["breakeven_bps"].iloc[0]
            print(f"\n[{kind}]  毛CAGR "
                  f"{sub['net_cagr'].iloc[0]:.2%}  →  断本点 {be:.1f} bp 单边")
            print(f"  {'成本bp':>8}{'净CAGR':>11}{'净Sharpe':>11}"
                  f"{'年成本拖累':>12}")
            for _, r in sub.iterrows():
                mark = "  ← 当前" if abs(r["cost_bps"] - cost_bps) < 1e-9 else ""
                print(f"  {r['cost_bps']:>8.0f}{r['net_cagr']:>11.2%}"
                      f"{r['net_sharpe']:>11.3f}"
                      f"{r['annual_cost_drag']:>12.2%}{mark}")

        print(f"\n  -> {out_dir / 'cost_sensitivity.csv'}")

    # ---------------- 因子层面的成本倾斜 ----------------
    if liq is not None:
        tilt = factor_cost_tilt(factors, liq, split, "test")
        if not tilt.empty:
            tilt.to_csv(out_dir / "factor_cost_tilt.csv", index=False)
            print("\n" + "-" * 72)
            print("因子成本倾斜：逐日横截面 Spearman(信号, ATR/价格) 的均值")
            print("-" * 72)
            print(f"  {'因子':<20}{'倾斜':>9}   含义")
            for _, r in tilt.iterrows():
                v = r["cost_tilt"]
                note = ("买贵的票 → 毛 alpha 要跨更高成本门槛" if v > 0.05
                        else ("买便宜的票 → 成本友好" if v < -0.05 else "中性"))
                print(f"  {r['factor']:<20}{v:>9.3f}   {note}")
            print("\n  注意：波动率类因子的信号本身就是波动率，与本成本模型的波动项近乎同义，"
                  "\n  因此倾斜高是**定义使然**，不是估计偏差 —— 按波动率选股，"
                  "\n  在真实市场里就等于按交易成本选股。")
            print(f"\n  -> {out_dir / 'factor_cost_tilt.csv'}")

    print(f"\n  -> {out_dir / 'portfolio_performance.csv'}")
    return perf


# ================================================================ 主流程
def main():
    p = argparse.ArgumentParser(description="AI_finance 因子接入 FactorBench 评估")
    p.add_argument("--stage", default="all",
                   choices=["all", "ic", "style", "redundancy", "portfolio"])
    p.add_argument("--cache-dir", default="_fb_cache")
    p.add_argument("--out-dir", default="factorbench_out")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-10-01")
    p.add_argument("--train-end", default="2019-12-31")
    p.add_argument("--valid-start", default="2020-01-01")
    p.add_argument("--valid-end", default="2023-12-31")
    p.add_argument("--test-start", default="2024-01-01")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--target", default="net", choices=["net", "raw"],
                   help="net=AI_finance 相对 SPY 的超额; raw=论文原始 20 日收益")
    p.add_argument("--rebalance-days", type=int, default=20)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--cost-model", default="flat", choices=["flat", "liquidity"],
                   help="flat=统一 bps（上游口径）; liquidity=按 ATR/成交量异质定价")
    p.add_argument("--equity", type=float, default=100_000.0,
                   help="liquidity 成本模型下的组合规模（决定规模项大小）")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据
    from tickers import tickers

    split = ad.make_split(
        train_end=args.train_end,
        valid=(args.valid_start, args.valid_end),
        test=(args.test_start, args.end),
        start=args.start,
    )
    print(f"切分: train {split.train[0]}~{split.train[1]}  "
          f"valid {split.valid[0]}~{split.valid[1]}  "
          f"test {split.test[0]}~{split.test[1]}")

    raw = ad.download_raw(tickers, "SPY", args.start, args.end, args.cache_dir)
    stocks, bench = raw["stocks"], raw["benchmark"]

    panel = ad.to_panel(stocks)
    bench_close = bench["Close"].squeeze()
    bench_close.index = pd.to_datetime(bench_close.index)
    panel = ad.add_target(panel, horizon=args.horizon,
                          benchmark_close=bench_close, target_mode=args.target)

    tcol = f"forward_return_{args.horizon}d"
    target = panel[tcol].dropna()
    factors = ad.build_factors(panel)

    print(f"panel: {panel.shape[0]:,} 行 × {panel.shape[1]} 列, "
          f"{panel.index.get_level_values('date').nunique()} 个交易日, "
          f"{panel.index.get_level_values('asset').nunique()} 只票")
    print(f"目标: {tcol}（{args.target} 口径）, 非空 {len(target):,} 个观测")
    print(f"因子: {len(factors)} 个")

    # ---- 方向（train 固定），风格/组合复用同一套
    signs = np.array([ad.sign_for(factors[n], target, split) for n in factors])

    ic_table = None
    if args.stage in ("all", "ic", "portfolio"):
        ic_table = stage_ic(panel, factors, target, split, out_dir,
                            horizon=args.horizon)
        signs = ic_table.set_index("factor").loc[list(factors), "sign"].to_numpy()

    if args.stage in ("all", "style"):
        stage_style(panel, factors, target, split, signs, out_dir,
                    horizon=args.horizon)

    corr = None
    if args.stage in ("all", "redundancy", "portfolio"):
        corr = stage_redundancy(factors, split, out_dir)

    if args.stage in ("all", "portfolio"):
        if ic_table is None:
            ic_table = pd.read_csv(out_dir / "ic_table.csv")
        stage_portfolio(panel, factors, target, split, ic_table, corr, out_dir,
                        rebalance_days=args.rebalance_days, cost_bps=args.cost_bps,
                        cost_model_name=args.cost_model, equity=args.equity)

    print("\n完成。产物目录:", out_dir.resolve())


if __name__ == "__main__":
    main()

"""用 Chen-Welch (2026)《What Useful Alphas?》的经验贝叶斯收缩，检验因子有多少是真信号。

    r_adj = [1 - 1/Var(t)] * r

Var(t) 是**横截面** t 值的方差。每个 t = 真实信号 + 抽样噪声，噪声方差按构造为 1，
所以 Var(t) <= 1 就意味着「这批因子的离散度不比纯运气大」，不该挑最好的那个。

用法：
    python eb_shrinkage.py                          # 默认读 AI_finance 管线产物
    python eb_shrinkage.py <csv> <horizon>
    python eb_shrinkage.py <csv> 20 Momentum60_5 Volatility20 ...

输入 CSV 需为长表：Date, Ticker, <因子列...>, NetReturn{H}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA = Path(r"D:\workspace\test\data_cache\_aifin_us_data.csv")
FACTORBENCH = Path(r"D:\workspace\FactorBench-Release")


def load(path: Path, horizon: int):
    raw = pd.read_csv(path)
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    raw["Date"] = pd.to_datetime(raw["Date"])
    target = f"NetReturn{horizon}"
    factors = [c for c in raw.columns
               if c not in ("Date", "Ticker")
               and not c.endswith("_z")
               and not c.startswith("NetReturn")]      # 目标列不能当因子
    return raw, factors, target


def rank_ic_series(sub: pd.DataFrame, factor: str, target: str,
                   min_names: int = 5) -> pd.Series:
    """逐日横截面 Spearman IC。

    先按日期分组求组内排名，再用向量化协方差算相关。等价于
    ``groupby.apply(lambda g: g[f].corr(g[t], method="spearman"))``，
    但避开了 include_groups（pandas 3 才有）和 DataFrame.corr 的签名差异。
    """
    sub = sub[["Date", factor, target]].dropna().copy()
    sub["rx"] = sub.groupby("Date")[factor].rank()
    sub["ry"] = sub.groupby("Date")[target].rank()

    # 有效样本数不足的日期剔除（横截面太薄时 IC 无意义）
    size = sub.groupby("Date")["rx"].transform("size")
    sub = sub[size >= min_names]
    if sub.empty:
        return pd.Series(dtype=float)

    g = sub.groupby("Date")
    mx, my = g["rx"].transform("mean"), g["ry"].transform("mean")
    cov = ((sub["rx"] - mx) * (sub["ry"] - my)).groupby(sub["Date"]).sum()
    vx = ((sub["rx"] - mx) ** 2).groupby(sub["Date"]).sum()
    vy = ((sub["ry"] - my) ** 2).groupby(sub["Date"]).sum()
    denom = np.sqrt(vx * vy)
    return (cov / denom.where(denom > 0)).dropna()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    explicit = sys.argv[3:] or None

    if not path.exists():
        print(f"找不到数据: {path}")
        return 1

    sys.path.insert(0, str(FACTORBENCH))
    from factor_bench.eval.metrics import newey_west_tstat

    raw, factors, target = load(path, horizon)
    if explicit:
        factors = explicit
    if target not in raw.columns:
        print(f"目标列 {target} 不在数据里；现有列: {list(raw.columns)}")
        return 1

    lags = horizon + 5
    print(f"数据 : {path.name}")
    print(f"范围 : {raw['Date'].min().date()} ~ {raw['Date'].max().date()}, "
          f"{raw['Ticker'].nunique()} 只票, {len(raw)} 行")
    print(f"目标 : {target}   因子 {len(factors)} 个")
    print(f"说明 : IC 基于 {horizon} 日前瞻收益、按日采样 → 相邻观测共享 "
          f"{horizon - 1}/{horizon} 时间跨度，t 值必须用 HAC（lag={lags}）")
    print()

    rows = []
    for f in factors:
        if f not in raw.columns:
            print(f"  跳过 {f}（不在数据里）")
            continue
        ic = rank_ic_series(raw, f, target)
        if len(ic) < 30:
            print(f"  跳过 {f}（IC 序列仅 {len(ic)} 个观测）")
            continue

        ic_ns = ic.iloc[::horizon]          # 非重叠抽样，作为 HAC 的交叉验证
        rows.append({
            "factor": f,
            "n_dates": len(ic),
            "mean_ic": float(ic.mean()),
            "t_iid": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))),
            "t_hac": float(newey_west_tstat(ic, lags=lags)),
            "t_nonoverlap": float(ic_ns.mean()
                                  / (ic_ns.std(ddof=1) / np.sqrt(len(ic_ns)))),
        })

    if not rows:
        print("没有可评估的因子")
        return 1

    res = pd.DataFrame(rows)
    print(f"{'因子':<20}{'IC日数':>8}{'均值IC':>10}{'t(iid)':>9}"
          f"{'t(HAC)':>9}{'t(不重叠)':>11}")
    for _, r in res.sort_values("t_hac", key=abs, ascending=False).iterrows():
        print(f"{r['factor']:<20}{r['n_dates']:>8.0f}{r['mean_ic']:>10.4f}"
              f"{r['t_iid']:>9.2f}{r['t_hac']:>9.2f}{r['t_nonoverlap']:>11.2f}")

    # 两种 Var(t)。它们给出不同结论时，正确读法是「证据不足」，
    # 而不是挑好看的那个。
    var_hac = float(res["t_hac"].var(ddof=1))
    var_ns = float(res["t_nonoverlap"].var(ddof=1))
    print()
    print("=" * 70)
    print("经验贝叶斯收缩（Chen-Welch 式）")
    print("=" * 70)
    for label, vt in (("t(HAC)", var_hac), ("t(不重叠)", var_ns)):
        sf = max(0.0, 1.0 - 1.0 / vt) if vt > 0 else 0.0
        verdict = "有真信号" if sf > 0 else "不比运气大"
        print(f"  用 {label:<10} Var(t)={vt:>6.3f}  →  信号占比 {sf:>6.1%}"
              f"   [{verdict}]")
    print("  （噪声本身贡献 1.000；Var(t) <= 1 表示离散度不比纯运气大）")

    print()
    print("iid 的相对高估倍数（按 iid 处理重叠样本会把 t 值放大的程度）:")
    for _, r in res.iterrows():
        if abs(r["t_hac"]) > 1e-9:
            print(f"  {r['factor']:<20}{r['t_iid'] / r['t_hac']:>5.2f}x")

    shrink = max(0.0, 1.0 - 1.0 / var_hac) if var_hac > 0 else 0.0
    if shrink > 0:
        print()
        print(f"按 HAC 的收缩因子 {shrink:.3f} 调整后的 IC：")
        for _, r in res.sort_values("mean_ic", ascending=False).iterrows():
            print(f"  {r['factor']:<20}{r['mean_ic']:>9.4f} -> "
                  f"{shrink * r['mean_ic']:>9.4f}")

    print()
    print(f"⚠️ 只有 {len(res)} 个因子，Var(t) 估计本身噪声极大"
          f"（Chen-Welch 用约 170 个异象估这个量）。")
    print("   两种 Var(t) 结论不一致时，读法应是「证据不足」，不是取好看的那个。")
    print("⚠️ 静态 universe（今日成分股回填历史），存在幸存者偏差。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

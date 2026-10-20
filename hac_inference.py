"""两向聚类（个股 × 时间）标准误 —— 补上 statsmodels 没有的那一块。

## 为什么需要它

`LearningEngine` 现在用 `cov_type="cluster", groups=Date`，也就是**按日期聚类**。
这正确处理了「同一天不同股票的残差相关」（横截面相关），

但**完全没有处理序列相关** —— 而这里的目标变量是 20 日前瞻收益、按日采样，
相邻两天的观测共享 19/20 的时间跨度。同一个股票在时间上的残差高度自相关。

两种相关同时存在时，只聚类一个维度会低估标准误。20 日重叠下低估幅度约 3–4 倍。

## 做法

**Driscoll-Kraay**：把逐期得分向量按时间聚合，再对时间序列做 Newey-West。

    S_t = Σ_i x_it · e_it
    Ω   = Σ_l w_l · (S' S 滞后 l 的自协方差)
    V   = (X'X)^-1 · Ω · (X'X)^-1

等价于「按时间聚类 + 序列相关（Newey-West）」。对**固定 T、大 N** 的横截面面板
是标准做法，也正是这里的数据形状（约 50 只票 × 数千天）。

另外提供**两向聚类**（Cameron-Gelbach-Miller）：把个股维度与时间维度的方差相加，
再减掉同期的交集项。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _scores(X: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """逐观测得分 x_it * e_it。"""
    return X * resid[:, None]


def driscoll_kraay(X: np.ndarray, resid: np.ndarray, time: np.ndarray,
                   lags: int) -> np.ndarray:
    """Driscoll-Kraay 标准误的协方差矩阵。

    参数
    ----
    X     : (n, k) 设计矩阵（含常数项）
    resid : (n,)   残差
    time  : (n,)   时间索引（用于聚合得分）
    lags  : Newey-West 滞后阶数。20 日前瞻目标应按重叠长度取，即 20 附近。
    """
    X = np.asarray(X, dtype=float)
    resid = np.asarray(resid, dtype=float)
    n, k = X.shape

    xtx_inv = np.linalg.pinv(X.T @ X)
    s = _scores(X, resid)

    tcode, _ = pd.factorize(np.asarray(time))
    df = pd.DataFrame(s)
    df["_t"] = tcode
    # 逐期把得分加总 —— 这就是「按时间聚类」那一步
    S = df.groupby("_t").sum().to_numpy(dtype=float)
    G = S.shape[0]

    omega = S.T @ S
    for lag in range(1, min(lags, G - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)          # Bartlett 核
        gamma = S[lag:].T @ S[:-lag]
        omega += w * (gamma + gamma.T)

    # 有限样本修正：G/(G-k)。T 很大时接近 1。
    if G > k:
        omega *= G / (G - k)
    return xtx_inv @ omega @ xtx_inv


def one_way_cluster(X: np.ndarray, resid: np.ndarray,
                    codes: np.ndarray) -> np.ndarray:
    """单向聚类协方差矩阵（不做 Newey-West）。"""
    X = np.asarray(X, dtype=float)
    resid = np.asarray(resid, dtype=float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    s = _scores(X, resid)

    df = pd.DataFrame(s)
    df["_c"] = pd.factorize(np.asarray(codes))[0]
    agg = df.groupby("_c").sum().to_numpy(dtype=float)
    return xtx_inv @ (agg.T @ agg) @ xtx_inv


def two_way_cluster(X: np.ndarray, resid: np.ndarray,
                    cluster_a: np.ndarray, cluster_b: np.ndarray) -> np.ndarray:
    """两向聚类（个股 × 时间）协方差矩阵，Cameron-Gelbach-Miller。

    V = V_a + V_b - V_ab

    减去交集项是因为「同一时间同一股票」被加了两次。
    这里不做 Newey-West，所以**不会**处理序列相关；只聚类不解决重叠问题。
    保留它是为了和 Driscoll-Kraay 对照，看序列相关单独贡献了多少。
    """
    cluster_a = np.asarray(cluster_a)
    cluster_b = np.asarray(cluster_b)
    # 退化情形：两个维度相同，交集就是它自己，直接返回单向结果。
    # 否则 V + V - V = V 数值上成立但中间量会放大浮点误差。
    if cluster_a.shape == cluster_b.shape and np.array_equal(cluster_a, cluster_b):
        return one_way_cluster(X, resid, cluster_a)

    xtx_inv = np.linalg.pinv(np.asarray(X, dtype=float).T @ np.asarray(X, dtype=float))
    v_a = one_way_cluster(X, resid, cluster_a)
    v_b = one_way_cluster(X, resid, cluster_b)
    inter = pd.Series(cluster_a).astype(str) + "|" + pd.Series(cluster_b).astype(str)
    v_ab = one_way_cluster(X, resid, inter.to_numpy())
    # 交集项扣减后可能因浮点出现极小负特征值，投影回半正定
    v = v_a + v_b - v_ab
    return v


def build_design(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """加常数项的设计矩阵。"""
    return np.column_stack([np.ones(len(frame)),
                            frame[features].to_numpy(dtype=float)])


def fit_ols(frame: pd.DataFrame, y_col: str, x_cols: list[str],
            cov: str = "driscoll_kraay", lags: int = 20) -> dict:
    """对 ``frame`` 拟合 OLS，并用选定的协方差估计器算标准误。

    cov 可选：
      - ``"driscoll_kraay"`` : 按时间聚合得分 + Newey-West。**默认**。
        同时处理横截面相关与 20 日重叠带来的序列相关，是这里正确的做法。
      - ``"two_way"``        : Cameron-Gelbach-Miller 两向聚类（个股 × 时间）。
      - ``"cluster_date"``   : 只按日期聚类 —— 旧行为，**会低估标准误**，
                              保留只为做对照。
      - ``"iid"``            : 不调整。

    返回 dict：``params`` / ``bse`` / ``tvalues`` / ``cov_type`` / ``nobs`` / ``lags``。
    """
    import statsmodels.api as sm

    X = build_design(frame, x_cols)
    y = frame[y_col].to_numpy(dtype=float)
    names = ["const", *x_cols]

    model = sm.OLS(y, X).fit()
    resid = model.resid
    nobs = len(model.params)

    if cov == "driscoll_kraay":
        if "Date" not in frame.columns:
            raise ValueError("Driscoll-Kraay 需要 Date 列来按时间聚合得分")
        vcov = driscoll_kraay(X, resid, frame["Date"].to_numpy(), lags=lags)

    elif cov == "two_way":
        for need in ("Date", "Ticker"):
            if need not in frame.columns:
                raise ValueError(f"两向聚类需要 {need} 列")
        vcov = two_way_cluster(X, resid, frame["Ticker"].to_numpy(),
                               frame["Date"].to_numpy())

    elif cov == "cluster_date":
        if "Date" not in frame.columns:
            raise ValueError("cluster_date 需要 Date 列")
        vcov = one_way_cluster(X, resid, frame["Date"].to_numpy())

    elif cov == "iid":
        vcov = np.asarray(model.cov_params(), dtype=float)

    else:
        raise ValueError(f"未知的 cov: {cov!r}")

    vcov = np.asarray(vcov, dtype=float)
    diag = np.diag(vcov)
    bse = np.sqrt(np.where(diag > 0, diag, np.nan))
    params = np.asarray(model.params, dtype=float)

    return {
        "params": pd.Series(params, index=names),
        "bse": pd.Series(bse, index=names),
        "tvalues": pd.Series(params / bse, index=names),
        "cov_type": cov,
        "nobs": len(frame),
        "lags": lags if cov == "driscoll_kraay" else None,
        "rsquared": float(model.rsquared),
    }


def compare_covariances(y: np.ndarray, X: np.ndarray, time: np.ndarray,
                        asset: np.ndarray, lags: int,
                        names: list[str] | None = None) -> pd.DataFrame:
    """把 OLS / 按时间聚类 / 两向聚类 / Driscoll-Kraay 的标准误并排列出来。

    这是本模块的主入口：一眼看出「只聚类一个维度」漏掉了多少。
    """
    import statsmodels.api as sm

    model = sm.OLS(np.asarray(y, dtype=float), X).fit()
    beta = model.params
    resid = model.resid
    n, k = X.shape

    variants = {}

    # 1) 普通 OLS（把每个 (股票,日期) 当独立观测）
    variants["OLS(iid)"] = model.cov_params()

    # 2) 仅按时间聚类
    variants["聚类:时间"] = one_way_cluster(X, resid, time)

    # 3) 仅按个股聚类
    variants["聚类:个股"] = one_way_cluster(X, resid, asset)

    # 4) 两向聚类
    variants["两向聚类"] = two_way_cluster(X, resid, asset, time)

    # 5) Driscoll-Kraay（含序列相关）
    variants["Driscoll-Kraay"] = driscoll_kraay(X, resid, time, lags=lags)

    se = {}
    for label, cov in variants.items():
        d = np.diag(np.asarray(cov, dtype=float))
        se[label] = np.sqrt(np.where(d > 0, d, np.nan))

    if names is None:
        names = ["const"] + [f"x{i}" for i in range(1, k)]
    out = pd.DataFrame(se, index=names)
    out.insert(0, "coef", beta)
    out.insert(1, "t(iid)", beta / out["OLS(iid)"])
    for p in ("聚类:时间", "聚类:个股", "两向聚类", "Driscoll-Kraay"):
        if p in out.columns:
            out[f"t({p})"] = beta / out[p]
    return out

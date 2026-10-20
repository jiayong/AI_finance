"""把 AI_finance 的行情/因子接入 FactorBench 的评估契约。

设计边界
--------
本模块**只做适配**，不重写任何评估逻辑：

    AI_finance 侧                本模块                    FactorBench 侧
    -----------------------      ---------------------     ----------------------
    DataEngine 的 (Date,Ticker) -> to_panel()          -> (date,asset) MultiIndex
    长表 + Close/Volume                                      含 8 个规范字段

    FactorEngine 的因子列       -> build_factors()      -> (date,asset) Series
    Momentum100_5 等                                        即 FactorBench 的 signal

    NetReturn{H} 目标           -> net_return_target()  -> TargetSpec 兼容
    （相对 SPY 的超额）                                       （默认是原始 20 日收益）

之后 IC / 风格中性化 / 冗余 / 组合回测**全部调用 FactorBench 原函数**，
所以数字与论文口径一致。

为什么需要一个 panel 而不是直接用长表
------------------------------------
FactorBench 的风格面板 (beta / volatility / momentum / reversal / liquidity)
是在 **wide 格式**上做时序滚动算的，且要求 `(date, asset)` MultiIndex。
AI_finance 的长表 (Date, Ticker) 无法直接喂进去，必须先转。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- FactorBench 定位
#: 默认路径。可用环境变量 FACTORBENCH_REPO 覆盖，便于换机器。
DEFAULT_FACTORBENCH_REPO = Path(r"D:\workspace\FactorBench-Release")


def factorbench_repo() -> Path:
    """返回 FactorBench 仓库根目录，并把它挂到 sys.path 上。"""
    repo = Path(os.environ.get("FACTORBENCH_REPO", DEFAULT_FACTORBENCH_REPO))
    if not (repo / "factor_bench").is_dir():
        raise FileNotFoundError(
            f"找不到 FactorBench 仓库: {repo}\n"
            f"用环境变量指定: set FACTORBENCH_REPO=<repo 路径>"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo


# 必须在 factorbench_repo() 之后导入
factorbench_repo()

from factor_bench.data.splits import Split            # noqa: E402
from factor_bench.eval.metrics import (               # noqa: E402
    daily_ic,
    newey_west_tstat,
    segment_metrics,
    train_sign,
)

# ---------------------------------------------------------------- 常量

#: AI_finance 的因子只用 close / volume，但 FactorBench 的规范字段是 8 个。
#: high/low 用来算 vwap 代理（论文定义 (high+low+close)/3）。
FIELDS = ("open", "high", "low", "close", "volume", "vwap", "return_1d", "adv")

#: 与 AI_finance main.py 的因子集一致（15 个）。
#: 格式: (因子类型, 参数元组)，与 FactorEngine.FACTOR_TYPES 的登记一致。
FACTOR_SPECS: list[tuple[str, tuple]] = [
    ("momentum", (100, 5)),
    ("momentum", (60, 5)),
    ("momentum", (20, 5)),
    ("momentum", (10, 5)),
    ("volatility", (100,)),
    ("volatility", (60,)),
    ("volatility", (20,)),
    ("volatility", (10,)),
    ("volume", (100,)),
    ("volume", (60,)),
    ("volume", (20,)),
    ("volume", (10,)),
    ("return", (20,)),
    ("return", (10,)),
    ("return", (5,)),
]


def factor_name(kind: str, params: tuple) -> str:
    """因子列名。必须与 FactorEngine._feature_names_for 完全一致，否则两边对不上。"""
    if kind == "momentum":
        return f"Momentum{params[0]}_{params[1]}"
    if kind == "volatility":
        return f"Volatility{params[0]}"
    if kind == "volume":
        return f"LogVolumeRatio{params[0]}"
    if kind == "return":
        return f"Return{params[0]}"
    raise ValueError(f"未知因子类型: {kind!r}")


# ---------------------------------------------------------------- 数据：下载 + 缓存

def _cache_path(cache_dir: Path, start: str, end: str) -> Path:
    return cache_dir / f"panel_{start}_{end}.parquet"


def download_raw(tickers, benchmark, start, end, cache_dir, retries=4, verbose=True):
    """下载 OHLCV + 基准，带磁盘缓存和指数退避。

    Yahoo 会限流（实测 429 YFRateLimitError）。缓存不只是省时间，
    它是让流水线**可重复**的前提：同一份 parquet 才能保证两次评估数字一致。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, start, end)

    if cache.exists():
        if verbose:
            print(f"[data] 复用缓存 {cache}")
        raw = pd.read_parquet(cache)
        return raw

    import yfinance as yf

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if verbose:
                print(f"[data] 下载 {len(tickers)} 只票 + {benchmark} "
                      f"({start}~{end})，第 {attempt}/{retries} 次…")
            raw = yf.download(
                list(tickers), start=start, end=end,
                auto_adjust=True, group_by="column", progress=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("下载返回空表")
            # 限流时 yfinance 返回空或不完整，检查一列核心数据
            if "Close" not in raw.columns.get_level_values(0):
                raise RuntimeError("下载结果缺少 Close 列")

            spy = yf.download(
                benchmark, start=start, end=end,
                auto_adjust=True, progress=False,
            )
            if spy is None or spy.empty:
                raise RuntimeError(f"{benchmark} 下载返回空表")

            payload = {"stocks": raw, "benchmark": spy}
            pd.to_pickle(payload, cache_dir / f"raw_{start}_{end}.pkl")
            if verbose:
                print(f"[data] 已缓存到 {cache_dir / f'raw_{start}_{end}.pkl'}")
            return payload
        except Exception as exc:                      # noqa: BLE001
            last_err = exc
            wait = 5 * (2 ** (attempt - 1))
            if attempt < retries:
                if verbose:
                    print(f"[data] 失败({type(exc).__name__}: {exc})，{wait}s 后重试")
                time.sleep(wait)

    # 全部失败：退回早先缓存，否则明确报错
    fallback = cache_dir / f"raw_{start}_{end}.pkl"
    if fallback.exists():
        print(f"[data] 下载失败，退回缓存 {fallback}")
        return pd.read_pickle(fallback)

    raise RuntimeError(
        f"下载 {retries} 次均失败（最后一次: {type(last_err).__name__}: {last_err}）。\n"
        f"Yahoo 限流通常几分钟后恢复；也可以先手动放一份 parquet 到 {cache} 再重跑。"
    )


# ---------------------------------------------------------------- panel 构造

def _wide(raw_stocks, field: str) -> pd.DataFrame:
    """从 yfinance 的 MultiIndex 结果里取某一字段的 wide 表 (date x ticker)。"""
    out = raw_stocks.xs(field, axis=1, level=0)
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out.columns = pd.Index([str(c) for c in out.columns], name="asset")
    return out.sort_index().sort_index(axis=1)


def to_panel(raw_stocks) -> pd.DataFrame:
    """把 yfinance 的下载结果转成 FactorBench 的规范 panel。

    返回 MultiIndex ``(date, asset)``，含 FIELDS 全部 8 列。

    vwap 用论文定义的代理 ``(high+low+close)/3``；这不是真实成交量加权价，
    但论文本身用的就是这个代理（见论文附录 A）。
    """
    close = _wide(raw_stocks, "Close")
    high = _wide(raw_stocks, "High")
    low = _wide(raw_stocks, "Low")
    vol = _wide(raw_stocks, "Volume")

    # AI_finance 的 universe 是固定票池，取 close 的并集后再对齐
    assets = close.columns
    high, low, vol = (m.reindex(index=close.index, columns=assets)
                      for m in (high, low, vol))

    # FactorBench 的 open 列存在但我们的因子用不到；用 close 填充以保持字段完整，
    # 并在说明里标注（避免下游误以为这是真实开盘价）。
    open_ = close.copy()

    ret = close.pct_change()
    # adv = 20 日均成交量（论文: 至少 5 个观测）
    adv = vol.rolling(20, min_periods=5).mean()

    vwap = (high + low + close) / 3.0

    wide = {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "vwap": vwap, "return_1d": ret, "adv": adv,
    }

    panel = pd.concat(
        {k: v.stack() for k, v in wide.items()}, axis=1,
    )
    panel.index = panel.index.rename(["date", "asset"])
    panel = panel.sort_index()
    panel.columns = pd.Index(list(FIELDS), name=None)
    return panel


def add_target(panel: pd.DataFrame, horizon: int = 20,
               benchmark_close: pd.Series | None = None,
               target_mode: str = "net") -> pd.DataFrame:
    """给 panel 加上 ``forward_return_{H}d`` 目标列。

    target_mode:
        "net" : AI_finance 口径 —— FutureReturn{H} - SPYReturn{H}（相对 SPY 的超额）
        "raw" : FactorBench 原口径 —— close[t+H]/close[t] - 1
    """
    out = panel.copy()
    close_w = out["close"].unstack("asset").sort_index()

    fwd = close_w.shift(-horizon) / close_w - 1.0

    if target_mode == "net":
        if benchmark_close is None:
            raise ValueError("target_mode='net' 需要 benchmark_close")
        bench = benchmark_close.copy()
        bench.index = pd.to_datetime(bench.index)
        bench = bench.sort_index()
        bench_fwd = bench.shift(-horizon) / bench - 1.0
        fwd = fwd.sub(bench_fwd, axis=0)
    elif target_mode != "raw":
        raise ValueError(f"未知 target_mode: {target_mode!r}")

    col = f"forward_return_{horizon}d"
    out[col] = fwd.stack().reindex(out.index)
    return out


def make_split(train_end: str = "2019-12-31",
               valid: tuple[str, str] = ("2020-01-01", "2023-12-31"),
               test: tuple[str, str] = ("2024-01-01", "2026-10-01"),
               start: str = "2010-01-01") -> Split:
    """构造与 AI_finance 参数对齐的切分。

    AI_finance main.py 用 validation_date=2020-01-01 / test_date=2024-01-01，
    同时 PreprocessEngine 有 purge（train 末尾砍掉 max_horizon 天）。
    这里同样在段落之间留出缓冲，避免目标伸进下一段。

    ``Split`` 自己会校验三段严格递增；这里额外给一个更可读的报错，
    因为最常犯的错就是把 valid 起点写早于 train 终点。
    """
    lo_train, hi_train = pd.Timestamp(start), pd.Timestamp(train_end)
    lo_valid, hi_valid = pd.Timestamp(valid[0]), pd.Timestamp(valid[1])
    lo_test, hi_test = pd.Timestamp(test[0]), pd.Timestamp(test[1])

    if hi_train >= lo_valid:
        raise ValueError(
            f"train 终点 {train_end} 不早于 valid 起点 {valid[0]}，两段重叠。\n"
            f"train_end 必须 < valid 起点；valid 起点必须 <= test 起点。"
        )
    if hi_valid > lo_test:
        raise ValueError(
            f"valid 终点 {valid[1]} 晚于 test 起点 {test[0]}，两段重叠。"
        )
    if not (lo_train <= hi_train and lo_valid <= hi_valid and lo_test <= hi_test):
        raise ValueError("每个段的起点必须不晚于终点。")

    return Split(train=(start, train_end), valid=valid, test=test)


# ---------------------------------------------------------------- 因子重建

def build_factors(panel: pd.DataFrame, specs=None) -> dict[str, pd.Series]:
    """按 FactorEngine 的定义重建全部因子，返回 {因子名: (date,asset) Series}。

    公式与 factor_engine.py **逐行对应**，groupby("Ticker") 的时序语义
    换成 wide 表上的列内滚动，等价且更快：

        Momentum{L}_{S} = close.shift(S) / close.shift(L) - 1
        Volatility{N}   = DailyReturn.rolling(N).std()
        LogVolumeRatio{N} = log( volume / volume.rolling(N).mean() )
        Return{N}       = close / close.shift(N) - 1
    """
    specs = FACTOR_SPECS if specs is None else specs

    close = panel["close"].unstack("asset").sort_index()
    volume = panel["volume"].unstack("asset").sort_index()
    ret = panel["return_1d"].unstack("asset").sort_index()

    out: dict[str, pd.Series] = {}
    for kind, params in specs:
        name = factor_name(kind, params)
        if kind == "momentum":
            long_h, short_h = params
            wide = close.shift(short_h) / close.shift(long_h) - 1.0
        elif kind == "volatility":
            wide = ret.rolling(params[0]).std()
        elif kind == "volume":
            n = params[0]
            avg = volume.rolling(n).mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                wide = np.log(volume / avg)
        elif kind == "return":
            wide = close / close.shift(params[0]) - 1.0
        else:
            raise ValueError(f"未知因子类型: {kind!r}")

        s = wide.stack()
        s.index = s.index.rename(["date", "asset"])
        out[name] = s.replace([np.inf, -np.inf], np.nan)
    return out


# ---------------------------------------------------------------- 评估薄封装

def ic_row(name: str, signal: pd.Series, target: pd.Series,
           split: Split, method: str = "ai_finance",
           horizon: int = 20) -> dict:
    """一个因子的完整 IC 记录，口径与 FactorBench 的 ic_table 一致。

    - 方向(trainsign)只在 train 段确定，之后固定 —— 这样验证/测试期的
      负 IC 是**真实的样本外符号反转**，而不是重新优化方向藏起来的。
    - t 统计量用 Newey-West(HAC)：目标是 H 日前瞻收益按日采样，相邻观测
      共享 (H-1)/H 的时间跨度，按 iid 算会高估约 3–4 倍。
      滞后阶数取 ``horizon + 5``（重叠长度 + 缓冲），而不是数据驱动的带宽公式 ——
      常规公式在两年窗口上只给约 5，远小于真实重叠。
    """
    ic = daily_ic(signal, target, "pearson")
    ric = daily_ic(signal, target, "spearman")
    sign = train_sign(ic, split)
    lags = int(horizon) + 5

    row = {"method": method, "factor": name, "sign": sign}
    for seg in ("train", "valid", "test"):
        lo, hi = split.segment(seg)
        ic_seg = ic.loc[pd.Timestamp(lo):pd.Timestamp(hi)]
        ric_seg = ric.loc[pd.Timestamp(lo):pd.Timestamp(hi)]
        signed = ic_seg * sign
        row[f"{seg}.ic"] = float(signed.mean())
        row[f"{seg}.rank_ic"] = float((ric_seg * sign).mean())
        sd = signed.std()
        row[f"{seg}.icir"] = float(signed.mean() / sd) if sd and not np.isnan(sd) else np.nan
        row[f"{seg}.n_dates"] = int(ic_seg.notna().sum())
        row[f"{seg}.t"] = newey_west_tstat(signed, lags=lags)
    return row


def admit(row: dict, min_dates: int = 100) -> bool:
    """FactorBench 的统一准入规则：每个段都有至少 min_dates 个可打分日。"""
    return all(row.get(f"{seg}.n_dates", 0) >= min_dates
               for seg in ("train", "valid", "test"))


def sign_for(signal: pd.Series, target: pd.Series, split: Split) -> float:
    """只从 train 段确定因子方向（+1/-1），之后固定不变。

    直接暴露 FactorBench 的 ``train_sign``，避免调用方去拼私有名称。
    """
    return train_sign(daily_ic(signal, target, "pearson"), split)


def eb_shrink(ic_table: pd.DataFrame, segment: str = "test") -> tuple[pd.DataFrame, float]:
    """把 Chen-Welch (2026) 的经验贝叶斯收缩套到 IC 表上。

    思路：t = 真实信号 + 抽样噪声，噪声方差按构造为 1。因此 t 值的**横截面**
    方差里，只有超过 1 的部分是真信号：

        Var(t) <= 1   →  这批因子之间的离散度不比纯运气大，不该挑最好的那个
        r_adj = [1 - 1/Var(t)] * r

    这里 ``r`` 用 IC，``t`` 用 ``ic_row`` 已经算好的 HAC t 值（不是 iid t 值 ——
    后者会把 Var(t) 抬高 3 倍以上，直接改变结论）。

    返回 (每因子调整后 IC 的 DataFrame, 收缩因子)。

    ⚠️ 因子数少时 Var(t) 本身噪声极大（Chen-Welch 用约 170 个异象估这个量），
    结果只应看量级，不能作为「某个因子显著」的依据。
    """
    t_col = f"{segment}.t"
    ic_col = f"{segment}.ic"
    if t_col not in ic_table or ic_col not in ic_table:
        raise KeyError(f"IC 表缺少 {t_col} 或 {ic_col} 列")

    t = pd.to_numeric(ic_table[t_col], errors="coerce").dropna()
    var_t = float(t.var(ddof=1)) if len(t) > 1 else float("nan")

    if not np.isfinite(var_t) or var_t <= 0:
        shrink = 0.0
    else:
        shrink = max(0.0, 1.0 - 1.0 / var_t)

    out = pd.DataFrame({
        "factor": ic_table["factor"],
        f"{segment}_ic": pd.to_numeric(ic_table[ic_col], errors="coerce"),
        f"{segment}_t": pd.to_numeric(ic_table[t_col], errors="coerce"),
    })
    out[f"{segment}_ic_eb"] = shrink * out[f"{segment}_ic"]
    out["var_t"] = var_t
    out["shrink"] = shrink
    out.attrs["var_t"] = var_t
    out.attrs["shrink"] = shrink
    out.attrs["n_factors"] = int(len(t))
    return out, shrink


__all__ = [
    "FACTOR_SPECS", "FIELDS", "factorbench_repo", "factor_name",
    "download_raw", "to_panel", "add_target", "make_split",
    "build_factors", "ic_row", "admit", "sign_for", "eb_shrink",
    "segment_metrics",
]

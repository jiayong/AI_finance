"""生成与 yfinance 输出同构的合成行情，用于在限流期间验证链路正确性。

⚠️ 这不是真实数据。它只能证明「代码每一环都通」，不能用于任何结论。

用法：
    python _make_synthetic_cache.py <cache_dir> <start> <end> [tickers...]

产出的 pkl 结构与 factorbench_adapter.download_raw 期望的完全一致，
所以把缓存放好之后，run_factorbench_eval.py 会直接复用、不再联网。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_ohlcv(symbols, dates, seed, base_vol=0.018, start_price=100.0):
    """造一组带共同市场因子的几何随机游走 OHLCV。"""
    rng = np.random.default_rng(seed)
    n_d, n_s = len(dates), len(symbols)

    # 共同市场因子 + 个股特质波动 —— 让横截面有可预测结构
    market = rng.normal(0.0003, 0.010, n_d)
    beta = rng.uniform(0.6, 1.4, n_s)
    idio = rng.normal(0.0, base_vol, (n_d, n_s))
    ret = market[:, None] * beta[None, :] + idio

    close = start_price * np.exp(np.cumsum(ret, axis=0))
    # 由 close 反推一个合理的 open/high/low，保证 high>=max(open,close) 等关系成立
    gap = rng.normal(0, 0.004, (n_d, n_s))
    open_ = close * (1 + gap)
    rng_hl = np.abs(rng.normal(0, 0.008, (n_d, n_s)))
    high = np.maximum(open_, close) * (1 + rng_hl)
    low = np.minimum(open_, close) * (1 - rng_hl)
    # 成交量：对数正态，且与 |收益| 正相关（更真实）
    volume = np.exp(rng.normal(15, 0.5, (n_d, n_s)) + 6 * np.abs(ret))

    out = {}
    for name, arr in (("Open", open_), ("High", high), ("Low", low),
                      ("Close", close), ("Volume", volume)):
        out[name] = pd.DataFrame(arr, index=dates, columns=symbols)
    return out


def main():
    cache_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "_fb_cache_synth")
    start = sys.argv[2] if len(sys.argv) > 2 else "2010-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2026-10-01"
    symbols = sys.argv[4:] or None
    if not symbols:
        from tickers import tickers as symbols

    cache_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(start, end, name="Date")

    frames = make_ohlcv(symbols, dates, seed=20261020)
    stocks = pd.concat(frames, axis=1)                 # MultiIndex (Price, Ticker)
    stocks.columns = stocks.columns.set_names(["Price", "Ticker"])

    spy_frames = make_ohlcv(["SPY"], dates, seed=7)
    spy = pd.concat(spy_frames, axis=1)
    spy.columns = spy.columns.set_names(["Price", "Ticker"])

    dest = cache_dir / f"raw_{start}_{end}.pkl"
    pd.to_pickle({"stocks": stocks, "benchmark": spy}, dest)
    print(f"合成数据已写入 {dest}")
    print(f"  票数 {len(symbols)}, 交易日 {len(dates)}, "
          f"区间 {dates[0].date()} ~ {dates[-1].date()}")
    print("  ⚠️ 这是合成数据，不是真实行情，只用于验证代码路径。")


if __name__ == "__main__":
    main()

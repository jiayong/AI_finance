"""带异质成本的回测 —— 与 FactorBench 的 ``backtest`` 逐日逻辑一致，只换成本口径。

上游把成本写成 ``turnover * cost_bps / 1e4``，这里改成对每一天、每一只股票
分别取成本：

    cost_d = Σ_i |Δw_i| × 0.5 × bps_i,d / 1e4

式中的 0.5 与上游 ``turnover = 0.5 * Σ|Δw|`` 的约定对齐，所以当所有 ``bps_i``
都等于常数 ``B`` 时，本函数与上游**逐位相同**。这条等价性由
``verify_factorbench.py`` 自动校验（见 ``检查 5``）。

除成本之外的所有细节——信号在 t 日收盘形成、t+lag 成交、只赚到 r[t+lag+1]、
权重随价格漂移、换手对漂移后的账本计算——都照抄上游，不做「改进」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_bench.portfolio.backtest import (
    TRADING_DAYS, BacktestConfig, long_only_weights, long_short_weights,
)


def backtest_with_cost_model(
    scores: pd.Series,
    returns: pd.DataFrame,
    config: BacktestConfig,
    cost_fn,
    kind: str = "long_only",
) -> pd.DataFrame:
    """与 ``factor_bench.portfolio.backtest.backtest`` 相同，但成本由 ``cost_fn`` 决定。

    ``cost_fn(date, delta_w) -> (cost_fraction, diagnostics)``：
    - ``delta_w`` 是当日权重变动 ``b - a``（卖出为负）
    - 返回的 ``cost_fraction`` 是**占组合净值的比例**
    """
    wide = scores.unstack("asset") if isinstance(scores.index, pd.MultiIndex) else scores
    wide = wide.reindex(columns=returns.columns)
    dates = returns.index
    wide = wide.reindex(dates)

    form_dates = dates[::config.rebalance_days]
    daily_borrow = config.borrow_bps_per_year / 1e4 / TRADING_DAYS

    held = pd.Series(dtype=float)
    rows = []
    pending: dict[int, pd.Series] = {}

    for i, d in enumerate(dates):
        turnover = 0.0
        trade_cost = 0.0
        vol_cost = 0.0
        size_cost = 0.0
        eq_bps = np.nan

        if i in pending:
            target = pending.pop(i)
            union = held.index.union(target.index)
            a = held.reindex(union, fill_value=0.0)
            b = target.reindex(union, fill_value=0.0)
            delta = b - a
            turnover = 0.5 * float(delta.abs().sum())
            trade_cost, diag = cost_fn(d, delta)
            vol_cost = diag.get("vol_cost", 0.0)
            size_cost = diag.get("size_cost", 0.0)
            eq_bps = diag.get("eq_bps", np.nan)
            held = b[b != 0.0]

        r = returns.loc[d]
        gross = float((held * r.reindex(held.index)).sum()) if len(held) else 0.0
        borrow = daily_borrow * float(held[held < 0].abs().sum()) if len(held) else 0.0
        rows.append({"date": d, "gross": gross, "turnover": turnover,
                     "trade_cost": trade_cost, "borrow_cost": borrow,
                     "net": gross - trade_cost - borrow,
                     "vol_cost": vol_cost, "size_cost": size_cost,
                     "eq_cost_bps": eq_bps,
                     "n_long": int((held > 0).sum()), "n_short": int((held < 0).sum()),
                     "gross_exposure": float(held.abs().sum()) if len(held) else 0.0,
                     "net_exposure": float(held.sum()) if len(held) else 0.0})

        # 权重随价格漂移（与上游一致）
        if len(held):
            grown = held * (1.0 + r.reindex(held.index).fillna(0.0))
            tot = grown.abs().sum()
            held = grown / tot * held.abs().sum() if tot > 0 else grown

        # 用当日信号形成目标仓，在 i+lag+1 进入账本
        if d in form_dates:
            effective = i + config.execution_lag + 1
            if effective < len(dates):
                s_t = wide.loc[d].dropna()
                if kind == "long_only":
                    tgt = long_only_weights(s_t, config.quantile, config.min_names)
                    if len(tgt):
                        tgt = tgt * config.gross_long
                elif kind == "long_short":
                    tgt = long_short_weights(s_t, config.quantile, config.min_names,
                                             config.ls_long, config.ls_short)
                else:
                    raise ValueError(f"unknown strategy {kind!r}")
                if len(tgt):
                    pending[effective] = tgt

    return pd.DataFrame(rows).set_index("date")

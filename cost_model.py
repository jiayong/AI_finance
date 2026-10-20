"""按流动性定价的交易成本模型。

## 为什么要异质成本

`FactorBench` 的原版回测收**统一 bps**（`BacktestConfig.cost_bps`）。这在某些条件下会
系统性低估成本：

**成本与信号相关时，平均成本不等于有效成本。**

具体到这里：AI_finance 的 15 个因子里有 4 个 `LogVolumeRatio`（成交量类），它们的冗余
相关系数高达 0.93–0.99。这类因子天然把仓位推向**低成交量**股票，而低成交量股票的真实
交易成本更高。用统一 10bp 衡量，等于给最贵的那些票按平均价收费。

## 模型

单边成本（bps）：

    one_way_bps_i = (ATR_i / close_i) * vol_factor * 1e4
                  + (order_shares_i / avg_volume_i) * size_factor * 1e4

- **波动项**：ATR 相对价格越高，冲击越大。S&P100 成分股典型 ATR/close ≈ 1.5–2.5%，
  乘 `vol_factor=0.1` → **15–25 bp 单边**。这比统一 10bp 更接近 Chen-Velikov (2023)
  估计的异象组合往返成本（约 20bp/月）。
- **规模项**：订单占日均成交量越大，冲击越大。`size_factor` 随账户规模递减。

## 关于规模项的一个反直觉事实

规模项 = `(成交股数 / 日均成交量) × size_factor`。对 $3,000 账户买 $100 的股票：

    30 股 / 5,000,000 股 × 0.03 = 1.8e-7 ≈ 0.00002%

比波动项（20bp）小**五个数量级**。也就是说在 $3k–$1M 这个区间，规模项实际是关闭的，
成本几乎完全由波动项决定。本节内置诊断会把这个分解打印出来，不让人误以为规模项在起作用。

## ATR 口径

用 Wilder 的递归平均（RMA，`alpha = 1/window`），与主流 TA 库的 `AverageTrueRange`
默认行为一致。`true_range = max(high-low, |high-prev_close|, |low-prev_close|)`。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_ONE_WAY = 0.10          # 单边成本上限 10%，与 VolatilityVolumeSlippageModel 一致


def _wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    return panel[field].unstack("asset").sort_index()


def true_range(panel: pd.DataFrame) -> pd.DataFrame:
    """真实波幅：max(high-low, |high-prev_close|, |low-prev_close|)。"""
    high = _wide(panel, "high")
    low = _wide(panel, "low")
    close = _wide(panel, "close")
    prev = close.shift(1)

    hl = (high - low).abs()
    hc = (high - prev).abs()
    lc = (low - prev).abs()
    tr = pd.concat([hl, hc, lc], keys=["hl", "hc", "lc"]).groupby(level=1).max()
    tr = tr.reindex(high.index)
    # 首个有效日的 prev_close 为 NaN，退化为 high-low
    return tr.where(tr.notna(), hl)


def atr(panel: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder ATR（递归平均，alpha = 1/window）。"""
    tr = true_range(panel)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=1).mean()


def atr_ratio(panel: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """ATR / close —— 无量纲的相对波动，用作成本代理。"""
    a = atr(panel, window)
    close = _wide(panel, "close").replace(0.0, np.nan)
    return (a / close).replace([np.inf, -np.inf], np.nan)


def avg_volume(panel: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """滚动日均成交量（股）。

    注意：原 `VolatilityVolumeSlippageModel` 的 docstring 写「rolling average」，
    代码用的却是 `ewm(span=...)`。指数加权有远超 window 的记忆，两者不是同一个量。
    这里按 docstring 的语义用滚动均值——ADV 的标准定义。
    """
    vol = _wide(panel, "volume").clip(lower=0)
    return vol.rolling(window, min_periods=max(3, window // 4)).mean()


def dynamic_size_factor(equity: float, initial_capital: float = 3000.0,
                        thresholds=(0.5, 1.0, 2.0),
                        factors=(0.03, 0.02, 0.01, 0.005)) -> float:
    """按账户规模选择 size_factor（规模越大，冲击越小）。

    边界语义：阈值处用 ``<=``，即恰好 50% 时仍取第一档。
    原实现用 ``<``，在正好边界上会掉到下一档，与其 docstring 的例子自相矛盾。
    """
    if initial_capital <= 0:
        raise ValueError("initial_capital 必须为正")
    if len(factors) != len(thresholds) + 1:
        raise ValueError("size_factors 的长度必须比 thresholds 多 1")

    ratio = equity / initial_capital
    for i, th in enumerate(thresholds):
        if ratio <= th:
            return factors[i]
    return factors[-1]


class LiquidityCostModel:
    """逐股逐日的单边成本（bps）。

    用法::

        model = LiquidityCostModel(panel, equity=100_000)
        bps = model.one_way_bps(date)          # Series: asset -> bps

    成本在 ``date`` 这一天的取值只用到 ``date`` 及之前的数据（ATR 与 ADV 都是滚动窗口，
    窗口末端含当日），与因子在 t 日形成信号的信息集一致，没有前视。
    """

    def __init__(self, panel: pd.DataFrame, *, equity: float = 100_000.0,
                 atr_window: int = 14, vol_window: int = 20,
                 vol_factor: float = 0.1,
                 initial_capital: float = 3000.0,
                 thresholds=(0.5, 1.0, 2.0),
                 factors=(0.03, 0.02, 0.01, 0.005),
                 max_one_way: float = MAX_ONE_WAY):
        self.atr_ratio = atr_ratio(panel, atr_window)
        self.avg_volume = avg_volume(panel, vol_window)
        self.close = _wide(panel, "close").replace(0.0, np.nan)
        self.equity = float(equity)
        self.vol_factor = float(vol_factor)
        self.max_one_way = float(max_one_way)
        self.size_factor = dynamic_size_factor(
            self.equity, initial_capital, thresholds, factors)
        # 诊断：最近一次 one_way_bps 的两个分量之和
        self.last_vol_bps: float = np.nan
        self.last_size_bps: float = np.nan

    def one_way_bps(self, date, assets) -> pd.Series:
        """给定日期与资产列表，返回各自的单边成本（bps）。

        规模项需要**成交金额**，而成交金额取决于目标权重。为了让成本只依赖市场数据、
        不依赖具体仓位，这里按「等权动用 equtiy」的假设计算成交股数；真实的逐笔规模项
        由 ``backtest_with_cost_model`` 在知道 |Δw| 之后另行叠加。
        """
        assets = pd.Index(assets)
        ar = self.atr_ratio.loc[date].reindex(assets) if date in self.atr_ratio.index \
            else pd.Series(np.nan, index=assets)
        vol_bps = ar.fillna(0.0) * self.vol_factor * 1e4
        return vol_bps.clip(0.0, self.max_one_way * 1e4)

    def cost_for_trades(self, date, delta_w: pd.Series) -> tuple[float, dict]:
        """一天的**总成本占组合的比例**，以及波动/规模分量的分解。

        参数
        ----
        delta_w : Series
            当日的权重变动 ``b - a``（卖出为负、买入为正），绝对值之和的一半即
            FactorBench 口径的 turnover。

        返回
        ----
        (总成本比例, 诊断字典)

        口径：``cost = Σ_i |Δw_i| × 0.5 × bps_i / 1e4``。其中的 0.5 与上游
        ``turnover = 0.5 * Σ|Δw|`` 的约定对齐 —— 这样当所有 ``bps_i`` 都等于 ``B`` 时，
        本式严格退化为上游的 ``turnover * B / 1e4``。
        """
        dw = delta_w.abs()
        dw = dw[dw > 0]
        if dw.empty:
            return 0.0, {"vol_cost": 0.0, "size_cost": 0.0, "eq_bps": 0.0}

        assets = dw.index
        ar = self.atr_ratio.loc[date].reindex(assets).fillna(0.0) \
            if date in self.atr_ratio.index else pd.Series(0.0, index=assets)
        adv = self.avg_volume.loc[date].reindex(assets) \
            if date in self.avg_volume.index else pd.Series(np.nan, index=assets)
        px = self.close.loc[date].reindex(assets) \
            if date in self.close.index else pd.Series(np.nan, index=assets)

        # 波动分量（比例）
        vol_cost = float((dw * (ar * self.vol_factor)).sum())

        # 规模分量：成交股数 = |Δw| * equity / price
        order_shares = (dw * self.equity) / px.replace(0.0, np.nan)
        size_frac = (order_shares / adv.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0) * self.size_factor
        size_cost = float((dw * size_frac).sum())

        total = vol_cost + size_cost
        total = float(min(max(total, 0.0), self.max_one_way))

        traded = float(dw.sum())
        eq_bps = 0.5 * total / traded * 1e4 if traded > 0 else 0.0
        self.last_vol_bps = 0.5 * vol_cost / traded * 1e4 if traded > 0 else 0.0
        self.last_size_bps = 0.5 * size_cost / traded * 1e4 if traded > 0 else 0.0

        return total, {
            "vol_cost": vol_cost,
            "size_cost": size_cost,
            "eq_bps": eq_bps,
        }


def constant_cost_model(bps: float):
    """常数成本模型：与上游 ``BacktestConfig.cost_bps`` 行为完全一致。

    存在的意义是当**校验用的对照** —— 异质回测在常数成本下必须与上游逐位相同，
    否则说明复制逻辑改坏了。
    """
    target = float(bps) * 0.5 / 1e4     # 对齐 turnover = 0.5*Σ|Δw| 的约定

    def _cost(date, delta_w: pd.Series) -> tuple[float, dict]:
        return float(delta_w.abs().sum()) * target, {
            "vol_cost": 0.0, "size_cost": 0.0, "eq_bps": float(bps)}

    return _cost

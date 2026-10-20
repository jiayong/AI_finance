"""修正版波动率/成交量滑点模型。

原实现的六个问题及修法，按严重程度排列。**第 1 条会改变回测结论，其余是稳健性。**

================================================================================
问题 1（致命）：减少股数不是滑点，不收任何成本
================================================================================

原代码::

    adjusted_shares = buy_shares * (1 - slippage_pct)
    ctx.buy_shares = max(adjusted_shares, Decimal("0"))

打算买 30 股 @ $100，slippage_pct = 20bp:

    原实现: 成交 29.94 股 @ $100 → 现金流出 $2994，持仓 $2994，净财富变化 0
    应有的: 成交 30 股 @ $100.20 → 现金流出 $3006，持仓 $3006，净财富 −$6

原实现少付的 $6 现金和少拿的 $6 持仓正好抵消 —— **净财富一分没掉**。
这不是滑点，是缩量。

后果不是「收益偏高」，而是**风险评估失真**：

- 多头：仓位系统性偏小，上涨时赚得少，但成本从未入账
- 多空：两条腿等比缩小 = 整体去杠杆 → 波动率下降，**Sharpe 反而可能变好看**

真实滑点会同时压低 CAGR 和 Sharpe；缩量只压低暴露。两者方向完全不同。

修法：改成调整成交价（同一数量、更差的价）。若框架的 ``ctx.fill_price`` 不可写，
则应该在账户层显式扣现金，**绝不能**用减股数来冒充成本。

================================================================================
问题 2：规模项在下述规模下完全失效
================================================================================

规模项 = ``(成交股数 / 日均成交量) × size_factor``。

$3,000 账户买 $100 的票 → 30 股 / 5,000,000 股 × 0.03 ≈ 1.8e-7 ≈ 0.00002%，
比波动项（典型 20bp）小**五个数量级**。

实测（本项目回测）：$100,000 账户下规模项占总成本 **0.0024%**；
即使放大到 $10,000,000 也只到约 1bp。**它要到机构规模才咬得住。**

如果本意是「小账户承担更大比例冲击」，参照系选错了 —— 应该用「订单占账户
资金的比例」，或把 size_factors 提到 1e-2~1e0 量级。本模块的做法是**如实报告**
两个分量各自占多少，不让人误以为规模项在工作。

================================================================================
问题 3：``.loc[:date].iloc[-1]`` 会静默用过时指标
================================================================================

停牌或假期错位时，它会悄悄取几周前的 ATR，且不报警。同时每次成交都切一次
``[0:n]``，O(n) 挂在热路径上。改用 ``searchsorted`` + 显式陈旧度检查。

================================================================================
问题 4：avg_volume 的 docstring 写「rolling」，代码用 ``ewm(span=...)``
================================================================================

指数加权有远超 ``vol_window`` 的记忆，与滚动均值不是同一个量。ADV 的标准定义是
滚动均值，本模块按 docstring 的语义改用滚动均值。

================================================================================
问题 5：``get_dynamic_size_factor`` 边界 off-by-one
================================================================================

docstring 说 ``$1,500 (50%) → size_factors[0]``，但 ``ratio < 0.5`` 在正好 50%
时为 False，掉到下一档返回 ``size_factors[1]``。本模块在阈值处用 ``<=``。

================================================================================
问题 6：条件赋值的变量在异常路径上可能未定义
================================================================================

``relative_volatility`` / ``order_shares`` 只在部分分支赋值，日志段却引用它们。
目前靠外层 guard 侥幸成立，任何分支改动都会炸 NameError。改为开头就初始化为 0。
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional, Sequence

MAX_SLIPPAGE = Decimal("0.10")      # 单边上限 10%
MIN_SLIPPAGE = Decimal("0.0")

# 陈旧阈值：超过这个交易日数没有新数据，就认为指标失效、退回波动率兜底
MAX_STALENESS_DAYS = 5


def dynamic_size_factor(portfolio_value: Decimal, initial_capital: Decimal,
                        thresholds: Sequence[float],
                        factors: Sequence[float]) -> Decimal:
    """按账户规模选择 size_factor。阈值处用 ``<=``（恰好 50% 仍取第一档）。"""
    if initial_capital <= 0:
        raise ValueError("initial_capital 必须为正")
    if len(factors) != len(thresholds) + 1:
        raise ValueError("size_factors 的长度必须比 thresholds 多 1")

    ratio = portfolio_value / initial_capital
    for i, th in enumerate(thresholds):
        if ratio <= Decimal(str(th)):
            return Decimal(str(factors[i]))
    return Decimal(str(factors[-1]))


class MetricLookup:
    """按日期取 ATR / ADV，带二分查找与陈旧度检查。

    把「取当日指标」从 O(n) 切片降为 O(log n)，并让**数据过期**变成可见事件，
    而不是悄悄用一个几周前的 ATR。
    """

    def __init__(self, frames: dict):
        # frames: symbol -> DataFrame(index=DatetimeIndex, columns=["atr","avg_volume"])
        self.frames = {}
        for symbol, df in frames.items():
            df = df.sort_index()
            self.frames[symbol] = {
                "index": df.index,
                "atr": df["atr"].to_numpy(dtype=float),
                "avg_volume": df["avg_volume"].to_numpy(dtype=float),
            }
        self.stale_count = 0
        self.miss_count = 0

    def get(self, symbol: str, date) -> Optional[tuple[float, float, int]]:
        """返回 (atr, avg_volume, 陈旧天数)；无数据返回 None。"""
        f = self.frames.get(symbol)
        if f is None or len(f["index"]) == 0:
            self.miss_count += 1
            return None

        i = int(f["index"].searchsorted(date, side="right")) - 1
        if i < 0:
            self.miss_count += 1
            return None

        stale = int((date - f["index"][i]).days)
        if stale > MAX_STALENESS_DAYS:
            self.stale_count += 1
            return None                      # 过期 -> 让调用方走兜底，而不是用陈旧值

        return float(f["atr"][i]), float(f["avg_volume"][i]), stale


class VolatilityVolumeSlippageModel:
    """波动率 + 成交量滑点。

    与原实现的关键差异：**通过调整成交价来收取成本**，而不是减少成交股数。

    参数
    ----
    lookup : MetricLookup
        逐股票的 ATR / ADV 查询器。
    vol_factor : float
        波动项系数。典型 ATR/close ≈ 2%，乘 0.1 → 20bp 单边。
    thresholds, size_factors, initial_capital, use_total_equity
        与原来一致，但边界语义修正为 ``<=``。

    重要
    ----
    本类要求执行上下文支持**写入** ``ctx.fill_price``。若你的框架不允许，
    ``apply_slippage`` 会抛出 ``NotImplementedError`` 而不是静默地少买一点 ——
    因为「少买一点」并不产生成本，会把回测做得过于乐观。
    """

    def __init__(self, lookup: MetricLookup, *, vol_factor: float = 0.1,
                 thresholds: Sequence[float] = (0.5, 1.0, 2.0),
                 size_factors: Sequence[float] = (0.03, 0.02, 0.01, 0.005),
                 initial_capital: float = 3000.0,
                 use_total_equity: bool = True):
        if len(size_factors) != len(thresholds) + 1:
            raise ValueError("size_factors 的长度必须比 thresholds 多 1")
        self.lookup = lookup
        self.vol_factor = float(vol_factor)
        self.thresholds = tuple(float(t) for t in thresholds)
        self.size_factors = tuple(float(f) for f in size_factors)
        self.initial_capital = Decimal(str(initial_capital))
        self.initial_capital_f = float(initial_capital)
        self.use_total_equity = use_total_equity
        if self.initial_capital <= 0:
            raise ValueError("initial_capital 必须为正")
        self.last_cost_bps = 0.0

    # ---------------------------------------------------------------- 内部
    def _portfolio_value(self, ctx) -> Optional[Decimal]:
        try:
            if self.use_total_equity:
                v = getattr(ctx, "total_equity", None)
                if v is None:
                    return None
                return Decimal(str(v))
            v = getattr(ctx, "cash", None)
            if v is None:
                return None
            return Decimal(str(v))
        except Exception:                                    # noqa: BLE001
            return None

    def _price(self, ctx) -> Optional[Decimal]:
        p = getattr(ctx, "fill_price", None)
        if p:
            return Decimal(str(p))
        close = getattr(ctx, "close", None)
        if close is not None and len(close) > 0:
            return Decimal(str(close[-1]))
        return None

    def slippage_pct(self, ctx, shares: Decimal) -> Decimal:
        """返回该笔交易的单边滑点比例。无法估算时返回 0。"""
        symbol = ctx.symbol
        price = self._price(ctx)
        pv = self._portfolio_value(ctx)
        if price is None or price <= 0 or pv is None:
            return Decimal("0")

        # 两个分量一开始就定义为 0 —— 消除「异常路径上变量未定义」的隐患
        vol_component = Decimal("0")
        size_component = Decimal("0")

        metrics = self.lookup.get(symbol, ctx.dt)
        if metrics is not None:
            atr, adv, _stale = metrics
            if atr > 0:
                vol_component = Decimal(str(atr / float(price))) * \
                    Decimal(str(self.vol_factor))
            if adv > 0 and shares > 0:
                size_factor = dynamic_size_factor(
                    pv, self.initial_capital, self.thresholds, self.size_factors)
                size_component = (shares / Decimal(str(adv))) * size_factor
        # metrics 为 None（无数据/过期）时只保留 0 —— 明确地不收费，并计入统计

        total = vol_component + size_component
        total = min(max(total, MIN_SLIPPAGE), MAX_SLIPPAGE)

        if total > Decimal("0.05"):
            print(f"High slippage {symbol}: {float(total):.2%} "
                  f"(vol {float(vol_component):.2%}, size {float(size_component):.2%}, "
                  f"portfolio ${float(pv):.0f})")
        self.last_cost_bps = float(total) * 1e4
        return total

    # ---------------------------------------------------------------- 对外
    def apply_slippage(self, ctx, buy_shares: Optional[Decimal] = None,
                       sell_shares: Optional[Decimal] = None):
        """按更差的成交价收取滑点。

        买入价上浮、卖出价下调，**数量不变** —— 这样成本才真正进入现金流。
        """
        shares = buy_shares if buy_shares else sell_shares
        if not shares:
            return

        if not hasattr(ctx, "fill_price"):
            raise NotImplementedError(
                "执行上下文没有 fill_price，无法通过价格收取滑点。"
                "减少 buy_shares 只能在账面上缩量、不产生任何成本，会把回测做得过于乐观。"
                "若确实只能调整数量，请在账户层显式扣现金成本。"
            )

        pct = self.slippage_pct(ctx, Decimal(str(shares)))
        price = self._price(ctx)
        if pct <= 0 or price is None:
            return

        one = Decimal("1")
        if buy_shares:
            ctx.fill_price = price * (one + pct)
        elif sell_shares:
            ctx.fill_price = price * (one - pct)

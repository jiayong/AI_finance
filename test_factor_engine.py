"""FactorEngine 配置分发与因子集校验的离线回归测试（不联网）。

运行：
    python test_factor_engine.py

用合成数据构造一个最小可用的面板，覆盖：
    - 正常配置 15 因子
    - 分发器的四类错误拦截
    - verify() 对缺失/常量/inf/全NaN/高NaN 的识别
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factor_engine import FactorEngine


class FakeDataEngine:
    """伪造 DataEngine，提供 FactorEngine 需要的字段。"""

    def __init__(self, n_days=400, n_tickers=10, forecast_horizon=20, seed=7):
        self.forecast_horizon = forecast_horizon
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2015-01-01", periods=n_days)
        tickers = [f"T{i:02d}" for i in range(n_tickers)]

        rows = []
        for ticker in tickers:
            price = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n_days)))
            volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
            for d, p, v in zip(dates, price, volume):
                rows.append((d, ticker, p, v))

        df = pd.DataFrame(rows, columns=["Date", "Ticker", "Close", "Volume"])
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        df["DailyReturn"] = df.groupby("Ticker")["Close"].pct_change()

        h = forecast_horizon
        df[f"FutureReturn{h}"] = df.groupby("Ticker")["Close"].shift(-h) / df["Close"] - 1
        df[f"SPYReturn{h}"] = 0.001
        df[f"NetReturn{h}"] = df[f"FutureReturn{h}"] - df[f"SPYReturn{h}"]
        self.data = df


SPECS = [
    ("momentum",   [100, 5]),
    ("momentum",   [60, 5]),
    ("momentum",   [20, 5]),
    ("momentum",   [10, 5]),
    ("volatility", [100]),
    ("volatility", [60]),
    ("volatility", [20]),
    ("volatility", [10]),
    ("volume",     [100]),
    ("volume",     [60]),
    ("volume",     [20]),
    ("volume",     [10]),
    ("return",     [20]),
    ("return",     [10]),
    ("return",     [5]),
]


# ----------------------------------------------------------------------
# 测试框架：断言失败即 FAIL，抛异常即 FAIL，除非用例本身声明要求抛异常
# ----------------------------------------------------------------------

RESULTS = []


def case(name):
    """装饰器：注册一个用例。"""

    def wrapper(fn):
        RESULTS.append((name, fn))
        return fn

    return wrapper


def raises(exc_type, fn):
    """断言 fn() 抛出指定异常。"""
    try:
        fn()
    except exc_type:
        return
    except Exception as other:  # 抛了异常但类型不对
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(other).__name__}: {other}"
        ) from other
    raise AssertionError(f"expected {exc_type.__name__}, but no error was raised")


def new_engine():
    return FactorEngine(FakeDataEngine())


# ----------------------------------------------------------------------
# 用例
# ----------------------------------------------------------------------

@case("正常配置 15 因子全部登记")
def _():
    fe = new_engine()
    fe.add_factors(SPECS)
    assert len(fe.features) == 15, fe.features
    assert len(set(fe.features)) == 15, "因子名有重复"


@case("因子名与类型表定义一致")
def _():
    fe = new_engine()
    fe.add_factor("momentum", [100, 5])
    fe.add_factor("volatility", [60])
    fe.add_factor("volume", [20])
    fe.add_factor("return", [10])
    assert fe.features == [
        "Momentum100_5", "Volatility60", "LogVolumeRatio20", "Return10"
    ], fe.features


@case("未知类型 kind 报错")
def _():
    raises(ValueError, lambda: new_engine().add_factor("banana", [5]))


@case("参数个数不匹配报错")
def _():
    raises(ValueError, lambda: new_engine().add_factor("momentum", [60]))
    raises(ValueError, lambda: new_engine().add_factor("volatility", [20, 30]))


@case("参数非正整数报错")
def _():
    raises(ValueError, lambda: new_engine().add_factor("volatility", [0]))
    raises(ValueError, lambda: new_engine().add_factor("return", [-5]))
    raises(ValueError, lambda: new_engine().add_factor("volume", [2.5]))


@case("重复登记同一因子报错")
def _():
    def dup():
        fe = new_engine()
        fe.add_factor("momentum", [60, 5])
        fe.add_factor("momentum", [60, 5])

    raises(ValueError, dup)


@case("空因子集 verify 报错")
def _():
    raises(RuntimeError, lambda: new_engine().verify(verbose=False))


@case("因子列缺失被识别并报错")
def _():
    fe = new_engine()
    fe.add_factor("volatility", [20])
    md = fe.build_model_data().drop(columns=["Volatility20"])
    raises(ValueError, lambda: fe.verify(md, verbose=False))


@case("常量因子被识别")
def _():
    fe = new_engine()
    fe.add_factor("volatility", [20])
    md = fe.build_model_data()
    md["Volatility20"] = 1.0
    report = fe.verify(md, verbose=False)
    assert report["constant"] == ["Volatility20"], report["constant"]


@case("inf 因子被识别")
def _():
    fe = new_engine()
    fe.add_factor("volatility", [20])
    md = fe.build_model_data()
    md.loc[md.index[0], "Volatility20"] = np.inf
    report = fe.verify(md, verbose=False)
    assert report["infinite"] == ["Volatility20"], report["infinite"]


@case("全 NaN 因子被识别")
def _():
    fe = new_engine()
    fe.add_factor("volatility", [20])
    md = fe.build_model_data()
    md["Volatility20"] = np.nan
    report = fe.verify(md, verbose=False)
    assert report["all_nan"] == ["Volatility20"], report["all_nan"]


@case("高 NaN 率被识别为警告")
def _():
    fe = new_engine()
    fe.add_factor("volatility", [20])
    md = fe.build_model_data()
    md.loc[md.index[: len(md) // 2], "Volatility20"] = np.nan
    report = fe.verify(md, verbose=False)
    assert report["high_nan"] == ["Volatility20"], report["high_nan"]


@case("strict=True 让警告升级为异常")
def _():
    fe = new_engine()
    fe.add_factor("volatility", [20])
    md = fe.build_model_data()
    md.loc[md.index[: len(md) // 2], "Volatility20"] = np.nan
    raises(ValueError, lambda: fe.verify(md, strict=True, verbose=False))


@case("正常因子集的 verify 报告干净")
def _():
    fe = new_engine()
    fe.add_factors(SPECS)
    report = fe.verify(verbose=False)
    assert report["n_features"] == 15
    for key in ("missing", "constant", "infinite", "all_nan", "empty", "high_nan"):
        assert not report[key], (key, report[key])
    assert set(report["nan_rates"]) == set(fe.features)


@case("add_factors 返回 self 可链式调用")
def _():
    fe = new_engine()
    assert fe.add_factors([("return", [5])]) is fe
    assert fe.features == ["Return5"]


# ----------------------------------------------------------------------

def main():
    passed, failed = 0, []
    for name, fn in RESULTS:
        try:
            fn()
            passed += 1
            print(f"OK    {name}")
        except Exception as exc:
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {name}")

    print("=" * 72)
    if failed:
        for name, detail in failed:
            print(f"FAILED: {name}\n        {detail}")
    print(f"passed {passed}/{len(RESULTS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

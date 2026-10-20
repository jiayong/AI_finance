"""校验 main.py 里的 FACTOR_SPECS 与 FactorEngine 分发器一致（不联网）。

用 AST 从 main.py 取出配置表，避免导入 main.py 触发它的
yfinance 下载循环。运行：
    python test_factor_specs.py
"""
import ast
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from factor_engine import FactorEngine


def load_specs_from_main():
    """从 main.py 源码里静态取出 FACTOR_SPECS，不执行 main.py。"""
    path = os.path.join(HERE, "main.py")
    tree = ast.parse(io.open(path, encoding="utf-8").read(), filename=path)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "FACTOR_SPECS":
                return ast.literal_eval(node.value)

    raise AssertionError("FACTOR_SPECS not found at module level in main.py")


class FakeDataEngine:
    def __init__(self, n_days=400, n_tickers=10, forecast_horizon=20, seed=11):
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


def main():
    specs = load_specs_from_main()
    print(f"main.py 中 FACTOR_SPECS 条目数: {len(specs)}")

    engine = FactorEngine(FakeDataEngine())
    engine.add_factors(specs)

    assert len(engine.features) == len(specs), (
        f"registered {len(engine.features)} != configured {len(specs)}"
    )
    assert len(set(engine.features)) == len(engine.features), "因子名有重复"

    print(f"成功登记因子数: {len(engine.features)}")

    report = engine.verify(verbose=False)
    assert not report["missing"], report["missing"]
    for key in ("constant", "infinite", "all_nan", "empty"):
        assert not report[key], (key, report[key])

    print("\n生效的因子（顺序即 features 顺序）:")
    for i, name in enumerate(engine.features, 1):
        print(f"  {i:2d}. {name}")

    print("\n所有因子列均存在于 model_data，无缺失/常量/inf")
    print("FACTOR_SPECS 与分发器一致 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())

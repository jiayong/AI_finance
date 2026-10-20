import numpy as np

class FactorEngine:

    def __init__(self, data_engine):

        self.data_engine = data_engine
        self.target = f"NetReturn{data_engine.forecast_horizon}"
        self.features = []
        self.data = data_engine.data.copy()

    def add_momentum(self,long_horizon,short_horizon):
        
        grouped_close = self.data.groupby("Ticker")["Close"]
        self.data[f"Momentum{long_horizon}_{short_horizon}"] = grouped_close.shift(short_horizon)/grouped_close.shift(long_horizon) - 1
        
        self.features.append(f"Momentum{long_horizon}_{short_horizon}")


    def add_volatility(self,horizon):

        self.data[f"Volatility{horizon}"] = self.data.groupby("Ticker")["DailyReturn"].transform(lambda x: x.rolling(horizon).std())

        self.features.append(f"Volatility{horizon}")

    def add_volume(self,horizon):

        self.data[f"AvgVolume{horizon}"] = self.data.groupby("Ticker")["Volume"].transform(lambda x:x.rolling(horizon).mean())
        self.data[f"VolumeRatio{horizon}"] = self.data["Volume"]/self.data[f"AvgVolume{horizon}"]
        self.data[f"LogVolumeRatio{horizon}"] = self.data[f"VolumeRatio{horizon}"].transform(np.log)
        self.data = self.data.replace([np.inf,-np.inf],np.nan)
        self.features.append(f"LogVolumeRatio{horizon}")

    def add_shortterm_return(self,horizon):
        
        grouped_close = self.data.groupby("Ticker")["Close"]
        self.data[f"Return{horizon}"] = self.data["Close"]/grouped_close.shift(horizon) -1
        self.features.append(f"Return{horizon}")

    # ------------------------------------------------------------------
    # 配置分发：把「因子集」变成纯配置数据，不用改引擎代码
    # ------------------------------------------------------------------

    # 因子类型 -> (构造函数名, 该构造函数的参数名)
    # 新增因子族时，只需在这里登记一行，加上一个新的 add_* 方法即可
    FACTOR_TYPES = {
        "momentum":   ("add_momentum",         ("long_horizon", "short_horizon")),
        "volatility": ("add_volatility",        ("horizon",)),
        "volume":     ("add_volume",            ("horizon",)),
        "return":     ("add_shortterm_return",  ("horizon",)),
    }

    def _feature_names_for(self, kind, params):
        """在不改动 self.data 的前提下，预测某个因子会生成哪些列名。

        只用于查重和报告，不产生副作用。
        """
        if kind == "momentum":
            long_horizon, short_horizon = params
            return [f"Momentum{long_horizon}_{short_horizon}"]

        if kind == "volatility":
            return [f"Volatility{params[0]}"]

        if kind == "volume":
            return [f"LogVolumeRatio{params[0]}"]

        if kind == "return":
            return [f"Return{params[0]}"]

        raise ValueError(f"Unknown factor kind: {kind}")

    def add_factor(self, kind, params):
        """按配置添加一个因子。

        kind   : FACTOR_TYPES 里的因子类型字符串
        params : 参数序列，顺序与 FACTOR_TYPES 里登记的 arg_names 一致

        例：
            factor_engine.add_factor("momentum", [60, 5])   -> Momentum60_5
            factor_engine.add_factor("volatility", [100])   -> Volatility100
        """
        if kind not in self.FACTOR_TYPES:
            raise ValueError(
                f"Unknown factor kind {kind!r}. "
                f"Available kinds: {sorted(self.FACTOR_TYPES)}"
            )

        method_name, arg_names = self.FACTOR_TYPES[kind]

        if len(params) != len(arg_names):
            raise ValueError(
                f"Factor {kind!r} expects {len(arg_names)} param(s) "
                f"{arg_names}, got {len(params)}: {params!r}"
            )

        # 查重：同一个因子被配了两次会让 LASSO 输入完全共线
        new_names = self._feature_names_for(kind, params)
        duplicated = [n for n in new_names if n in self.features]
        if duplicated:
            raise ValueError(
                f"Duplicate factor {duplicated} — it is already registered. "
                f"Check the factor configuration for repeated entries."
            )

        # 参数合法性：窗口必须为正整数
        for name, value in zip(arg_names, params):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(
                    f"Parameter {name}={value!r} for factor {kind!r} "
                    f"must be a positive integer."
                )

        getattr(self, method_name)(*params)

        # 登记表必须与配置一致，否则下游会静默漏掉因子
        for name in new_names:
            if name not in self.features:
                raise RuntimeError(
                    f"Factor {kind!r} did not register {name} into "
                    f"self.features — check that add_* appends its name."
                )

    def add_factors(self, specs):
        """批量添加因子。specs 为 (kind, [params]) 序列。

        specs = [
            ("momentum",   [100, 5]),
            ("momentum",   [60, 5]),
            ("volatility", [100]),
        ]
        返回 self，便于链式调用。
        """
        for kind, params in specs:
            self.add_factor(kind, params)
        return self

    # ------------------------------------------------------------------
    # 因子集校验：跑模型前确认「我以为生效的因子」真的生效了
    # ------------------------------------------------------------------

    def verify(self, model_data=None, strict=False, verbose=True):
        """校验当前登记的因子集。

        返回一个 dict 报告，包含：
            features        实际登记的因子名
            n_features      因子个数
            missing         在 model_data 中找不到的因子（致命）
            constant        在所有行上取值恒定、无截面区分度的因子（致命）
            infinite        仍含 inf 的因子（致命）
            all_nan         全部为 NaN 的因子（致命）
            empty           全部为空（全 0 或全 NaN）的因子（致命）
            high_nan        NaN 比例超过阈值的因子（警告）
            nan_rates       每个因子的 NaN 比例

        strict=True 时，任何警告都会升级为异常。
        """
        if model_data is None:
            model_data = self.build_model_data()

        report = {
            "features": list(self.features),
            "n_features": len(self.features),
            "missing": [],
            "constant": [],
            "infinite": [],
            "all_nan": [],
            "empty": [],
            "high_nan": [],
            "nan_rates": {},
        }

        # 因子集为空：下游会得到一个只有常数的回归，必须明确报错
        if not self.features:
            raise RuntimeError(
                "Factor set is empty — no factor was registered. "
                "Check the factor configuration."
            )

        report["missing"] = [f for f in self.features if f not in model_data.columns]

        available = [f for f in self.features if f not in report["missing"]]

        for feature in available:
            column = model_data[feature]

            n_nan = int(column.isna().sum())
            report["nan_rates"][feature] = n_nan / len(column) if len(column) else 1.0

            # 无穷值：会污染当天整个截面的均值/标准差
            if np.isinf(column.to_numpy(dtype="float64", na_value=np.nan)).any():
                report["infinite"].append(feature)
                continue

            if column.isna().all():
                report["all_nan"].append(feature)
                continue

            # 该因子是否在整个样本上恒定（没有截面区分度）
            try:
                if column.nunique(dropna=True) <= 1:
                    report["constant"].append(feature)
            except TypeError:
                pass

            if (column.fillna(0) == 0).all():
                report["empty"].append(feature)

            # 暖机期带来的高 NaN 率会大幅削减可用样本
            if report["nan_rates"][feature] > 0.2:
                report["high_nan"].append(feature)

        fatal = (
            report["missing"]
            + report["constant"]
            + report["infinite"]
            + report["all_nan"]
            + report["empty"]
        )

        if verbose:
            print(f"\n[FactorEngine] Registered {report['n_features']} factor(s):")
            for feature in report["features"]:
                rate = report["nan_rates"].get(feature, float("nan"))
                flag = ""
                if feature in report["missing"]:
                    flag = "  <-- MISSING COLUMN"
                elif feature in fatal:
                    flag = "  <-- INVALID"
                elif feature in report["high_nan"]:
                    flag = "  <-- HIGH NaN"
                print(f"    {feature:<24} NaN={rate:6.2%}{flag}")
            print(f"[FactorEngine] model_data rows: {len(model_data)}")

        if report["missing"]:
            raise ValueError(
                f"These factors are missing from model_data: {report['missing']}. "
                f"Available columns include: "
                f"{sorted(set(model_data.columns) - {'Date', 'Ticker', self.target})[:20]}"
            )

        for key, label in (
            ("constant", "constant across all rows (no cross-sectional variation)"),
            ("infinite", "contain infinite values"),
            ("all_nan", "are entirely NaN"),
            ("empty", "are entirely zero/NaN"),
        ):
            if report[key]:
                message = f"These factors {label}: {report[key]}"
                if strict:
                    raise ValueError(message)
                print(f"[FactorEngine] WARNING: {message}")

        if report["high_nan"]:
            message = (
                f"These factors have >20% NaN (long warm-up): {report['high_nan']}. "
                f"They will shrink the usable sample via dropna()."
            )
            if strict:
                raise ValueError(message)
            print(f"[FactorEngine] WARNING: {message}")

        return report

    def build_model_data(self):
        
        model_data = self.data[["Date","Ticker"]+self.features + [self.target]].copy()
        model_data = model_data.dropna()

        return model_data


import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso

from hac_inference import fit_ols


class LearningEngine:
    """拟合与样本外评估。

    关于标准误（``cov_type``）——这是本项目最容易出错、也最容易低估的地方：

    目标变量是 ``NetReturn{H}``，即 H 日前瞻收益**按日采样**。相邻两天共享
    H-1/H 的时间跨度，残差在时间上高度自相关；同时同一天不同股票的残差也
    相关（横截面相关）。`cov_type="cluster", groups=Date` 只处理了后者。

    实测（50 只票，2015-2025，H=20）：只按日期聚类会把标准误低估 **1.8–3.6 倍**，
    足以把一个 t=3.15 的噪声因子变成「显著」。

    因此默认改为 ``driscoll_kraay``：先把逐期得分按时间聚合，再对时间序列做
    Newey-West —— 同时吸收两种相关。

    可选：
      - ``driscoll_kraay``（默认）：横截面 + 序列相关
      - ``two_way``：两向聚类（个股 × 时间），不含 Newey-West
      - ``cluster_date``：旧行为，**会低估标准误**，仅保留做对照
      - ``iid``：不调整
    """

    COV_TYPES = ("driscoll_kraay", "two_way", "cluster_date", "iid")

    def __init__(self, model_type, train, validation, test, standardized_features,
                 factor_engine, jump, alpha, offset=0,
                 cov_type="driscoll_kraay", hac_lags=None):

        self.model_type = model_type
        self.train_data = train
        self.validation_data = validation
        self.test_data = test
        self.standardized_features = standardized_features
        self.model = None
        self.factor_engine = factor_engine
        self.jump = jump
        self.alpha = alpha
        self.offset = offset

        if cov_type not in self.COV_TYPES:
            raise ValueError(
                f"cov_type 必须是 {self.COV_TYPES} 之一，收到 {cov_type!r}"
            )
        self.cov_type = cov_type
        # HAC 滞后阶数应按已知的重叠长度取，而不是数据驱动的带宽公式：
        # 常规 Newey-West 公式在两年窗口上只给约 5，远小于真实的 20 日重叠。
        self.hac_lags = hac_lags
        # 最近一次 LR 拟合的推断结果，供报告使用
        self.inference = None

    def _lags(self) -> int:
        """HAC 滞后阶数：默认取预测周期（= 目标重叠长度）。

        目标列名形如 ``NetReturn20``，末尾数字就是 horizon，直接解析最可靠；
        解析不出来再回退到属性查找，最后兜底 20。
        """
        if self.hac_lags is not None:
            return int(self.hac_lags)

        target = getattr(self.factor_engine, "target", "")
        if isinstance(target, str) and target.startswith("NetReturn"):
            suffix = target[len("NetReturn"):]
            if suffix.isdigit():
                return int(suffix)

        horizon = getattr(self.factor_engine, "forecast_horizon", None)
        if horizon is None:
            horizon = getattr(getattr(self.factor_engine, "data_engine", None),
                              "forecast_horizon", None)
        return int(horizon) if horizon else 20

    def train(self):

        x_train = self.train_data[self.standardized_features]
        y_train = self.train_data[self.factor_engine.target]

        if self.model_type == "LR":

            # 走 hac_inference：显式给出 t 值，而不是依赖 statsmodels 的默认协方差
            self.inference = fit_ols(
                self.train_data,
                y_col=self.factor_engine.target,
                x_cols=list(self.standardized_features),
                cov=self.cov_type,
                lags=self._lags(),
            )
            # 保留 .model 的接口（validate/test 依赖 .predict 与 .params）
            x_train_c = sm.add_constant(x_train)
            self.model = sm.OLS(y_train, x_train_c).fit()

        elif self.model_type == "LASSO":
            self.model = Lasso(
                    alpha = self.alpha,
                    fit_intercept = True,
                    max_iter = 10000
                    )
            self.model.fit(
                    x_train,
                    y_train
                    )
        return self.model

    def validate(self):
        x_validation = self.validation_data[self.standardized_features]
        if self.model_type == "LR":

            x_validation = sm.add_constant(
                x_validation,
                has_constant="add"
            )
            self.validation_data["PredictedReturn"] = self.model.predict(
                x_validation
            )

        elif self.model_type == "LASSO":

            self.validation_data["PredictedReturn"] = self.model.predict(x_validation)

            # Take every jump-th unique date
        validation_dates = (
            self.validation_data["Date"]
            .drop_duplicates()
            .sort_values()
        )
        sampled_dates = validation_dates.iloc[
            self.offset::self.jump
        ]
            # Keep all stocks for those dates
        independent_validation = self.validation_data[
            self.validation_data["Date"].isin(sampled_dates)
        ]
        actual = independent_validation[self.factor_engine.target]
        predicted = independent_validation["PredictedReturn"]
        oos_r2 = 1 - (
            ((actual - predicted) ** 2).sum()
            / (actual ** 2).sum()
        )
        return independent_validation, oos_r2

    def summary(self) -> str:
        """当前拟合的系数与 t 值（按选定的协方差估计器）。"""
        if self.inference is None:
            return "尚未拟合 LR 模型，或 model_type 不是 LR。"
        inf = self.inference
        lines = [
            f"协方差估计: {inf['cov_type']}"
            + (f"（HAC lag={inf['lags']}）" if inf["lags"] else ""),
            f"样本: {inf['nobs']:,}  R²: {inf['rsquared']:.4f}",
            f"{'变量':<24}{'系数':>12}{'标准误':>12}{'t':>9}",
        ]
        for name in inf["params"].index:
            lines.append(
                f"{name:<24}{inf['params'][name]:>12.6f}"
                f"{inf['bse'][name]:>12.6f}{inf['tvalues'][name]:>9.2f}"
            )
        return "\n".join(lines)

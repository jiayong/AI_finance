# AI_finance 因子工作模式与原理 · 完整教程

> 面向本项目 `AI_finance`（Yazdan-Babazadeh）的因子研究管线，逐行拆解「因子怎么算、怎么标准化、怎么被模型选中、怎么变成组合」，
> 并给出配置、运行、扩展新因子的操作方法。
>
> 阅读前提：本教程对照以下文件，建议边读边开着源码：
> `data_engine.py`、`factor_engine.py`、`preprocess_engine.py`、`learning_engine.py`、`backtest_engine.py`、`main.py`、`tickers.py`
>
> ⚠️ 注意：仓库有多个分支，**因子研究最完整的版本在 `medium-blog` 分支**（`main.py` 545 行，含 15 因子库、factor ablation、LASSO alpha 调参、forecast horizon 扫描、offset 稳健性）。
> `portfolio-optimization` 分支的 `main.py` 只有 47 行，是回测/组合优化的精简演示版。本教程以 `medium-blog` 为主干，并标注两者的差异。

---

## 目录

1. [一句话理解这个系统](#1-一句话理解这个系统)
2. [整体架构与数据流](#2-整体架构与数据流)
3. [第一层：DataEngine —— 目标变量的构造](#3-第一层dataengine-目标变量的构造)
4. [第二层：FactorEngine —— 因子库](#4-第二层factorengine-因子库)
5. [第三层：PreprocessEngine —— 截面标准化与切分](#5-第三层preprocessengine-截面标准化与切分)
6. [第四层：LearningEngine —— LR 与 LASSO](#6-第四层learningengine-lr-与-lasso)
7. [第五层：BacktestEngine —— 两种组合构造](#7-第五层backtestengine-两种组合构造)
8. [实验模式一：Forecast Horizon 扫描](#8-实验模式一forecast-horizon-扫描)
9. [实验模式二：Factor Ablation](#9-实验模式二factor-ablation)
10. [实验模式三：Alpha 调参与 Offset 稳健性](#10-实验模式三alpha-调参与-offset-稳健性)
11. [实跑结果解读](#11-实跑结果解读)
12. [操作手册：如何配置与运行](#12-操作手册如何配置与运行)
13. [如何扩展：新增因子的标准流程](#13-如何扩展新增因子的标准流程)
    - [13.5 配置驱动切换因子](#135-配置驱动切换因子factorengine-扩展)
14. [工程陷阱与已知问题](#14-工程陷阱与已知问题)
15. [后续研究路线](#15-后续研究路线)

---

## 1. 一句话理解这个系统

这是一个**截面（cross-sectional）股票收益预测系统**。它不问「市场明天涨还是跌」，而是每天问：

> 在这 49 只股票里，**哪些会跑赢 SPY，哪些会跑输 SPY**？

所以它的预测目标不是绝对收益，而是**相对基准的超额收益**（`NetReturn = 个股未来收益 − SPY 未来收益`）。
因子只负责在这个横截面上给股票排序，模型负责把排序变成一个连续的打分，回测负责把打分变成多空组合。

这个定位决定了后面所有设计：为什么要在**每一天内做截面标准化**、为什么用**带符号的多空组合**、为什么用 **sum(w)=0 的美元中性约束**。

---

## 2. 整体架构与数据流

```
                    ┌─────────────────────────────────────────┐
                    │  tickers.py  (49 只美股，7 个行业)       │
                    │  benchmark = "SPY"                      │
                    └────────────────┬────────────────────────┘
                                     ↓
┌────────────────────────────────────────────────────────────────────┐
│ ① DataEngine      yfinance 下载 → 长表 → 收益率 → 目标变量          │
│    输出 data: [Date, Ticker, Close, Volume, DailyReturn,            │
│               FutureReturnH, SPYClose, SPYReturnH, NetReturnH]      │
└────────────────────────────────┬───────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ ② FactorEngine    在长表上追加因子列，并登记 self.features           │
│    输出 model_data: [Date, Ticker, 15 个因子, NetReturnH]           │
└────────────────────────────────┬───────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ ③ PreprocessEngine                                                 │
│    normalize()   : 按 Date 分组做 z-score → 因子名加 _z 后缀        │
│    split_data()  : 按时间切 train / validation / test（带 purge）   │
└────────────────────────────────┬───────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ ④ LearningEngine   LR(OLS, cluster by Date) 或 LASSO               │
│    输出 validation_data["PredictedReturn"] + OOS R²                │
└────────────────────────────────┬───────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────┐
│ ⑤ BacktestEngine                                                   │
│    A. backtest_rank_portfolio()       : 预测值排名 20/80 多空       │
│    B. backtest_optimized_portfolio()  : 均值方差优化 + Ledoit-Wolf  │
│        max μᵀw − λ wᵀΣw                                            │
│        s.t. sum(w)=0, ||w||₁≤2, −5% ≤ wᵢ ≤ 5%                      │
└────────────────────────────────────────────────────────────────────┘
```

核心设计思想：**每一层只做一件事，层与层之间用 DataFrame + 一个 `features` 列表传递**。
所以换模型、加因子、换组合构造方法，都只需要改一层，其余层不用动。

---

## 3. 第一层：DataEngine —— 目标变量的构造

`data_engine.py` 是整个项目里**数学上最关键**的一层，因为目标变量的定义直接决定了因子在哪一个尺度上被评价。

### 3.1 下载与长表化

```python
raw_data = yf.download(self.tickers, start, end, auto_adjust=True,
                       group_by="columns", progress=True)
close  = raw_data.xs("Close",  axis=1, level="Price")
volume = raw_data.xs("Volume", axis=1, level="Price")
```

`yfinance` 返回的是宽表（MultiIndex：Price × Ticker）。这里取 `Close` 和 `Volume` 两层，
然后 `.stack()` 把宽表压成长表（long format）：

| Date | Ticker | Close | Volume |
|---|---|---|---|
| 2015-01-02 | AAPL | 24.68 | 212838000 |
| 2015-01-02 | MSFT | 42.31 | 27913900 |

**为什么必须转成长表？** 因为后面所有因子都要按 `Ticker` 分组做时序运算（rolling / shift），
按 `Date` 分组做截面运算（z-score）。长表配合 `groupby` 是 pandas 里表达这两种运算最自然的形式。

两个细节：
- `auto_adjust=True`：价格已做拆股与分红调整，避免 AAPL 的 4:1 拆股造成假的 −75% 收益。
- `.sort_values(["Ticker", "Date"])`：**这是必须的**。`groupby().shift()` 依赖行内顺序，顺序错一天，动量就会取到错误的历史点。

### 3.2 三个收益率字段

```python
# 当日收益，用于波动率与协方差
self.data["DailyReturn"] = self.data.groupby("Ticker")["Close"].pct_change()

# 个股未来 H 日收益
self.data[f"FutureReturn{H}"] = (
    self.data.groupby("Ticker")["Close"].shift(-H) / self.data["Close"] - 1
)

# 基准未来 H 日收益
spy[f"SPYReturn{H}"] = spy["SPYClose"].shift(-H) / spy["SPYClose"] - 1

# 目标：基准相对超额收益
self.data[f"NetReturn{H}"] = (
    self.data[f"FutureReturn{H}"] - self.data[f"SPYReturn{H}"]
)
```

三个要点：

**(1) `groupby("Ticker")` 是防串行的护栏。**
`.shift(-H)` 如果不用 groupby，最后一只股票的最后一行会取到**下一只股票**的前几行，产生完全错误的标签。
这是本项目里最容易犯、也最难发现的 bug。

**(2) 用减法而不是除法。**
`NetReturn = r_i − r_SPY` 是**算术超额收益**。严格来说对数收益下相除更「正确」，
但在 H=20 这个尺度上差值很小，且减法让后续 OLS 的 β 解释更直接（β 就是因子每变动 1 个标准差对应的超额收益，单位是小数收益率）。

**(3) `NetReturn` 是项目的"灵魂字段"。**
它同时是：
- OLS / LASSO 的 y（`FactorEngine.target`）
- 回测里多空组合的实际收益（`backtest_engine` 里 `factor_engine.target`）

**因此回测收益天然已经是基准中性的**，不需要再减一次 SPY。这一点在写报告时容易被说错。

### 3.3 一个必须知道的时间学问题

`FutureReturn{H}` 使用了**未来 H 天**的价格。这意味着：

> 在日期 t 这一行上，标签用的是 t+H 的信息。

所以 `main.py` 里如果只做 `self.model_data["Date"] < start_date` 这种朴素切分，
**训练集最后 H 天的标签会"看到"验证集早期的价格** —— 这就是经典的 **look-ahead / 标签泄漏**。

`medium-blog` 分支已经修掉了这个问题（见 [5.2](#52-split_data-purge-机制)），
但 `portfolio-optimization` 分支的 `preprocess_engine.py` 仍是朴素切分。
**用旧分支跑研究结论时要记住这个偏差。**

---

## 4. 第二层：FactorEngine —— 因子库

`factor_engine.py` 一共只有 45 行、4 个因子构造函数，但覆盖了动量、波动率、量能、短期反转四大经典因子族。

### 4.1 引擎骨架

```python
class FactorEngine:
    def __init__(self, data_engine):
        self.data_engine = data_engine
        self.target = f"NetReturn{data_engine.forecast_horizon}"   # 自动跟随 H
        self.features = []                                          # 因子登记表
        self.data = data_engine.data.copy()                         # 拷贝，不动原始数据
```

设计上非常干净的两点：

- **`self.target` 由 `forecast_horizon` 派生**，所以 `factor_engine.target` 永远等于当前实验的标签名，
  下游 `LearningEngine` 和 `BacktestEngine` 都不用再关心 H 是多少。
- **`self.features` 自动登记**：每调用一次 `add_xxx()` 就往列表里 append 一次名字。
  这个列表是**全管线的唯一真相来源**：
  - `PreprocessEngine.normalize()` 遍历它生成 `_z` 字段
  - `main.py` 的 factor ablation 遍历它枚举子集
  - LASSO 的系数解读遍历它对齐名字

**这意味着「加一个因子」的代价是：写一个 `add_xxx()` + 在 `main.py` 调一次。** 没有别的注册流程。

### 4.2 因子一：动量 Momentum（跳月动量）

```python
def add_momentum(self, long_horizon, short_horizon):
    grouped_close = self.data.groupby("Ticker")["Close"]
    self.data[f"Momentum{long}_{short}"] = (
        grouped_close.shift(short) / grouped_close.shift(long) - 1
    )
```

`Momentum60_5 = Close[t−5] / Close[t−60] − 1`

**原理：** 每个时间点同时用两个 `shift`，等于把窗口整体**往回推了 `short_horizon` 天**。
所以测的是「从 60 天前到 5 天前」这 55 天的涨跌幅，**故意跳过最近 5 天**。

**为什么跳过最近几天？** 因为最近 1～5 天是**短期反转效应**最强的区间
（流动性冲击、买卖价差回补、超买超卖）。如果不跳过，你就在把
「动量（中期延续）」和「反转（短期回归）」两个方向相反的信号混在一个因子里，系数会被互相抵消掉。

所以 `short_horizon` 实际上是一个**信号净化参数**，把动量区间和反转区间分开，各自单独测算。
这也是为什么库里同时有 `Momentum*_5` 和 `Return5`，它们不冗余，而是刻意分工。

**理论背景：** Jegadeesh & Titman (1993) 的动量效应，以及 Asness / Moskowitz / Pedersen (2013)
关于"动量无处不在"的研究。跳月（skip-a-month）是业界标准做法。

### 4.3 因子二：波动率 Volatility

```python
def add_volatility(self, horizon):
    self.data[f"Volatility{h}"] = (
        self.data.groupby("Ticker")["DailyReturn"]
        .transform(lambda x: x.rolling(h).std())
    )
```

`Volatility20 = DailyReturn 的 20 日滚动标准差`

**为什么用 `transform` 而不是直接 `.rolling()`？**
因为 `rolling()` 在 groupby 之后会返回一个**多层索引**的结果，跟原始行号对不齐。
`transform` 保证输出与输入**行数、索引完全一致**，可以直接赋值成新列。
这是 pandas 里「分组算特征再贴回原表」的标准写法，`add_volume` 里也用了同样的模式。

**符号预期（重要）：** 在结果文件里，`Volatility100` 的系数**一直是正的**
（H=20 时 +0.0038，H=50 时 +0.0105，H=200 时 +0.0654）。
这符合**低波动异象的反面解读**：在只看大市值、已被筛选过的股票池里，
高波动股票在接下来 20 天到 200 天里反而跑赢。这与经典的低波动异象方向相反，
**很可能是因为股票池本身有幸存者偏差 + 都是大市值成长股**，见 [14.1](#141-幸存者偏差)。
这正是本项目值得深挖的一个点。

### 4.4 因子三：量能 Volume（对数相对成交量）

```python
def add_volume(self, horizon):
    self.data[f"AvgVolume{h}"]     = groupby.Volume.transform(rolling(h).mean())
    self.data[f"VolumeRatio{h}"]   = self.data["Volume"] / self.data[f"AvgVolume{h}"]
    self.data[f"LogVolumeRatio{h}"]= self.data[f"VolumeRatio{h}"].transform(np.log)
    self.data = self.data.replace([np.inf, -np.inf], np.nan)
    self.features.append(f"LogVolumeRatio{h}")   # 注意：只登记 Log 版本
```

计算了三列，但**只把 `LogVolumeRatio` 登记进 features**，`AvgVolume` 和 `VolumeRatio` 是中间产物。

**为什么取 log？**
1. **压缩右偏**：成交量比值分布严重右偏（放量时可能是 3 倍甚至 10 倍，缩量时最多只能到 0）。
   `log` 让「放量 2 倍」和「缩量到 1/2」变成 `+0.69` 和 `−0.69`，**变得近似对称**。
2. **稳定方差**：避免单个巨量交易日主导标准化后的截面分布。
3. **让 OLS 的线性假设更合理**：原始比值与未来收益的关系更可能是对数线性的。

**为什么必须 replace inf？**
如果某只股票在 `AvgVolume` 窗口内恰好完全停牌/无成交，均值可能是 0，除出来是 `inf`，
然后 `log(inf) = inf`。`inf` 会污染 z-score 的均值和标准差，让**当天整个截面**的标准化全部失效。
所以这里做了一次清洗 —— 但**这还不够彻底**，见 [14.3](#143-无穷值的清洗位置)。

### 4.5 因子四：短期收益 Return（反转因子）

```python
def add_shortterm_return(self, horizon):
    grouped_close = self.data.groupby("Ticker")["Close"]
    self.data[f"Return{h}"] = self.data["Close"] / grouped_close.shift(h) - 1
```

`Return5 = Close[t] / Close[t−5] − 1`，即**不含跳空的最短动量** = 短期反转信号。

和 `Momentum60_5` 的区别：`Return5` 用的是**当天收盘价**，包含最近 5 天的全部信息；
`Momentum60_5` 刻意把最近 5 天排除。两者互为补充，不是重复。

在 LR 结果里 `Return5_z` 的系数是 −0.0002、p = 0.731（**不显著**），
而在 H=20 的 LASSO 里它根本没被选进来（只有 1～3 个因子幸存）。
这说明在本股票池、本区间内**短期反转信号很弱**。

### 4.6 建表

```python
def build_model_data(self):
    model_data = self.data[["Date","Ticker"] + self.features + [self.target]].copy()
    model_data = model_data.dropna()
    return model_data
```

只保留 `Date`、`Ticker`、登记过的因子、目标。一次 `dropna()` 把所有滚动窗口暖机期（前 100 天）和标签尾部（最后 H 天）的行全部清掉。

**副作用提醒：** `dropna()` 是**整行**丢弃。如果在 LASSO 里把 `Volatility100` 加进来，
那前 100 天的数据全没了；如果某个因子暖机期是 200 天，整个早期样本都会消失。
这就是为什么 `main.py` 里的 `Train_Rows` 随 H 变化（119196 → 111946），
因为 H 越大，尾部被 `shift(-H)` 挖掉的行越多。

### 4.7 `medium-blog` 的 15 因子库

`main.py` 里实际调用的组合（这是**当前的主力因子集**）：

| 因子族 | 具体因子 | 经济含义 |
|---|---|---|
| 动量 | `Momentum10_5`, `Momentum20_5`, `Momentum60_5`, `Momentum100_5` | 4 个不同时间尺度的中期动量，检验动量是短期现象还是长期现象 |
| 波动率 | `Volatility10`, `Volatility20`, `Volatility60`, `Volatility100` | 4 个尺度的风险代理 |
| 量能 | `LogVolumeRatio10`, `LogVolumeRatio20`, `LogVolumeRatio60`, `LogVolumeRatio100` | 4 个尺度的相对成交量 |
| 短期收益 | `Return5`, `Return10`, `Return20` | 3 个尺度的短期反转 |

**为什么要为每个因子族铺开多个时间尺度？**
这不是为了"多凑因子"，而是为了回答一个具体问题：
**这个因子的有效尺度在哪？** 如果 `Momentum60_5` 显著而 `Momentum10_5` 不显著，
说明动量需要至少 60 天的形成期。这比只放一个 60 天动量能提供的信息多得多。

副作用是**因子之间高度相关**（`Momentum10_5` 和 `Momentum20_5` 相关系数很高），
这会让 OLS 系数不稳定 —— 而这恰恰是引入 LASSO 的动机（见 [6.2](#62-lasso为什么需要它)）。

---

## 5. 第三层：PreprocessEngine —— 截面标准化与切分

### 5.1 normalize() —— 为什么必须做截面 z-score

```python
for feature in self.factor_engine.features:
    new_name = feature + "_z"
    self.model_data[new_name] = (
        self.model_data.groupby("Date")[feature]
        .transform(lambda x: (x - x.mean()) / x.std())
    )
    self.standardized_features.append(new_name)
    self.model_data = self.model_data.replace([np.inf, -np.inf], np.nan)
    self.model_data = self.model_data.dropna(
        subset=self.standardized_features + [self.factor_engine.target]
    ).reset_index(drop=True)
```

**关键点一：`groupby("Date")`，不是全局标准化。**

```python
z_{i,t} = (x_{i,t} − mean_j(x_{j,t})) / std_j(x_{j,t})
```

- 全局标准化回答的是：「AAPL 今天 0.3 的动量，在**历史上所有股票所有时间**里算高吗？」
- 截面标准化回答的是：「AAPL 今天 0.3 的动量，**在今天的 49 只股票里**算高吗？」

**后者才是我们要的。** 因为：
1. 我们做的是**截面排序**，只需要知道"今天谁比谁强"。
2. 它自动**消除市场层面的时间变动**。2015 年美股整体波动率和动量水平与 2020 年疫情完全不同，
   全局标准化会把"2015 年 vs 2020 年"的差异混进因子值里，而截面标准化把它干净地剔除。
3. 它让不同因子**可比**：`Volatility100` 的量纲是 0.01 级别，`LogVolumeRatio` 是 0.1 级别，
   标准化后两者都是均值 0、标准差 1，系数可以直接横向比较大小。

**关键点二：循环内每加一个因子就重算一次 `dropna`。**

```python
self.model_data = self.model_data.dropna(subset=self.standardized_features + [target])
```

这段写在 `for` 循环**内部**，所以第 2 个因子标准化时会用到第 1 个因子已经清干净的行。
逻辑上等价于最后统一 dropna 一次，但效率低（每次都重算），且**在循环内 `reset_index` 后再按位置切片会有隐患**（下一段讲）。

**关键点三：`std()` 用的是 pandas 默认的样本标准差（ddof=1）。**
如果某一天只有 1 只股票有数据（理论可能，实际上不会），`std` 是 `NaN`，该行会被 drop。
更要提防的是**同一天内因子值完全相同**（比如某天全市场成交量都是 0），`std=0` → 除零 → `inf`。
所以紧接着的 `replace([inf,-inf], nan)` 是必要的护栏。

### 5.2 split_data() —— purge 机制

这是 `medium-blog` 分支相对旧版**最重要的修正**：

```python
forecast_horizon = self.factor_engine.data_engine.forecast_horizon

dates = self.model_data["Date"].drop_duplicates().sort_values().reset_index(drop=True)

dates_before_validation  = dates[dates < start_date]
dates_before_test        = dates[dates < validation_date]

train_cutoff      = dates_before_validation.iloc[-forecast_horizon]
validation_cutoff = dates_before_test.iloc[-max_horizon]

train      = model_data[model_data["Date"] <  train_cutoff]
validation = model_data[(model_data["Date"] >= start_date) &
                        (model_data["Date"] <  validation_cutoff)]
test       = model_data[model_data["Date"] >= validation_date]
```

**它在做什么？** 把训练集的结束日期**往前推 H 天**，把验证集的结束日期**往前推 max_horizon 天**。

**为什么？** 回到 [3.3](#33-一个必须知道的时间学问题)：

| 日期 | 标签 NetReturn20 用到的价格 |
|---|---|
| 2021-12-15 | 2022-01-13 |
| 2021-12-31 | 2022-01-31 |

如果不 purge，训练集最后一行的标签来自 **2022-01-31**，
而验证集从 **2022-01-01** 就开始了 —— 模型在验证期"提前看过了"未来。
这会让 OOS R² 被系统性高估。

**`train_cutoff` 用的是 `iloc[-forecast_horizon]`**：从 `start_date` 往前数第 H 个**交易日**。
用交易日而不是日历日，是因为 H 的单位是交易日（H=20，不是 20 个自然日）。
`validation_cutoff` 用 `max_horizon`（所有实验 H 的最大值，本例 200），
是为了让**同一套切分在不同 H 之间可比**，同时也是对 H=200 这个最大标签留出足够间隔。

**`test` 集没有被 purge** —— 它直接从 `validation_date` 开始。
如果将来要用 test 做最终评估，这里同样需要 −H 的间隔，否则 test 的第一批标签也会与 validation 重叠。

### 5.3 重叠样本问题（overlapping labels）

H=20 意味着日期 t 和 t+1 的标签共享 19 天的价格路径。
这带来两个后果：

1. **有效样本数被高估**。118446 行训练数据，独立观测数大约只有 `118446/20 ≈ 5900` 个。
2. **残差高度自相关**。相邻样本的误差同向，OLS 的经典标准误会低估，t 值虚高。

**项目已做的应对：** `LearningEngine.train()` 里用了 `cov_type="cluster"` 按日期聚类：

```python
sm.OLS(y_train, x_train).fit(cov_type="cluster", cov_kwds={"groups": self.train_data["Date"]})
```

**聚类按 `Date` 是不够的。** 同一天的股票之间独立（聚类正好处理这个），
但**不同天的重叠窗口**问题依然存在。正确的做法应该是按「重叠块」聚类
（比如把 t 到 t+H−1 归为一块），或者用 `jump = H` 做非重叠采样。
项目在**验证阶段**做到了非重叠（见 [6.3](#63-validate-非重叠验证)），但**训练阶段没有**。
这是下一个可以改进的点。

---

## 6. 第四层：LearningEngine —— LR 与 LASSO

### 6.1 LR：带聚类稳健标准误的 OLS

```python
if self.model_type == "LR":
    x_train = sm.add_constant(x_train)
    self.model = sm.OLS(y_train, x_train).fit(
        cov_type="cluster", cov_kwds={"groups": self.train_data["Date"]}
    )
```

模型就是最朴素的线性回归：

```
NetReturn_H = β₀ + β₁·Momentum_z + β₂·Volatility_z + β₃·LogVolumeRatio_z + β₄·Return_z + ε
```

`add_constant` 加截距项。截距 β₀ 有经济含义：**当所有因子都处于截面均值时，股票的预期超额收益**。
`results1.txt` 里 β₀ = 0.0043，z = 16.6 —— 在 20 天（约一个月）尺度上，
"所有因子都平均的股票"预期跑赢 SPY **0.43%**。这个正截距主要来自**幸存者偏差**
（股票池是现在仍然存在的大盘股，本身就是过去十年跑赢的赢家）。作者在 commit message 里已经指出了这点。

**实际拟合结果（`results1.txt`，H=20，4 因子）：**

| 变量 | 系数 | z | P>\|z\| | 解读 |
|---|---|---|---|---|
| const | 0.0043 | 16.624 | 0.000 | 幸存者偏差的体现 |
| `Momentum60_5_z` | 0.0014 | 2.705 | 0.007 | 动量显著，1 个标准差动量 → +0.14% 超额（20 天） |
| `Volatility20_z` | 0.0047 | 8.646 | 0.000 | **波动率是最强因子** |
| `LogVolumeRatio20_z` | −0.0006 | −1.972 | 0.049 | 勉强显著，量能高 → 收益低 |
| `Return5_z` | −0.0002 | −0.344 | 0.731 | **不显著** |

整体 R² = 0.006。**这个数字低得让人不安，但对于月度截面收益预测是完全正常的**：
在学术文献里，月度股票收益的截面 R² 通常在 1%～3% 之间（Gu, Kelly & Xiu 2020 的机器学习论文里，
最好的神经网络 OOS R² 也就 0.4% 左右）。**不要用"低 R²"否定因子。
真正的评价标准是 OOS R² 的符号、以及能否转化为正的多空收益和 Sharpe。**

### 6.2 LASSO：为什么需要它

`main.py` 的因子库有 15 个高度相关的因子。如果直接上 OLS：

1. **多重共线性**：`Momentum10_5`、`Momentum20_5`、`Momentum60_5` 相关性极高，
   系数会剧烈摆动，符号甚至可能反转，完全无法解读。
2. **过拟合**：训练集 11.8 万行、15 个因子，OLS 会利用共线结构的噪声来拟合。
3. **无法自动选择**：OLS 不会把没用的因子系数变成 0，你只能靠 p 值人工筛。

LASSO（Tibshirani 1996）在损失函数上加 L1 惩罚：

```
minimize  (1/2n)·||y − Xβ||² + α·||β||₁
```

**L1 惩罚的几何性质会直接把不重要的系数压到恰好 0**，所以 LASSO 天然是一个**因子选择器**。
这不是精度上的小改进，而是**改变了输出的性质**：OLS 输出 15 个系数（全部非零），
LASSO 输出一个稀疏的因子子集。你打开 `offset_results_H20.csv` 看到的
`Selected_Factors` 就是这么来的。

```python
elif self.model_type == "LASSO":
    self.model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=10000)
    self.model.fit(x_train, y_train)
```

- `alpha` 越大 → 惩罚越重 → 越稀疏。`main.py` 搜索 8 个值：`1e-5` 到 `1e-2`。
- `max_iter=10000` 是必要的：坐标下降法在因子高度相关时收敛很慢，默认 1000 次常常不收敛（会警告）。
- ⚠️ **LASSO 前没有再做标准化**：因为 `PreprocessEngine` 已经做了截面 z-score，
  `fit_intercept=True` 处理了均值。但严格来说 LASSO 的惩罚是对**系数**对称的，
  要求各特征**方差相等**。截面 z-score 让每日方差为 1，但**跨全部日期汇总后**方差未必精确为 1，
  所以惩罚有轻微的不对称。这是个小瑕疵。

**注意 `main.py` 里怎么提取被选中的因子：**

```python
for feature, coefficient in zip(features, model.coef_):
    if abs(coefficient) > 1e-10:          # 阈值判断
        selected_factors.append((feature, coefficient))
```

用 `> 1e-10` 而不是 `!= 0`，是**数值稳健性**的做法：LASSO 用坐标下降求解，
被"压到 0"的系数在浮点运算下可能是 `1e-17` 这种极小非零值。加个阈值过滤才可靠。

`zip(features, model.coef_)` 依赖 `features = factor_engine.features` 的顺序与
`standardized_features` 的顺序一致 —— 这在 `normalize()` 里是有保证的（同样遍历 `features`）。

### 6.3 validate() —— 非重叠验证

这是 `medium-blog` 分支引入的第二个重要修正：

```python
if self.model_type == "LR":
    x_validation = sm.add_constant(x_validation, has_constant="add")
    self.validation_data["PredictedReturn"] = self.model.predict(x_validation)

elif self.model_type == "LASSO":
    self.validation_data["PredictedReturn"] = self.model.predict(x_validation)

# 关键：按 jump 采样，只保留部分日期
validation_dates = self.validation_data["Date"].drop_duplicates().sort_values()
sampled_dates = validation_dates.iloc[self.offset::self.jump]

independent_validation = self.validation_data[self.validation_data["Date"].isin(sampled_dates)]

actual    = independent_validation[self.factor_engine.target]
predicted = independent_validation["PredictedReturn"]

oos_r2 = 1 - ((actual - predicted)**2).sum() / (actual**2).sum()
```

**两个关键设计：**

**(1) `iloc[offset::jump]` —— 每隔 H 天取一天。**

`jump = forecast_horizon = H`，所以对 H=20 的实验，采样后的日期是
`offset, offset+20, offset+40, ...`。

这样做的目的是让**被评估的样本之间不再重叠**：
第 1 个样本的标签覆盖 `[t, t+20]`，第 2 个样本覆盖 `[t+20, t+40]`，正好首尾相接不重叠。

**代价是样本量暴跌**：H=20 时验证行数从 3 万多降到 **3350**（67 个独立日期）；
H=200 时只有 **300 行 / 6 个独立日期**。
这直接解释了为什么 horizon 越大 R² 越高 —— 见 [11.2](#112-为什么-h-越大-r²-越高)。

**(2) `offset` 参数 —— 稳健性检验的核心。**

`offset` 控制从验证期的第几天开始采样。`offset=0,2,4,...,19` 就是 10 个
不同的、互不重叠的采样相位。

**如果因子是真实有效的，那么无论从哪天开始采样，OOS R² 都应该差不多。
如果 R² 随 offset 剧烈波动，说明这个"效果"只是采样相位的巧合。**

这正是 `main.py` 里那个三层循环的意义所在，也是作者最值得肯定的设计之一。

**(3) OOS R² 的定义：**

```
R²_OOS = 1 − Σ(y − ŷ)² / Σy²
```

注意分母是 **`Σy²` 而不是 `Σ(y − ȳ)²`**。

- 分母用 `Σ(y−ȳ)²` 是常规 R²（"模型比只猜均值好多少"）。
- 分母用 `Σy²` 是**相对"预测 0"的基准**。

因为 y 是 `NetReturn = r_i − r_SPY`，**0 是一个有意义的基准**
（"预测所有股票都跟 SPY 一样"）。在金融里这是标准做法
（Campbell & Thompson 2008 的 R²_OOS）。所以这里的 R² 可以直接读作：
**"我的模型比'认为所有股票都等于基准'好多少。"**

`R²_OOS ≤ 0` 意味着模型的预测还不如直接预测 0。**这是金融机器学习里最常见的结果**，
不用因此沮丧，但一定要如实报告。

### 6.4 一个真实的 bug

`validate()` 里有这行：

```python
self.validation_data["PredictedReturn"] = self.model.predict(x_validation)
```

`self.validation_data` 是调用方传入的 DataFrame。`main.py` 里用
`validation_data.copy()` 传参（`LearningEngine(..., validation_data.copy(), ...)`），
**这是对的**，避免了对原始验证集的写入污染。

但 `LR` 分支和 `LASSO` 分支的 `validate()` 结构不同：
LASSO 分支里有采样逻辑，LR 分支**没有**（LR 的 `validate()` 直接返回全部验证数据）。
所以 `main.py` 的 LR 分支跑 factor ablation 时，`independent_validation` 里的行数
是全部验证行（45300 行 / 906 个日期），而 LASSO 分支只有几千行。

**这导致两条线的 R² 不可直接比较**：
`results.csv` 里 LR 的 H=20 R² = 0.0034，而 LASSO 的 H=20 是 0.0019。
这个差异**既来自模型不同，也来自评估样本不同**（LR 用了重叠样本）。
比较时必须说明这一点。

### 6.5 另一个 bug：LR 分支的 `validate()` 缩进

`learning_engine.py` 的 LR 分支中，`x_validation` 的赋值在 `if self.model_type == "LR":` 内部，
但 `return` 语句在 `if` **外部**，且 `LR` 分支的 `oos_r2` 计算依赖 `actual`/`predicted`
这两个只在 LR 分支定义的变量。如果 `model_type` 既不是 "LR" 也不是 "LASSO"，
`validate()` 会因为 `oos_r2` 未定义而抛 `UnboundLocalError`。
当前只有这两个模型所以不会触发，但**扩展新模型时要注意**。

另外，文件里 `validate()` 的 LR 分支用的是**制表符缩进**（mix of tabs and spaces），
而其他地方是空格。Python 3 会拒绝混合缩进 —— 文件能跑说明 tab 被一致地用在那一块。
**改这段代码时小心，编辑器可能把它"格式化"成空格而导致 `TabError`。**

---

## 7. 第五层：BacktestEngine —— 两种组合构造

### 7.1 construct_portfolio() —— 把预测变成可交易的时点

```python
validation_dates = self.validation["Date"].drop_duplicates().sort_values()
self.rebalance_dates = validation_dates.iloc[::self.forecast_horizon]
self.portfolio_data  = self.validation[self.validation["Date"].isin(self.rebalance_dates)].copy()
self.periods_per_year = 252 / self.forecast_horizon
```

**`iloc[::H]`：每 H 个交易日取一个调仓日。** 逻辑闭环：
预测的是**未来 H 天**的收益，那就应该**每 H 天调一次仓**，
让持仓期与预测期对齐。H=20 就是约每月调仓一次。

**`periods_per_year = 252/H`：** 这是年化 Sharpe 的核心参数。
H=20 时，一年有 `252/20 = 12.6` 个非重叠持仓期。

### 7.2 方法 A：20/80 排名多空组合

```python
self.portfolio_data["Rank"] = self.portfolio_data.groupby("Date")["PredictedReturn"].rank(pct=True)
self.portfolio_data["Position"] = 0
self.portfolio_data.loc[self.portfolio_data["Rank"] > 0.8, "Position"] =  1
self.portfolio_data.loc[self.portfolio_data["Rank"] < 0.2, "Position"] = -1

long_returns  = portfolio_data[Position ==  1].groupby("Date")[target].mean()
short_returns = portfolio_data[Position == -1].groupby("Date")[target].mean()

backtest["LongShortReturn"] = long_returns - short_returns
backtest["CumulativeReturn"] = (1 + backtest["LongShortReturn"]).cumprod()

sharpe = backtest["LongShortReturn"].mean() * sqrt(periods_per_year) / backtest["LongShortReturn"].std()
```

**`rank(pct=True)` 在 `groupby("Date")` 内**：把每一天的预测值转成 0～1 的百分位。
这是**截面排序**，自动消除了预测值的绝对水平差异
（比如模型在某段时间系统性预测偏高，排名法不受影响）。

**等权多空：** 多头取预测值最高的 20%（49 只里约 10 只），空头取最低的 20%。
组合内**等权**，没有用优化器。

**为什么这个朴素方法很重要？** 它是**优化组合的对照组**。
如果均值方差优化跑不赢简单的 20/80 等权，那说明优化器的估计误差
（μ 和 Σ 都不准）吃掉了理论上的收益 —— 这是非常常见的结论，
也是 DeMiguel, Garlappi & Uppal (2009) "1/N 组合难以击败"的经典发现。

**Sharpe 年化公式：** `mean × sqrt(periods_per_year) / std`。
⚠️ 这个式子里 `std` 是**每个持仓期收益的标准差**，
乘以 `sqrt(252/H)` 的年化假设是「各期收益独立同分布」。
对于非重叠的 H 天收益，这个假设基本成立，所以公式是对的。
但如果误用了重叠采样，标准差会被低估、Sharpe 会被高估 —— 这里用 `rebalance_dates` 采样，是对的。

### 7.3 方法 B：均值方差优化组合

这是整个项目**技术含量最高**的部分：

```python
for current_date in self.rebalance_dates:
    current_portfolio = self.portfolio_data[self.portfolio_data["Date"] == current_date].copy()
    current_tickers   = current_portfolio["Ticker"].tolist()
    mu                = current_portfolio["PredictedReturn"].values        # 期望收益 = 模型预测

    returns_wide = data.pivot(index="Date", columns="Ticker", values="DailyReturn").dropna()
    past_returns = returns_wide.loc[returns_wide.index < current_date, current_tickers].tail(lookback)

    # —— 协方差估计 ——
    sigma_daily = past_returns.cov().values      # 普通样本协方差（被下面覆盖）
    sigma = self.forecast_horizon * sigma_daily

    lw = LedoitWolf(); lw.fit(past_returns)      # Ledoit-Wolf 收缩（实际使用的）
    sigma_daily = lw.covariance_
    sigma = self.forecast_horizon * sigma_daily

    # —— 优化 ——
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, sigma))
    constraints = [cp.sum(w) == 0, cp.norm1(w) <= 2, w <= 0.05, w >= -0.05]
    problem = cp.Problem(objective, constraints)
    problem.solve()
    optimal_weights = w.value

    current_portfolio["Weights"] = optimal_weights
    predicted_portfolio_return = (current_portfolio["Weights"] * current_portfolio["PredictedReturn"]).sum()
    realized_portfolio_return  = (current_portfolio["Weights"] * current_portfolio[self.factor_engine.target]).sum()
```

**逐块解释：**

**(1) `mu` 直接就是模型预测。** 这是 Markowitz 框架的关键接口：
因子模型输出 μ，组合优化器把它和风险模型 Σ 结合成权重。
**μ 的质量决定一切** —— 如果 μ 是噪声，再好的 Σ 估计也救不了（"garbage in, garbage out"），
这就是为什么前面花那么大力气做因子和 OOS 验证。

**(2) 只用 `current_date` 之前的数据估 Σ。**

```python
past_returns = returns_wide.loc[returns_wide.index < current_date, current_tickers].tail(lookback)
```

`index < current_date` 是**严格的防止前视**：调仓日当天及以后的数据一律不用。
`lookback = 60` 天，用最近 60 个交易日估协方差。
**这是逐日滚动的 walk-forward 逻辑，没有用到任何未来信息。** 这点做得很干净。

`current_tickers` 只取**当天有预测的股票**（约 49 只），保证 `mu`、`Σ`、`w` 三者维度对齐。

**(3) 为什么必须用 Ledoit-Wolf 收缩？**

这是整个回测里最容易被忽视、但影响最大的技术细节。

**问题：** 49 只股票需要估计 `49×50/2 = 1225` 个协方差参数，
但只有 `lookback = 60` 天数据。**样本数 60 < 参数数 1225**。
在这种"高维小样本"下，样本协方差矩阵是**奇异或接近奇异**的，
特征值极度分散（最大特征值巨大，最小特征值接近 0），
优化器会**利用这些虚假的极端特征值**下出极其极端的权重
（比如在几乎零方差的方向上放大杠杆）—— 这是 Markowitz 优化在实盘中臭名昭著的不稳定性来源。

**解法：** 把样本协方差向一个**结构化的、良态的目标矩阵**收缩：

```
Σ_shrunk = (1 − δ)·S + δ·F
```

- `S` = 样本协方差矩阵（无偏但噪声大）
- `F` = 收缩目标，Ledoit-Wolf 用的是**对角化目标**（只保留方差、把协方差全部设为 0）
- `δ` ∈ [0,1] = 收缩强度，由 Ledoit & Wolf (2004) 给出**闭式最优解**（不用交叉验证）

`lw.shrinkage_` 就是估计出来的 δ。代码里 `print("amount of shrinkage: ", lw.shrinkage_)`
把这个值打印出来了，**这是很好的调试习惯** —— δ 接近 1 就说明样本协方差基本不可信，
接近 0 说明样本信息充足。

**"收缩"的统计本质是 bias-variance tradeoff**：
主动接受一点偏差（把 Σ 往对角矩阵拉），换取方差的大幅降低。
在有真实信号的场景里这个交换总是值得的。

**(4) `sigma = H × sigma_daily` —— 时间尺度的匹配。**

优化问题是关于**未来 H 天**的，所以 Σ 必须是 H 天的协方差。
代码用简单的线性放大 `H × Σ_daily`。

**这个近似假设日收益独立同分布**（`Var(sum of H days) = H × Var(1 day)`），
忽略了**自相关**。对于 H=200，这个线性放大可能显著高估或低估风险。
更严谨的做法是直接用 H 天重叠收益估协方差。
**这是一个可以改进的点。**

注意 `risk_aversion = 5` 是与 H 天方差的相对关系，H 变了应该重新调 λ。

**(5) 三个约束，每一个都有明确目的：**

| 约束 | cvxpy 写法 | 经济含义 |
|---|---|---|
| `sum(w) = 0` | `cp.sum(w) == 0` | **美元中性**：多头总金额 = 空头总金额。组合对市场整体涨跌免疫，只赚"选股"的钱。与预测目标 `NetReturn`（超额收益）一致 |
| `\|\|w\|\|₁ ≤ 2` | `cp.norm1(w) <= 2` | **总杠杆 ≤ 200%**（多头 100% + 空头 100%）。控制总敞口，也是防止数值上无限放大的护栏 |
| `−0.05 ≤ wᵢ ≤ 0.05` | `w <= 0.05, w >= -0.05` | **单票 ±5%**。避免模型把宝全压在一两只股票上，规避特异性风险 |

这三个约束配合，产生的就是**分散化的美元中性多空组合**。
`cp.quad_form(w, sigma)` 要求 `sigma` 半正定 —— **这正是需要 Ledoit-Wolf 的原因**：
若样本协方差有负特征值，`quad_form` 不是凸的，cvxpy 会报 `DCPError`。

**(6) 两个收益指标，不要混淆：**

```python
predicted_portfolio_return = (Weights * PredictedReturn).sum()   # 模型的自我预测
realized_portfolio_return  = (Weights * NetReturn).sum()         # 真实实现
```

`predicted` 是"模型以为会赚多少"，`realized` 是"实际赚了多少"。
**两者的累积曲线背离程度就是模型过度自信的度量。**
代码把两条累积曲线都算出来了（`CumulativePredictedReturn`、`CumulativeRealizedReturn`），
**这是非常好的一步** —— 一条上扬的预测曲线配一条平坦的实际曲线，
是"过拟合"最直观的展示。仓库 `results/` 目录里的两张截图应该就是这个图。

**(7) 性能问题：** `data.pivot(...)` 写在 `for current_date` **循环内部**，
每次迭代都把 26MB 的长表重新透视一遍（49 只 × 2700 天的宽表）。
这个 pivot 应该提到循环外做一次。当前数据量下能跑完，但**扩到几百只股票会明显变慢**。

---

## 8. 实验模式一：Forecast Horizon 扫描

`main.py` 的主循环：

```python
forecast_horizon_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200]
max_horizon = max(forecast_horizon_values)     # = 200

for forecast_horizon in forecast_horizon_values:
    jump = forecast_horizon                    # 调仓/采样周期 = H
    # 1. 重新下载数据（目标变量随 H 改变）
    # 2. 重建 15 因子库
    # 3. 标准化 + 带 purge 的切分
    # 4. LASSO: 遍历 offset × alpha
    # 5. 汇总 mean/median/std/min/max R²
full_results.to_csv(f"forecast_horizon_{learning_model}_offset_robustness.csv")
```

**这个循环回答的是 `ideas.txt` 里的问题 5：**
> "how does performance of the model change with respect to the future horizon"

**几个设计要点：**

- **`max_horizon = 200` 在循环外算一次**，传给 `split_data()` 作为统一的 purge 间隔。
  这保证了**所有 H 用的是同一套 train/validation 切分**。
  如果每个 H 用自己的 horizon 做 purge，不同 H 的数据集就不一样了，比较就不公平。
  这是一个**很关键的实验设计细节**。
- **`jump = forecast_horizon`** 让验证采样周期跟随 H，保证"非重叠"。
- 整个循环跑 17 个 H × 10 个 offset × 8 个 alpha = **1360 次 LASSO 拟合**，
  每次都在 11.8 万行 × 15 因子上。这是 `main.py` 要跑很久的原因。
- ⚠️ **每个 H 都重新 `yf.download`**：数据是一样的（只有标签不同），
  **下载应该提到循环外做一次**。这是当前最大的性能浪费。

---

## 9. 实验模式二：Factor Ablation

`main.py` 的 `learning_model == "LR"` 分支：

```python
for n in range(1, len(features) + 1):
    for subset in combinations(features, n):
        standardized_subset = [f + "_z" for f in subset]
        learning_engine = LearningEngine("LR", train_data, validation_data.copy(),
                                         test_data, standardized_subset, factor_engine, jump, 0, 0)
        model = learning_engine.train()
        independent_validation, oos_r2 = learning_engine.validate()
        results.append({"Factors": subset, "N_Factors": len(subset), "Validation_R2": oos_r2})
results.sort_values("Validation_R2", ascending=False)
best = results.iloc[0]
```

**这就是因子消融（ablation study）：枚举所有可能的因子子集，看哪个组合的 OOS R² 最高。**

**(1) 为什么用 LR 而不是 LASSO 做消融？**
LASSO 本身已经做因子选择，但它**内部**选择的过程无法告诉你
"去掉 Volatility20 会损失多少"。穷举子集的 OLS 能给出**每个子集的独立评分**，
这才是消融研究要的。

**(2) ⚠️ 组合爆炸 —— 这是当前代码的致命限制。**

`combinations(features, n)` 遍历所有子集，总数是 `2^15 − 1 = 32767` 个。
每个都要在 11.8 万行上 fit 一次 OLS。**这在实际时间上是不可行的**
（单次 OLS 假设 0.5 秒，32767 次 = 4.5 小时，乘以 11 个 H 就更久）。

**证据：** `results.csv` 里每个 H 只列出了 1 个"best"，但文件里
`Train_Rows` 随 H 递减（59650 → 54900），说明**这个 LR 消融是用一个更小的因子集跑的**
（从 `results.csv` 的 `Factors` 列看，候选只有 `Volatility20`、`LogVolumeRatio20`、`Return5`、
`Momentum60_5` 这几个），而不是 15 个。
而且 `Validation_Rows` 恒为 45300、`Validation_Dates` 恒为 906 —— 这正是
[6.4](#64-一个真实的-bug) 说的"LR 分支没有非重叠采样"的证据。

**所以 `results.csv` 是早期小因子集实验的产物，不能和 `forecast_horizon_LASSO_*.csv` 直接对比。**

**(3) 这正是引入 LASSO 的根本动机。**
LASSO 用**一次拟合**同时完成"选因子 + 估系数"，替代了 `2^15` 次穷举。
从"穷举不可行"到"一次求解"，这是 LASSO 相对 OLS 在因子库变大后的**决定性优势**。

**(4) 如果确实要做消融，正确做法是：**
- **前向逐步选择（forward stepwise）**：一次加一个最优因子，只做 `15 + 14 + ... = 120` 次拟合
- **留一消融（leave-one-out）**：从全因子集出发，每次去掉一个，做 15 次拟合
- **LASSO 的约束路径**：用 `LassoCV` 一次性得到所有 α 下的稀疏解

留一消融最有解释力：「去掉哪个因子让 OOS R² 掉得最多，它就是最重要的因子」。

### 9.1 实跑结果（`results.csv`）

| Forecast Horizon | 最优因子子集 | Validation R² |
|---|---|---|
| 5 | `Volatility20` | 0.00105 |
| 10 | `Volatility20` | 0.00174 |
| 20 | `Volatility20`, `LogVolumeRatio20` | 0.00337 |
| 30 | `Volatility20`, `Return5` | 0.00663 |
| 40 | `Volatility20` | 0.00612 |
| 50 | `Volatility20`, `Return5` | 0.01033 |
| 60 | `Volatility20` | 0.01660 |
| 70 | `Volatility20`, `LogVolumeRatio20`, `Return5` | 0.01840 |
| 80 | `Volatility20` | 0.02246 |
| 90 | `Volatility20`, `Return5` | 0.01720 |
| 100 | `Volatility20` | 0.02235 |

**最重要的观察：`Volatility20` 出现在 11/11 个 H 的最优子集中，而且经常是单独出现。**

**这强烈说明波动率是本项目股票池里最重要、最稳健的因子。**
这个结论在 LASSO 那边也得到了独立验证（见 [11.3](#113-lasso-选出来的因子)）。

---

## 10. 实验模式三：Alpha 调参与 Offset 稳健性

`main.py` 里的三层嵌套：

```python
n_offsets = 10
number_of_offsets = min(n_offsets, forecast_horizon)
offset_values = np.unique(np.linspace(0, forecast_horizon - 1, number_of_offsets, dtype=int))

for offset in offset_values:                    # 第 1 层：相位（稳健性）
    alpha_results = []
    for alpha in alpha_values:                  # 第 2 层：正则强度
        learning_engine = LearningEngine("LASSO", train, validation.copy(), test,
                                         standardized_features, factor_engine, jump, alpha, offset)
        model = learning_engine.train()
        independent_validation, oos_r2 = learning_engine.validate()
        # 记录被选中的因子
        alpha_results.append({"Alpha": alpha, "Validation_R2": oos_r2,
                              "Selected_Factors": selected_factors, "N_Selected": len(selected_factors), ...})
    # 每个 offset 内的最优 alpha
    best_alpha_result = alpha_results.sort_values("Validation_R2", ascending=False).iloc[0]
    offset_results.append({"Offset": offset, "Best_Alpha": ..., "Validation_R2": ..., ...})
```

**(1) `np.linspace(0, H-1, n_offsets, dtype=int)` 生成什么？**

对 H=20：`linspace(0, 19, 10)` → `[0, 2, 4, 6, 8, 10, 12, 14, 16, 19]`
（`np.unique` 去掉了浮点转整数产生的重复值，所以 H=5 时只有 5 个 offset）。

对 H=5：`number_of_offsets = min(10, 5) = 5` → `[0,1,2,3,4]`，正好是 5 个相位。

**为什么要 `min(n_offsets, H)`？** 因为 `iloc[offset::H]` 的采样相位只有 H 种
（offset 从 0 到 H−1），offset=H 和 offset=0 是同一个采样。这是正确的边界处理。

**(2) 稳健性检验的统计逻辑：**

对每个 offset，先在该相位内部选出最优 α，得到 `Validation_R2`。
于是 `offset_results` 里有 10 个 R² 值（同一模型、同一数据、只是采样起点不同）。

```python
mean_r2   = offset_results["Validation_R2"].mean()
median_r2 = offset_results["Validation_R2"].median()
std_r2    = offset_results["Validation_R2"].std()
min_r2    = offset_results["Validation_R2"].min()
max_r2    = offset_results["Validation_R2"].max()
```

**`std_r2` 是这里最有信息量的数字。** 它度量了「结果对采样相位的敏感度」：

- `std_r2` 小 → 结论稳健，换个起点 R² 也差不多 → **因子很可能是真实的**
- `std_r2` 大 → 结论依赖相位 → **很可能是运气**

**看 H=20 的实测：** mean = 0.00297，std = 0.00079，min = 0.00187，max = 0.00446。
变异系数约 27%，**偏高**。R² 从 0.19% 到 0.45% 波动 2.4 倍 —— 说明 H=20 的信号**不太稳健**。

**看 H=150 的实测：** mean = 0.0509，std = 0.0100。
绝对值大得多，但相对变异（20%）**比 H=20 更小**。不过要注意 H=150 时只有 10 个采样相位、
每个相位样本数极少，统计意义有限。

#### 10.3 一个方法论上的隐患 best_alpha 是在验证集上选的

```python
best_alpha_result = alpha_results.sort_values("Validation_R2", ascending=False).iloc[0]
```

**用验证集选择 α，然后报告验证集的 R²，这是"在验证集上过拟合"。**
选出的 `Validation_R2` 是 8 个 α 中的**最大值**，天然比真实水平偏高（选择性偏差）。

**正确做法：**
- 用**训练集内部**的交叉验证选 α（`LassoCV`），然后在验证集上看一次结果；或
- 把验证集再切成两半：一半选 α，一半报告 R²；或
- 报告**所有 α 的完整结果**（不选最优），看结论是否对 α 不敏感。

**(4) 从结果文件能看出的 α 行为：**

`offset_results_H20.csv` 里所有 offset 的 `Best_Alpha` 都是 `0.001`。
`H=100` 时出现了 `1e-5`、`3e-5`、`0.001`、`0.005`、`0.01` 各种值。

**规律：H 越大，选出的最优 α 越分散；H 越小，最优 α 越集中在 1e-3 附近。**
这符合直觉：H 大时验证样本少（H=200 时只有 300 行），
α 的选择更容易受噪声影响，所以会出现"时而要重惩罚、时而要轻惩罚"的不稳定。

**更有意义的观察：以 `N_Selected` 看稀疏度的稳定性。**

H=20 时：offset 0→1 个因子，offset 2~19→**恒定 3 个因子**（`Momentum60_5`, `Volatility100`, `LogVolumeRatio60`），
而且系数值完全相同。这说明 **H=20 的因子选择非常稳健**，
10 个相位中有 9 个选出完全一样的因子集，只是 R² 不同。

**这是一个比 R² 稳健性更可信的证据。** 因子选择对相位不敏感，
说明 LASSO 找到的是一个**真实的结构**，而不是样本特异的噪声。**这是项目里最有价值的一个发现。**

H=50 时则相反：offset 21 选出 14 个因子、offset 5 只选出 1 个，
`N_Selected` 从 1 到 14 剧烈跳动 —— **H=50 的因子选择非常不稳健**。

---

## 11. 实跑结果解读

### 11.1 Horizon vs OOS R²（主结果）

`forecast_horizon_LASSO_offset_robustness.csv`：

| H | Mean R² | Median | Std | Min | Max | N_Offsets | Train_Rows |
|---|---|---|---|---|---|---|---|
| 5 | 0.00053 | 0.00050 | 0.00010 | 0.00043 | 0.00069 | 5 | 119196 |
| 10 | 0.00133 | 0.00132 | 0.00023 | 0.00108 | 0.00175 | 10 | 118946 |
| 20 | 0.00297 | 0.00293 | 0.00079 | 0.00187 | 0.00446 | 10 | 118446 |
| 30 | 0.00520 | 0.00548 | 0.00115 | 0.00306 | 0.00652 | 10 | 117946 |
| 40 | 0.00740 | 0.00658 | 0.00180 | 0.00512 | 0.01022 | 10 | 117446 |
| 50 | 0.01115 | 0.01144 | 0.00171 | 0.00888 | 0.01324 | 10 | 116946 |
| 60 | 0.01503 | 0.01384 | 0.00474 | 0.00950 | 0.02509 | 10 | 116446 |
| 70 | 0.01895 | 0.01934 | 0.00491 | 0.01223 | 0.02614 | 10 | 115946 |
| 80 | 0.02322 | 0.02207 | 0.00441 | 0.01812 | 0.03004 | 10 | 115446 |
| 90 | 0.02618 | 0.02594 | 0.00784 | 0.01254 | 0.03736 | 10 | 114946 |
| 100 | 0.03158 | 0.03244 | 0.00994 | 0.01694 | 0.04392 | 10 | 114446 |
| 110 | 0.03572 | 0.03638 | 0.01414 | 0.01255 | 0.05720 | 10 | 113946 |
| 120 | 0.03931 | 0.03963 | 0.00976 | 0.02461 | 0.05426 | 10 | 113446 |
| 130 | 0.03956 | 0.04118 | 0.01140 | 0.02204 | 0.05687 | 10 | 112946 |
| 140 | 0.04681 | 0.04499 | 0.00993 | 0.03345 | 0.05988 | 10 | 112446 |
| 150 | 0.05087 | 0.04611 | 0.01005 | 0.03865 | 0.06574 | 10 | 111946 |

（`offset_results_H200.csv` 存在，但汇总表里没有 H=200 —— 说明那次运行在 H=200 之前就中断了。）

### 11.2 为什么 H 越大 R² 越高？

R² 从 H=5 的 0.05% 单调上升到 H=150 的 5.1%，**涨了 96 倍**。这个现象必须小心解读，因为它是**三个效应叠加**的结果：

**(1) 信噪比确实随 H 提高（真实效应）。**
短周期收益被微观结构噪声主导（买卖价差、日内噪声），因子信号被淹没。
拉长到季度尺度，基本面的影响显现，信噪比改善。
这是**真实的、经济学上合理的**效应。

**(2) 非重叠采样的样本量暴跌（统计效应）。**
H=20 时验证有 3350 行 / 67 个日期；H=200 时只有 300 行 / **6 个日期**。
**R² 是在 6 个独立观测上算出来的。** 这个数字的置信区间极宽，
几乎无法与"随机"区分。

**(3) 重叠 + 少量样本 → R² 的不确定性被放大。**
看 `Std_R2` 的走势：H=5 是 0.0001，H=150 是 0.0100 —— **涨了 100 倍**，
和 Mean 同比例增长。所以**相对变异（std/mean）大致恒定在 20%～25%**，
说明 R² 的"相对精度"并没有随 H 改善。

**结论：不要把这些 R² 的绝对值当作可信的业绩指标。**
应该报告的是：
1. R² **随着 H 增长这个趋势本身**（这是稳健的、有理论支持的）
2. 每个 H 下的 **std / N_Offsets**（说明不确定性有多大）
3. **N_Selected 的稳定性**（比 R² 更可信的稳健性证据）

### 11.3 LASSO 选出来的因子

`results_jump_150_LASSO_alpha_tunning.csv`（offset=0 的那条路径）：

| H | Best Alpha | N_Selected | 选中的因子 | R² |
|---|---|---|---|---|
| 5 | 3e-4 | 5 | `Momentum100_5`, `Momentum60_5`, `Volatility100`, `LogVolumeRatio10`, `Return10` | 0.00043 |
| 10 | 1e-3 | 2 | `Momentum60_5`, `Volatility100` | 0.00128 |
| 20 | 3e-3 | 1 | `Volatility100` | 0.00187 |
| 30 | 3e-3 | 1 | `Volatility100` | 0.00560 |
| 40 | 3e-3 | 2 | `Momentum60_5`, `Volatility100` | 0.00644 |
| 50 | 3e-3 | 2 | `Momentum60_5`, `Volatility100` | 0.00902 |
| 60 | 5e-3 | 2 | `Momentum60_5`, `Volatility100` | 0.01232 |
| 70 | 5e-3 | 2 | `Momentum60_5`, `Volatility100` | 0.01247 |
| 80 | 3e-3 | 4 | `Momentum100_5`, `Momentum60_5`, `Volatility100`, `Return20` | 0.01812 |
| 90 | 3e-3 | 4 | `Momentum100_5`, `Momentum60_5`, `Volatility100`, `Return20` | 0.02447 |
| 100 | 5e-3 | 2 | `Momentum100_5`, `Volatility100` | 0.01873 |
| 110 | 1e-2 | 1 | `Volatility100` | 0.01255 |
| 120 | 1e-2 | 1 | `Volatility100` | 0.02628 |
| 130 | 5e-3 | 3 | `Momentum100_5`, `Momentum60_5`, `Volatility100` | 0.02204 |
| 140 | 3e-3 | 4 | `Momentum100_5`, `Momentum60_5`, `Volatility100`, `Volatility60` | 0.03683 |
| 150 | 5e-3 | 4 | `Momentum100_5`, `Momentum60_5`, `Volatility100`, `Volatility60` | 0.04548 |

**三条非常清晰、可复现的规律：**

**(1) `Volatility100` 出现在 16/16 个 H 中，是最核心的因子。**
100 日波动率，而不是 10/20/60 日 —— **说明在本股票池里，长期波动率比短期波动率更有预测力**。
这与 `results.csv` 里 `Volatility20` 占优的结论**表面矛盾**，实际上不矛盾：
LR 消融用的是小因子集（只有 20 日波动率），LASSO 用的是完整的 4 个波动率尺度，
在 LASSO 的"竞争"下 100 日尺度胜出。**这说明"提供多个时间尺度让模型自己选"是有价值的。**

**(2) 动量只在 60 日和 100 日两个尺度上被选中，10/20 日从未被选中。**
**这是一个干净的经济学结论**：本股票池的动量效应需要至少 60 天的形成期，
短期（10/20 天）价格变动里没有可用的延续性信息（反而被反转效应主导）。
这条结论对"该用什么窗口构因子"有直接的指导意义。

**(3) `N_Selected` 大多在 1～4 之间，非常稀疏。**
15 个因子里通常只有 2～4 个存活。这既是 LASSO 的正常行为，
也**间接说明大量因子是冗余的**（4 个动量尺度之间、4 个量能尺度之间高度相关，
LASSO 只留一个代表）。

**(4) ⚠️ 一个必须警惕的点：α 太大 + 稀疏 = 可能选错了。**
H=110、120 时 `Best_Alpha = 1e-2`（最大惩罚），只留下 1 个因子 `Volatility100`，
R² = 0.0126 / 0.0263，**明显低于邻近的 H=100（0.0187）和 H=130（0.0220）**。
这是"α 选得过大导致欠拟合"的典型迹象。
更合理的做法是**限制 α 的搜索范围上限**，或者用 `LassoCV` 让算法自己定路径。

### 11.4 验证集的 R² 与多空组合的关系（重要但缺失的一环）

**这是当前项目最大的空白：**

`main.py` 的 `medium-blog` 版本里，回测部分**被整体注释掉了**：

```python
########## Backtesting:
#backtest_engine = BacktestEngine(forecast_horizon, validation_data, factor_engine)
#rebalance_dates, portfolio_data = backtest_engine.construct_portfolio()
#backtest_engine.backtest_rank_portfolio()
```

**所以 `medium-blog` 分支只回答了"因子有没有预测力"（R²），
完全没有回答"这个预测力能不能转成钱"（Sharpe）。**

而这恰恰是项目 README 里自己提出的核心问题：
> "whether modest predictive signals can become economically useful when combined with appropriate portfolio construction"

**R² = 0.003（H=20）到底意味着什么？**
- 从统计上说：模型的预测比"猜 0"好 0.3%，**非常微弱**
- 从经济上说：**完全未知**。一个 R²=0.003 的信号，如果集中在少数股票上、
  配合好的风险模型，可能产生 Sharpe > 1 的组合；
  也可能因为交易成本而完全不划算。

**这正是 `portfolio-optimization` 分支 `backtest_engine.py` 存在的意义。**

### 11.5 `portfolio-optimization` 分支的回测逻辑

`backtest_engine.py` 的两个方法（详见 [7.2](#72-方法-a2080-排名多空组合)、[7.3](#73-方法-b均值方差优化组合)）：

**方法 A（20/80 排名）** 打印 `Sharpe ratio for 20-80 portfolio equals: ...`
**方法 B（均值方差）** 打印 `Optimized Sharpe Ratio: ...`

`results1.txt` 最后一行 `0.005576000021140892` 和 `results2.txt` 最后一行
`0.00562792756637942` —— 这两个数字是 **OOS R²**（`validate()` 的返回值，
R²=0.006 的 4 因子模型和 3 因子模型），**不是 Sharpe**。

**`results/` 目录里那张 `Cumulative Return` 截图是仓库里唯一的组合业绩证据。**
要给出 Sharpe 的具体数字，需要重新跑一遍回测（见 [12.3](#123-跑回测portfolio-optimization-分支)）。

---

## 12. 操作手册：如何配置与运行

### 12.0 环境准备

```bash
pip install pandas numpy yfinance statsmodels scikit-learn cvxpy matplotlib
```

`requirements.txt` 只在 `portfolio-optimization` 分支有（`medium-blog` 分支没有这个文件）。

### 12.1 跑因子研究（`medium-blog` 分支）

```bash
git checkout medium-blog
python main.py
```

**会跑很久**（17 个 H × 10 offset × 8 alpha = 1360 次 LASSO），产出：

| 输出文件 | 内容 |
|---|---|
| `offset_results_H{5..200}.csv` | 每个 H 下 10 个 offset 的最优 alpha、R²、选中因子 |
| `forecast_horizon_LASSO_offset_robustness.csv` | 17 个 H 的 R² 汇总（mean/median/std/min/max） |

**`main.py` 里要调的开关（第 19～71 行）：**

```python
learning_model = "LASSO"          # 或 "LR"（LR 会触发 2^N 穷举消融，很慢）

start_date       = "2010-01-01"
validation_date  = "2020-01-01"   # 训练期 < 2020-01-01（再往前推 H 天做 purge）
test_date        = "2026-01-01"   # 验证期 = [2020-01-01, 2026-01-01)

forecast_horizon_values = [5, 10, ..., 200]
alpha_values            = [1e-5, 3e-5, ..., 1e-2]
n_offsets               = 10
```

**快速试跑建议：** 先把 `forecast_horizon_values` 改成 `[20, 60, 150]`、
`alpha_values` 改成 `[1e-3, 3e-3]`，把耗时从几小时压到几分钟，验证管线通了再全量跑。

### 12.2 加/减因子

在 `main.py` 的「FACTOR LIBRARY」段落里增删调用即可，**不需改其他任何文件**：

```python
factor_engine.add_momentum(100, 5)         # Momentum100_5
factor_engine.add_momentum(60, 5)          # Momentum60_5
factor_engine.add_volatility(100)          # Volatility100
factor_engine.add_volume(60)               # LogVolumeRatio60
factor_engine.add_shortterm_return(20)     # Return20
```

顺序无所谓（`features` 的顺序就是调用顺序，下游自动对齐）。

### 12.3 跑回测（`portfolio-optimization` 分支）

```bash
git checkout portfolio-optimization
python main.py
```

`main.py` 里的关键参数：

```python
start_date      = "2015-01-01"
validation_date = "2022-01-01"    # 训练期
test_date       = "2024-01-01"    # 验证期
forecast_horizon = 20

factor_engine.add_momentum(60, 5)
factor_engine.add_volatility(20)
factor_engine.add_volume(20)
factor_engine.add_shortterm_return(5)

learning_engine = LearningEngine("LR", train_data, validation_data, test_data,
                                 standardized_features, factor_engine)
model = learning_engine.train()
validation_data, oos_r2 = learning_engine.validate()

backtest_engine = BacktestEngine(forecast_horizon, validation_data, factor_engine)
rebalance_dates, portfolio_data = backtest_engine.construct_portfolio()
backtest_engine.backtest_rank_portfolio()          # 20/80 排名组合

lookback       = 60      # 协方差估计窗口（交易日）
risk_aversion  = 5       # λ
backtest_engine.backtest_optimized_portfolio(lookback, risk_aversion, data)
```

**`lookback` 和 `risk_aversion` 的调参直觉：**

| 参数 | 调大的效果 | 建议 |
|---|---|---|
| `lookback` | Σ 估计更稳定但更迟钝（含更久远的信息） | 60～252；股票数越多要越大（估计 49×49 的 Σ 至少需要几百天） |
| `risk_aversion` λ | 更保守，权重更分散，更贴近最小方差组合 | 1～10；**H 变了要重调**，因为 Σ 用了 `H × Σ_daily` 放大 |

**注意 `risk_aversion` 和 Σ 的乘数效应**：Σ 被乘了 H，所以 λ 的实际影响力也被放大 H 倍。
H=20 的 λ=5 和 H=200 的 λ=5 是**完全不同的风险偏好**。这是个容易踩的坑。

### 12.4 改股票池

编辑 `tickers.py`，或直接改行数更多的 `medium-blog` 版本（该分支的 `tickers.py` 有 49 只）。

**改股票池时要注意：**
- 截面 z-score 需要**每天至少有几十只股票**才有意义。少于 10 只时 `std` 不稳定。
- 换行业结构会改变因子有效性（比如加入小盘股后，波动率因子的符号很可能反转）。
- `tickers.py` 是**静态列表**，这引入了 [14.1](#141-幸存者偏差) 的偏差。

---

## 13. 如何扩展：新增因子的标准流程

假设要加一个**换手率因子** `Turnover20 = 20日均成交量 / 流通股本`。

**标准流程是 4 步：**

### Step 1：在 `factor_engine.py` 加方法

```python
def add_turnover(self, horizon):
    self.data[f"AvgVolume{horizon}"] = (
        self.data.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(horizon).mean())
    )
    self.data[f"Turnover{horizon}"] = self.data[f"AvgVolume{horizon}"] / self.data["SharesOutstanding"]
    self.data = self.data.replace([np.inf, -np.inf], np.nan)
    self.features.append(f"Turnover{horizon}")
```

**三个必须遵守的约定：**
1. **按 `Ticker` 分组**做时序运算（防串行）
2. **用 `transform`** 而不是 `rolling()`/`apply()`，保证索引对齐
3. **`replace([inf,-inf], nan)`** 清洗除零
4. **`self.features.append(名字)`** —— 漏了这行，因子就会静默失效（下游完全看不到它）

### Step 2：在 `data_engine.py` 准备所需字段

如果因子需要新字段（如 `SharesOutstanding`）：

```python
shares = yf.download(self.tickers, ...)   # 或 raw_data.xs("Volume", ...)
# 合并进 self.data
```

注意 `build_model_data()` 的 `[["Date","Ticker"] + self.features + [target]]`
只保留登记过的列，所以新增的中间列**不会被带进 model_data**，不会污染下游。

### Step 3：在 `main.py` 调用

```python
factor_engine.add_turnover(20)
```

### Step 4：验证因子真的进来了

**每次加完因子，第一件事是确认它进了 `model_data`：**

```python
model_data = factor_engine.build_model_data()
print("Factors:", factor_engine.features)
print("Shape:", model_data.shape)
print(model_data[["Date","Ticker"] + factor_engine.features].describe())
print("NaN rate:", model_data[factor_engine.features].isna().mean())
```

**要检查的三件事：**
1. `features` 列表里有新因子名
2. `describe()` 里数值量纲合理（不是全 0、不是 1e15）
3. **NaN 率不要异常高** —— 如果新因子的 NaN 率是 30%，`dropna()` 会砍掉 30% 的样本，
   训练集从 11.8 万掉到 8 万，这会让和之前的结果**不可比**

### Step 5：看它是否被 LASSO 选中

跑完 LASSO 后，检查 `offset_results_H*.csv` 的 `Selected_Factors` 列里有没有它。
**如果一个新因子在任何 H、任何 offset 下都没被选中，它就没有增量信息**
（可能是因为它和已有因子高度相关，也可能是算错了）。

**进阶：想看一个因子的"独立贡献"，做留一消融：**

```python
# 从全因子集出发，每次去掉一个
for dropped in features:
    subset = [f for f in features if f != dropped]
    # 用 OLS 拟合，比较 R² 相对全因子集的下降幅度
```

只做 N 次拟合（15 次），比 `2^15` 次穷举可行得多，而且解释力更强。

---

## 13.5 配置驱动切换因子（FactorEngine 扩展）

> 本节对应 `medium-blog` 分支上对 `factor_engine.py` 与 `main.py` 的改造。

### 13.5.1 为什么要改

改造前，因子集是**硬编码在 `main.py` 里的 15 次手写调用**：

```python
factor_engine.add_momentum(100, 5)
factor_engine.add_momentum(60, 5)
# ... 共 15 次
```

想换因子组合，必须**改源码**；而且没有任何机制保证「我以为加进去的因子」真的加进去了 ——
如果哪个 `add_*` 忘了 `features.append()`，这个因子会**静默失效**，模型照跑，结果无声错误。

改造后：因子集变成一张**纯配置表**，加/减/换只改这张表。

### 13.5.2 配置表

`main.py` 顶部新增：

```python
FACTOR_SPECS = [
    # Momentum
    ("momentum",   [100, 5]),
    ("momentum",   [60, 5]),
    ("momentum",   [20, 5]),
    ("momentum",   [10, 5]),

    # Volatility
    ("volatility", [100]),
    ("volatility", [60]),
    ("volatility", [20]),
    ("volatility", [10]),

    # Volume
    ("volume",     [100]),
    ("volume",     [60]),
    ("volume",     [20]),
    ("volume",     [10]),

    # Short-Term Return
    ("return",     [20]),
    ("return",     [10]),
    ("return",     [5]),
]

N_FACTORS_EXPECTED = len(FACTOR_SPECS)
STRICT_FACTOR_CHECK = False
```

循环里简化为：

```python
factor_engine = FactorEngine(data_engine)
factor_engine.add_factors(FACTOR_SPECS)

# 配置与登记表必须一致，否则下游会静默漏掉因子
assert len(factor_engine.features) == N_FACTORS_EXPECTED, (...)

model_data = factor_engine.build_model_data()

# 跑模型前校验：确认「以为生效的因子」真的生效且数值健康
factor_engine.verify(model_data, strict=STRICT_FACTOR_CHECK)
```

**换因子组合的例子：**

```python
# 只要动量 + 波动率
FACTOR_SPECS = [
    ("momentum",   [60, 5]),
    ("volatility", [100]),
]

# 试试 30 日动量、2 日跳空（探索参数空间）
FACTOR_SPECS = [
    ("momentum",   [60, 2]),
    ("momentum",   [30, 2]),
    ("volatility", [60]),
]

# 留一消融：只去掉 Volatility20
FACTOR_SPECS = [s for s in FACTOR_SPECS
                if s != ("volatility", [20])]
```

### 13.5.3 因子类型表

`FactorEngine.FACTOR_TYPES` 把类型字符串映射到构造函数：

```python
FACTOR_TYPES = {
    "momentum":   ("add_momentum",        ("long_horizon", "short_horizon")),
    "volatility": ("add_volatility",       ("horizon",)),
    "volume":     ("add_volume",           ("horizon",)),
    "return":     ("add_shortterm_return", ("horizon",)),
}
```

| 类型字符串 | 参数 | 生成列名 |
|---|---|---|
| `"momentum"` | `[long_horizon, short_horizon]` | `Momentum{L}_{S}` |
| `"volatility"` | `[horizon]` | `Volatility{H}` |
| `"volume"` | `[horizon]` | `LogVolumeRatio{H}` |
| `"return"` | `[horizon]` | `Return{H}` |

**新增一个因子族，只需两步：**
1. 在 `factor_engine.py` 写 `add_xxx()` 方法（记得 `self.features.append(...)`）
2. 在 `FACTOR_TYPES` 里登记一行

之后就能在配置表里用 `("xxx", [params])`，**`main.py` 不用动**。

### 13.5.4 分发器会拦截的四类错误

`add_factor()` / `add_factors()` 在添加时就报错，而不是等到模型跑出莫名其妙的结果：

| 错误 | 触发条件 | 报错信息 |
|---|---|---|
| 未知类型 | `add_factor("banana", [5])` | 列出可用类型 |
| 参数个数不对 | `add_factor("momentum", [60])` | 列出期望的参数名与个数 |
| 参数非正整数 | `add_factor("volatility", [0])` | 指出哪个参数非法 |
| **重复登记** | 同一因子加两次 | 指出重复的因子名 |

**「重复登记」这条最重要**：同一个因子加两次会让 LASSO 的输入出现**完全共线的两列**（相关系数 = 1），
系数在两者之间任意分配，结果不可解读。这是手工维护因子集时极易犯、且很难察觉的错误。

分发器还做了一次**闭环校验**：调用 `add_*` 之后回头确认列名真的进了 `self.features`，
防止漏写 `features.append()` 导致因子静默失效。

### 13.5.5 `verify()` 因子集体检

在 `build_model_data()` 之后调用，逐因子打印 NaN 率和健康状态，并返回一份报告：

```python
report = factor_engine.verify(model_data, strict=False, verbose=True)
```

输出形如：

```
[FactorEngine] Registered 15 factor(s):
    Momentum100_5            NaN=24.75%
    Volatility100            NaN=24.75%
    LogVolumeRatio10         NaN= 2.25%
    ...
[FactorEngine] model_data rows: 118446
```

| 报告字段 | 含义 | 级别 |
|---|---|---|
| `missing` | 登记的因子在 `model_data` 里找不到该列 | **致命** |
| `constant` | 整个样本上取值恒定（`nunique <= 1`），**没有截面区分度** | **致命** |
| `infinite` | 仍含 `inf`（会污染当天整个截面的 z-score） | **致命** |
| `all_nan` | 全部为 NaN | **致命** |
| `empty` | 全部为 0 或 NaN | **致命** |
| `high_nan` | NaN 比例 > 20%（暖机期过长，会通过 `dropna()` 砍掉大量样本） | 警告 |
| `nan_rates` | 每个因子的 NaN 比例 | 信息 |

**致命问题一律抛异常**（`ValueError`），不会让你带着坏因子往下跑。
`high_nan` 默认只警告；把 `STRICT_FACTOR_CHECK = True` 可以让它升级为异常。

`high_nan` 这个检查针对一个很实际的坑：**加一个长窗口因子（比如 200 日波动率）会让前 200 天的数据全被 `dropna()` 丢掉**，
训练集从 11.8 万行掉到 8 万行，于是**和之前的结果不再可比**。
warning 会明确提示这一点。

### 13.5.6 实测验证

改造后离线跑了 15 项测试（合成数据，不联网），全部通过。
测试文件随代码一起留在仓库里，可随时回归：

```bash
python test_factor_engine.py     # 15/15：分发器 + verify() 的行为
python test_factor_specs.py      # 配置表与分发器一致性
```

| 测试 | 结果 |
|---|---|
| 正常配置 15 因子全部登记 | OK |
| 因子名与类型表定义一致 | OK |
| 未知类型 kind 报错 | OK |
| 参数个数不匹配报错 | OK |
| 参数非正整数报错（0 / 负数 / 浮点） | OK |
| 重复登记同一因子报错 | OK |
| 空因子集 verify 报错 | OK |
| 因子列缺失被识别并报错 | OK |
| 常量因子被识别 | OK |
| inf 因子被识别 | OK |
| 全 NaN 因子被识别 | OK |
| 高 NaN 率被识别为警告 | OK |
| `strict=True` 让警告升级为异常 | OK |
| 正常因子集的 verify 报告干净 | OK |
| `add_factors` 返回 self 可链式调用 | OK |

另外用 AST 从 `main.py` 里取出**真实的 `FACTOR_SPECS`** 跑了一遍分发器，
确认 15 个因子名与改造前完全一致：

```
Momentum100_5, Momentum60_5, Momentum20_5, Momentum10_5,
Volatility100, Volatility60, Volatility20, Volatility10,
LogVolumeRatio100, LogVolumeRatio60, LogVolumeRatio20, LogVolumeRatio10,
Return20, Return10, Return5
```

**与 `offset_results_H*.csv` 里 `Selected_Factors` 记录的名字逐字一致**，
所以改造**不改变任何因子定义**，之前所有实验结果仍然可比。

### 13.5.7 一个仍然存在的限制

**新增因子族仍然需要写代码 + 登记一行。** 配置驱动只解决了「已有因子的组合切换」，
没有解决「新增因子不改引擎」。要做到后者需要**装饰器注册表 + 自动发现**：

```python
@register_factor("turnover", params=("horizon",))
def add_turnover(self, horizon):
    ...
```

这是另一个量级的改造。就当前研究阶段（4 个因子族、15 个因子实例）而言，
`FACTOR_TYPES` 显式登记表的**可读性更好**，也更符合「配置即文档」的思路。

---

## 14. 工程陷阱与已知问题

### 14.1 幸存者偏差

`tickers.py` 是**手写的静态列表**（AAPL、MSFT、NVDA...），
这些都是**今天仍然存在且是大盘股**的公司。

**后果：**
- 退市的公司（Lehman、GE 的部分业务、无数被并购/破产的小盘股）**完全不在样本里**
- 回测会系统性高估收益（作者在 `2453068` 的 commit message 里已经指出了）
- **OLS 的截距 β₀ = 0.0043 显著为正，就是这个偏差最直接的证据**：
  一只"所有因子都平均"的股票，预期每月跑赢 SPY 0.43%，这在真实市场里不成立

**更隐蔽的后果：** 波动率因子的符号可能被这个偏差翻转。
在本股票池里 `Volatility100` 系数**为正**（高波动 → 高收益），
这与经典低波动异象相反。一个合理的解释是：
**能活到今天的高波动股票，恰恰是活下来的赢家**（NVDA、AMD、META 都是高波动大赢家），
而**同样的高波动在包含退市股票的全样本里会亏钱**。

**修正方向：** 使用 point-in-time 的历史成分股数据（CRSP、Compustat），
或者至少用 ETF 成分股的历史快照。

### 14.2 历史样本期与时间穿越

`medium-blog` 的 `start_date = "2010-01-01"`、`end_date = "2026-01-01"`。

**⚠️ 注意 `2026-01-01` 是未来日期**（相对当前时间）。
`yfinance` 对超出实际数据的日期只会返回已有部分，所以**不会出错**，
但**必须确认**你拿到的数据实际截止到哪一天。作者在 `ideas.txt` 里写的
"expand the previous data set" 说明他清楚这个问题。

### 14.3 无穷值的清洗位置

`factor_engine.py` 里 `replace([inf,-inf], nan)` 只在 `add_volume` 里做了一次：

```python
def add_volume(self, horizon):
    ...
    self.data = self.data.replace([np.inf, -np.inf], np.nan)   # 只有这里有
```

而 `add_momentum`、`add_volatility`、`add_shortterm_return` **都没有**。

**理论上这不会出问题**，因为：
- `add_momentum`：`Close.shift(short) / Close.shift(long)` —— 只要价格不为 0 就不会出 `inf`
- `add_volatility`：`rolling().std()` —— 不会产生 `inf`
- `add_shortterm_return`：`Close / Close.shift(h)` —— 同理

但在**极端情况**下（停牌期间价格为 0、数据源异常、拆股未调整）就可能产生 `inf`。
而 `inf` 一旦进入 `normalize()`，会污染**当天整个截面**的均值和标准差
（`mean` 变成 `inf`，`std` 变成 `nan`，那一整天所有股票标准化全废）。

**稳健的做法：把清洗放在 `build_model_data()` 里统一做一次。**

```python
def build_model_data(self):
    model_data = self.data[["Date","Ticker"] + self.features + [self.target]].copy()
    model_data = model_data.replace([np.inf, -np.inf], np.nan)   # 统一护栏
    model_data = model_data.dropna()
    return model_data
```

### 14.4 循环内 pivot（性能）

`backtest_engine.py` 的 `backtest_optimized_portfolio()` 里：

```python
for current_date in self.rebalance_dates:
    ...
    returns_wide = data.pivot(index="Date", columns="Ticker", values="DailyReturn")   # 每次迭代都重算！
    returns_wide = returns_wide.dropna()
```

**这个 pivot 的结果在循环内是不变的，应该提到循环外。**

改法：

```python
def backtest_optimized_portfolio(self, lookback, risk_aversion, data):
    returns_wide = data.pivot(index="Date", columns="Ticker", values="DailyReturn").dropna()
    portfolio_results = []
    for current_date in self.rebalance_dates:
        ...
        past_returns = returns_wide.loc[returns_wide.index < current_date, current_tickers].tail(lookback)
```

当前数据量（49 只 × 2700 天）下能跑完，但这是**扩展股票池时的第一个性能瓶颈**。

### 14.5 每个 H 重复下载数据（性能）

`main.py` 的 horizon 循环里，`DataEngine.download_data()` 在每个 H 都被调用一次。
**数据本身完全相同**（只有 `FutureReturnH` 和标签列随 H 变），
`yfinance` 的下载被重复了 17 次。

**改法：** 把 close/volume 的下载提到循环外，循环内只重算标签列。
这能把总运行时间砍掉相当一部分（网络请求是主要开销之一）。

### 14.6 LR 与 LASSO 的评估口径不一致

见 [6.4](#64-一个真实的-bug)。LR 的 `validate()` 不做非重叠采样，
LASSO 做。所以 `results.csv`（LR）和 `forecast_horizon_LASSO_*.csv`（LASSO）的 R²
**不可直接比较**。

**修正：** 让 LR 的 `validate()` 也接受 `jump`/`offset` 并做同样的采样。
`LearningEngine.__init__` 已经接收了 `jump` 和 `offset` 参数（LR 分支传的是 `jump, 0, 0`），
只是 LR 的 `validate()` 没用它们。

### 14.7 Ledoit-Wolf 的协方差与 `quad_form` 的交互

`backtest_engine.py` 里先算样本协方差 `past_returns.cov()`，**紧接着就被 Ledoit-Wolf 覆盖了**：

```python
sigma_daily = past_returns.cov().values      # 算了但没用（被下一行覆盖）
sigma = self.forecast_horizon * sigma_daily
lw = LedoitWolf(); lw.fit(past_returns)
sigma_daily = lw.covariance_                 # 实际使用的
sigma = self.forecast_horizon * sigma_daily
```

**这一点是好的**：说明 Ledoit-Wolf 是**有意选择**的，不是意外。
但两行死代码应该清理掉，否则读者会困惑到底用了哪个。

**另一个隐患：** Ledoit-Wolf 收缩保证 `Σ` 半正定，但如果 `past_returns` 有
**完全共线的列**（两只股票在窗口内收益完全一样，比如都是停牌），
`lw.covariance_` 可能仍不是严格的半正定，`cp.quad_form` 会报 `DCPError`。
**建议加 try/except 捕获并跳过当天的调仓**，而不是让整个回测崩掉。

### 14.8 训练阶段的样本重叠未处理

见 [5.3](#53-重叠样本问题overlapping-labels)。
`fit(cov_type="cluster", groups=Date)` 处理了同日截面相关，
但**没处理跨日的时间重叠**。H=20 时，t 和 t+1 的残差共享 19 天价格路径。

**后果：** t 值虚高，因子显著性被高估。`results1.txt` 里 `Volatility20_z` 的 z=8.65
看起来很显著，但考虑到重叠样本，真实显著性要打折。

**修正方向：**
- 训练集也做非重叠采样（`jump=H`），代价是有效样本减少到 1/H
- 用 Newey-West 时间序列稳健标准误（`cov_type="hac"`, `maxlags=H`）
- 用 block bootstrap 做显著性检验

### 14.9 `test` 集从未被使用

`split_data()` 精心切出了 `test_data`，`LearningEngine` 也接收了它，
但 **`main.py` 从头到尾没有用过 `test_data`**。

**这是项目里最"便宜"的改进**：test 集是完全留存的、从未被模型选择过程污染的数据。
在 H=20 上跑一次最终评估（用选定好的 α），就能得到一个**无偏的性能估计**。
README 的 Research Directions 里也把 "Final evaluation on the untouched test period" 列为待办。

### 14.10 误入库的垃圾文件

`portfolio-optimization` 分支里被 git 跟踪的文件包括：
- `__pycache__/*.pyc`（12 个）
- `.DS_Store`
- `data.csv`（26 MB）、`stock_data.csv`（10.8 MB）
- `results/` 下的两张 1 MB 截图

**建议加 `.gitignore`：**

```gitignore
__pycache__/
*.pyc
.DS_Store
data.csv
stock_data.csv
spy_data.csv
results/*.png
```

（注意：**已跟踪的大文件需要 `git rm --cached` 才能停止跟踪**，
但**历史里仍然存在**，要真正瘦身需要用 `git filter-repo` 重写历史。
考虑到这是研究项目，也可以先不处理历史，只防止继续变大。）

---

## 15. 后续研究路线

按「投入产出比」排序，这是我建议的优先级：

### 第一梯队：改动小、价值大

**1. 用 test 集做最终评估**（[14.9](#149-test-集从未被使用)）
代码已经准备好了，只需要在 `main.py` 里跑一次。**这是最便宜的一个改进，立刻能得到无偏结论。**

**2. 把回测接回 horizon 循环**（[11.4](#114-验证集的-r²-与多空组合的关系重要但缺失的一环)）
现在只算 R²，不算 Sharpe。把 `BacktestEngine` 接上，就能回答"预测力 → 钱"这个核心问题。
**这是项目的最终目标，不做这个前面所有工作都缺一个结尾。**

**3. 修掉 LR/LASSO 的评估口径不一致**（[14.6](#146-lr-与-lasso-的评估口径不一致)）
让 LR 的 `validate()` 也做非重叠采样，两条线的 R² 就能直接比。
`jump` 和 `offset` 参数已经传进来了，只需在 `validate()` 里用上。

**4. 统一 inf 清洗位置**（[14.3](#143-无穷值的清洗位置)）
把 `replace([inf,-inf], nan)` 移到 `build_model_data()` 里，一行代码，消除一个隐患。

**5. 循环内 pivot 和重复下载移出去**（[14.4](#144-循环内-pivot性能)、[14.5](#145-每个-h-重复下载数据性能)）
把 17 次下载变成 1 次，把 N 次 pivot 变成 1 次。

### 第二梯队：需要一些设计工作

**6. 加交易成本**（`ideas.txt` 第 9 条）
现在的 Sharpe 是**毛收益**。H=20 每月调仓、每次换仓约 40% 的仓位，
按单边 10bp 算，年化拖累约 `12 × 0.4 × 0.001 × 2 ≈ 0.96%`。
**这个量级足以吃掉全部 alpha**（H=20 的月均超额大概就 0.1%～0.3%）。
所以**加交易成本是全项目最重要的一次现实检验**。

**7. 换手率约束 / turnover-aware optimization**
在 cvxpy 问题里加 `cp.norm1(w - w_prev) <= turnover_limit`。
比事后算成本更优雅，能让优化器**自己**在收益和换手之间权衡。

**8. 滚动窗口系数稳定性**（`ideas.txt` 第 11 条）
用滚动窗口重复训练 LR，画出 β 随时间的变化。
**如果某个因子的系数符号在时间里翻转过，那它对未来就不可信。**
这是因子"是不是真实"的最直接检验，比 OOS R² 更直观。

**9. 市场状态分析**（`ideas.txt` 第 13 条）
按波动率水平 / 牛熊市分组，比较各组的 IC 和 Sharpe。
预期结果：动量在趋势市中有效，反转在震荡市中有效。
这类分析能让因子有**条件使用的场景**，而不是"永远用同一套权重"。

**10. 用 Newey-West 处理重叠样本**（[14.8](#148-训练阶段的样本重叠未处理)）
`fit(cov_type="hac", maxlags=forecast_horizon)`，检查显著性是否还成立。

### 第三梯队：架构级改进

**11. 引入 IC（信息系数）作为因子评价指标**
现在只有 R²。IC 是**因子值本身与未来收益的截面相关系数**，它：
- **不依赖模型**：能单独评价一个因子，而不受回归设定影响
- **有时序含义**：IC 的均值（预测力）、标准差（稳定性）、ICIR（=均值/标准差，核心指标）、
  胜率（IC>0 的比例）
- 是业界评价因子的**标准语言**

**对每个因子单独算 IC 时间序列，画 ICIR 排行，会比现在的 R² 结论丰富得多。**

**12. 用 LassoCV 替代手工 α 网格**（[10.3](#103-一个方法论上的隐患-best_alpha-是在验证集上选的)）
`LassoCV(cv=5)` 在训练集内部做交叉验证选 α，**彻底消除"在验证集上选 α"的选择性偏差**。
同时自动探索整条正则化路径。

**13. 修正幸存者偏差**（[14.1](#141-幸存者偏差)）
接入 point-in-time 成分股数据。**这是唯一能让结论"可信"的改进**，
但成本高（需要付费数据）。在拿到之前，**所有结论都应该明确标注"含幸存者偏差"**。

**14. 引入非线性模型**
`LearningEngine` 已经为多模型设计好了（`model_type` 分支）。
梯度提升树（LightGBM/XGBoost）在 Gu, Kelly & Xiu (2020) 的对比中
显著优于线性模型。**但要注意：树模型在小样本、低信噪比的金融数据上极易过拟合，
必须配合严格的时序交叉验证。**

**15. 因子正交化**
对高度相关的因子族（4 个动量尺度）做 Gram-Schmidt 正交化，
或者用 PCA 提取主成分。这样能**分离出每个因子族的独立贡献**，
而不是让 LASSO 在共线的因子里随便挑一个（挑哪个可能不稳定）。

---

## 附录 A：字段与命名速查

| 字段 | 产生位置 | 含义 |
|---|---|---|
| `Date`, `Ticker` | DataEngine | 面板键 |
| `Close`, `Volume` | DataEngine | 已调整的收盘价、成交量 |
| `DailyReturn` | DataEngine | `Close.pct_change()`，同日因子 |
| `FutureReturn{H}` | DataEngine | 个股未来 H 日收益（**标签原料**） |
| `SPYClose` | DataEngine | 基准收盘价 |
| `SPYReturn{H}` | DataEngine | 基准未来 H 日收益 |
| **`NetReturn{H}`** | DataEngine | **`FutureReturn{H} − SPYReturn{H}`，模型标签** |
| `Momentum{L}_{S}` | FactorEngine | `Close[t−S]/Close[t−L] − 1` |
| `Volatility{H}` | FactorEngine | `DailyReturn` 的 H 日滚动标准差 |
| `AvgVolume{H}` | FactorEngine | 中间产物，不登记 |
| `VolumeRatio{H}` | FactorEngine | 中间产物，不登记 |
| `LogVolumeRatio{H}` | FactorEngine | `log(Volume/AvgVolume)` |
| `Return{H}` | FactorEngine | `Close[t]/Close[t−H] − 1` |
| `{feature}_z` | PreprocessEngine | 截面 z-score，**模型实际输入** |
| `PredictedReturn` | LearningEngine | 模型预测的超额收益 |
| `Rank` | BacktestEngine | 预测值的截面百分位 |
| `Position` | BacktestEngine | 1 = 多头，−1 = 空头，0 = 不持有 |
| `Weights` | BacktestEngine | 均值方差优化的权重 |

## 附录 B：超参数速查

| 参数 | 位置 | `medium-blog` | `portfolio-optimization` | 作用 |
|---|---|---|---|---|
| `forecast_horizon` | DataEngine | 5～200（循环） | 20 | 预测期，决定标签和调仓周期 |
| `start_date` | main.py | 2010-01-01 | 2015-01-01 | 数据起点 |
| `validation_date` | main.py | 2020-01-01 | 2022-01-01 | 训练/验证分界 |
| `test_date` | main.py | 2026-01-01 | 2024-01-01 | 验证/测试分界 |
| `learning_model` | main.py | `"LASSO"` | `"LR"` | 模型类型 |
| `alpha` | main.py | 1e-5～1e-2（8 个） | — | LASSO 正则强度 |
| `n_offsets` | main.py | 10 | — | 相位个数（稳健性检验） |
| `jump` | main.py | `= forecast_horizon` | `= forecast_horizon` | 验证采样间隔 |
| `lookback` | main.py | — | 60 | 协方差估计窗口（交易日） |
| `risk_aversion` λ | main.py | — | 5 | 均值方差风险厌恶系数 |

## 附录 C：一句话总结每层的"为什么"

| 层 | 核心设计 | 为什么必须这样 |
|---|---|---|
| DataEngine | `NetReturn = r_i − r_SPY` | 做的是选股（截面），不是择时，所以目标是超额收益 |
| DataEngine | `groupby("Ticker").shift(-H)` | 不分组会让相邻股票的数据串行，标签全错 |
| FactorEngine | `Momentum{L}_{S}` 双 shift | 跳过后 S 天，把动量和短期反转两个反向信号分开 |
| FactorEngine | `LogVolumeRatio` 取 log | 成交量比值右偏严重，log 后对称、方差稳定 |
| FactorEngine | `self.features` 自动登记 | 单一真相来源，加因子不用改下游任何代码 |
| PreprocessEngine | `groupby("Date")` 做 z-score | 只需要"今天谁比谁强"，且自动剔除市场层面的时间变动 |
| PreprocessEngine | `train_cutoff = dates.iloc[-H]` | 标签用了未来 H 天价格，必须 purge 否则泄漏 |
| LearningEngine | `cov_type="cluster"` by Date | 同一天的股票残差相关，普通标准误会低估 |
| LearningEngine | `iloc[offset::jump]` 验证 | 让验证样本不重叠，R² 才不被重叠样本虚高 |
| LearningEngine | LASSO 替代 OLS | 15 个高相关因子，OLS 共线不可解读；穷举 2^15 不可行 |
| BacktestEngine | `cp.sum(w)==0` | 与"超额收益"目标一致，做美元中性组合 |
| BacktestEngine | Ledoit-Wolf 收缩 | 49 只股票要估 1225 个协方差但只有 60 天数据，样本 Σ 奇异 |
| BacktestEngine | 同时打印 predicted 与 realized | 两条累积曲线的背离 = 模型过度自信的度量 |
| main.py | `max_horizon` 统一 purge | 保证所有 H 用同一套切分，比较才公平 |
| main.py | `offset` 循环 | 若结论只在一个采样相位成立，那它是运气不是因子 |

---

## 参考文献

- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*. Journal of Finance.
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). *Value and Momentum Everywhere*. Journal of Finance.
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. JRSS-B.
- Ledoit, O. & Wolf, M. (2004). *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*. Journal of Multivariate Analysis.
- Campbell, J. & Thompson, S. (2008). *Predicting Excess Stock Returns Out of Sample*. Review of Financial Studies.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). *Optimal Versus Naive Diversification*. Review of Financial Studies.
- Gu, S., Kelly, B. & Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies.

---

*本教程基于 `medium-blog` 分支 commit `89e6f4a` 与 `portfolio-optimization` 分支 commit `55719da` 的源码与实跑结果整理。*

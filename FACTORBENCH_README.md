# AI_finance × FactorBench 对接说明

把 AI_finance 的因子送进 [FactorBench](https://github.com/ZhuoHan1998/FactorBench-Release)
的四级评估管线。评估统计**全部调用 FactorBench 原库**，所以口径与论文一致，
不重新发明指标。

---

## 1. 为什么值得接

AI_finance 原来只看回归 R² 和 LASSO 系数，这回答不了三个关键问题，而 FactorBench 正好各有一把尺：

| 问题 | AI_finance 原来 | FactorBench 给的尺子 |
|---|---|---|
| 因子的预测力是真的还是搜出来的？ | 只有 R² 均值 | IC / RankIC / ICIR **分段**（train/valid/test）+ Newey-West t |
| `Volatility100` 是 alpha 还是波动率风格换了个壳？ | 看不出来 | **风格中性化**：剥掉 beta/波动率/动量/反转/流动性，看残差 IC 还剩多少 |
| 15 个因子有几条独立信息？ | 靠 LASSO 系数猜 | **冗余矩阵**：因子间横截面 |Spearman| 直接量 |
| 扣掉成本还能赚吗？ | 回测段被注释掉了 | **组合回测**：选择→合成→含成本执行，多头与多空 |

第三条特别关键。你之前在 `medium-blog` 上发现 `Volatility100` 在 16/16 个周期都被选中、
且系数为正（与经典低波动异象相反），怀疑是幸存者偏差。风格中性化就是**直接检验这个怀疑**的工具。

---

## 2. 新增文件

| 文件 | 作用 |
|---|---|
| `factorbench_adapter.py` | 适配层：行情下载+缓存 → 规范 panel → 按 `FactorEngine` 公式重建 15 个因子 → IC 记录 |
| `run_factorbench_eval.py` | 运行器：四个阶段的 CLI |
| `verify_factorbench.py` | 自检：不联网验证因子名一致性、全链路、风格门槛 |
| `_make_synthetic_cache.py` | 造与 yfinance 同构的合成行情（限流时用于验证代码路径） |

**没有改动任何原有文件**（`main.py` / `factor_engine.py` 的改动是上一轮「配置驱动切换因子」的工作，与本次无关）。

---

## 3. 用法

### 3.1 全量跑（推荐）

```bash
python run_factorbench_eval.py --stage all
```

默认参数**对齐 `main.py`**：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--start` / `--end` | `2010-01-01` / `2026-10-01` | 与 `main.py` 的 `start_date` 一致 |
| `--train-end` | `2019-12-31` | |
| `--valid-start` / `--valid-end` | `2020-01-01` / `2023-12-31` | 对齐 `validation_date=2020-01-01` |
| `--test-start` | `2024-01-01` | 对齐 `portfolio-optimization` 的 `test_date` |
| `--horizon` | `20` | 预测周期 |
| `--target` | `net` | `net`=相对 SPY 超额（AI_finance 口径）；`raw`=论文原始收益 |
| `--rebalance-days` | `20` | 调仓间隔 |
| `--cost-bps` | `10.0` | 单边成本 |

> **为什么 `--test-start` 默认是 2024 而不是 `main.py` 里的 `test_date=2026-01-01`？**
> `main.py` 那个值等于当前时间、又没有结束日，`test` 段会是**空的**。
> 2024-01-01 才是它的设计意图（见 `portfolio-optimization` 分支），也让 test 段有约 22 个月真实数据。

### 3.2 单阶段

```bash
python run_factorbench_eval.py --stage ic          # 只出 IC 表
python run_factorbench_eval.py --stage style       # 只做风格中性化
python run_factorbench_eval.py --stage redundancy
python run_factorbench_eval.py --stage portfolio
```

### 3.3 自检（不联网）

```bash
python verify_factorbench.py          # 全部三项
python verify_factorbench.py --fast   # 跳过全链路，只查因子名与门槛
```

---

## 4. 产物

输出到 `factorbench_out/`（可用 `--out-dir` 改）：

| 文件 | 内容 |
|---|---|
| `ic_table.csv` | 每因子 × 每段的 IC / RankIC / ICIR / 可打分日数 / Newey-West t / 是否准入 |
| `style_neutralisation.csv` | 每因子 × 每段：原始 IC、残差 IC、ΔIC、风格 R²、退化率、ΔIC 的 HAC t |
| `redundancy_spearman.csv` / `redundancy_pairs.csv` | 因子间 \|Spearman\| 矩阵与配对明细 |
| `portfolio_performance.csv` | 多头/多空的净 CAGR、净 Sharpe、最大回撤、年换手 |
| `backtest_daily_*.csv` | 逐日净值/成本/换手明细 |

---

## 5. 口径说明（重要，决定数字怎么读）

### 5.1 目标变量是你原来的口径

`--target net` 下，目标是 `FutureReturn{H} − SPYReturn{H}`，与 `DataEngine` 完全一致。
代价：它**不是市场中性组合的 P&L**——对 beta≈1 的大盘股，超额收益≈0，
会让 IC 整体偏小。论文用的是原始收益（`--target raw`）。
**两种都跑一遍对比**，差异本身就是信息。

### 5.2 因子方向在 train 段锁定

方向的确定只允许用 train 段（`train_sign`），之后 valid/test 固定不变。
这样**验证期/测试期的负 IC 是真实的符号反转**，不会被「重新优化方向」藏起来。
这是 FactorBench 的核心纪律之一。

### 5.3 IC 的 t 值用 Newey-West（25 lags）

目标是 20 日前瞻收益、按日采样，相邻观测共享 19/20 的时间跨度。
按 iid 算 t 值会高估约 4 倍，所以必须用 HAC。

### 5.4 风格面板的 5 个暴露

| 风格 | 定义 |
|---|---|
| `beta` | 252 日 `cov(r_i, r_mkt) / var(r_mkt)`，等权市场 |
| `volatility` | 60 日收益标准差 |
| `momentum` | `close[t-21]/close[t-252] − 1`（跳过最近一个月） |
| `reversal` | 过去 21 日收益取负 |
| `liquidity` | `log(20 日均 dollar volume)`，即 `close × volume` |

**`VolatilityN` 与 `volatility` 风格高度同源**——所以这个因子的风格 R² 会很高、
残差 IC 会大幅缩水。这正是要看到的证据，不是 bug。

### 5.5 横截面回归门槛 = 30 只

FactorBench 要求每个交易日**至少 30 只股票**同时具备 5 个风格才做回归。
AI_finance 的 51 只票满足；票池更小会被守卫拦住并明确报错（而不是静默返回 NaN）。

### 5.6 目标周期与风格的耦合（值得注意）

`--horizon 20` 时，风格里的 `reversal`（过去 21 日）和 `momentum`（跳过最近 21 日）
与目标窗口**几乎完全重叠**，风格回归会吃掉很大一块信号。
换 `--horizon 5` 再看一遍，能判断到底是风格重叠还是真没信息。

---

## 6. 已知限制

1. **`open` 列是 `close` 的副本。** AI_finance/DataEngine 只取 Close+Volume，
   yfinance 虽然能拿到 OHLC，但为了与你的管线逐位对齐，适配层只用 Close/Volume/High/Low。
   `vwap` 按论文定义 `(high+low+close)/3` 计算，是**代理**不是真实成交均价。
   → 15 个因子本身不受影响（它们只用 close 和 volume）。

2. **静态 universe，含幸存者偏差。** 与论文同样的问题，也与 AI_finance 一致。
   tickers 列表是**今天**的名单回填历史。论文附录明确承认这点。
   你在 `test/AI_finance` 那边做的 PIT 池 + 退市股修正，比两边都更严谨。

3. **风格中性化是紧贴 FactorBench 口径的复刻实现。**
   `_residualise` / `_xs_corr` / `_ranks` 三个向量化函数按 `style.py` 的数值定义复刻
   （因为上游是私有名，直接 import 会随其改动而断）。常量全部取自上游公开名
   （`MIN_OBS_PER_DATE` / `WINSOR` / `DEGENERATE_REL` / `FLAT_REL` / `HAC_LAGS`）。

4. **`--target net` 下 IC 偏小**，见 §5.1。

5. **数据缓存是必须的，不是优化。** Yahoo 会限流（实测 429），
   缓存才让「两次评估数字一致」成为可能。缓存在 `_fb_cache*/`，已 gitignore。

---

## 7. 统计推断与成本敏感性

这一节是本项目**最容易出错、也最影响结论**的部分。

### 7.1 为什么日频重叠样本必须用 HAC

目标是 `NetReturn{H}`，即 H 日前瞻收益**按日采样**。相邻两天共享 `(H−1)/H`
的时间跨度（H=20 时是 19/20），残差在时间上高度自相关——实测 lag-1 自相关约 0.95。

按 iid 或「只按日期聚类」处理，会把标准误**低估 1.8–3.6 倍**：

| 因子 | 按日期聚类 t | Driscoll-Kraay t | 标准误放大 |
|---|---|---|---|
| `Volatility20` | 11.74 | **3.30** | 3.55x |
| `Momentum60_5` | 3.15 | **0.94** | 3.37x |
| `Return20` | −1.02 | **−0.35** | 2.92x |
| `LogVolumeRatio20` | −1.26 | **−0.71** | 1.79x |

`Momentum60_5` 在旧口径下 **t=3.15 看起来显著**，正确处理时间相关后只剩 **0.94**。
这就是为什么这个修正比找新因子更重要。

**做了什么：**

- 新增 `hac_inference.py`：Driscoll-Kraay（按时间聚合得分 + Newey-West）、
  两向聚类、单向聚类，以及 `fit_ols` 统一入口。
- `learning_engine.py` 默认改用 `cov_type="driscoll_kraay"`。
  旧行为保留为 `cov_type="cluster_date"`，**仅供对照**。
- 滞后阶数按已知重叠长度取 `horizon + 5`，**不用**数据驱动的带宽公式
  （常规公式在两年窗口上只给约 5，远小于真实的 20 日重叠）。

### 7.2 经验贝叶斯收缩（Chen-Welch 2026）

$$r_{adj} = \left[1 - \frac{1}{\mathrm{Var}(t)}\right] \cdot r$$

t 值 = 真实信号 + 抽样噪声，噪声方差按构造为 1。所以 t 值的**横截面**方差里，
只有超过 1 的部分是真信号；`Var(t) ≤ 1` 就说明这批因子的离散度**不比纯运气大**。

IC 表阶段会自动输出 `ic_eb_shrunk.csv`，并打印 `Var(t)` 与收缩后的 IC。

> ⚠️ 因子数少时 `Var(t)` 本身噪声极大（Chen-Welch 用约 170 个异象估这个量）。
> 15 个因子算出来的收缩因子只应看量级，**不能**当作「某因子显著」的依据。

独立工具：`python eb_shrinkage.py [csv] [horizon] [因子名...]`

### 7.3 成本敏感性

成本对净收益是**线性**的（`net = gross − turnover × cost_bps/1e4`），且只作用在调仓日。
所以跑一次零成本回测就能解析出任意成本档下的净指标，不必重跑整个流程。

`stage_portfolio` 会输出 `cost_sensitivity.csv`，并打印每档成本下的净 CAGR、
净 Sharpe、年成本拖累，以及**断本点**（净收益归零时的单边成本）。

默认网格：`0 / 5 / 10 / 15 / 20 / 25 / 30 / 50` bp。
Chen-Velikov (2023) 估计异象组合的合理往返成本约 **20 bp/月**，
对应到双边单边口径可比 `--cost-bps 20~25` 更贴近现实——默认的 10 bp 偏乐观。

> 本项目的合成数据跑出来断本点很高（long_only 约 365 bp），
> 但那是合成行情的事，真实数据必然低得多。**这张表的价值在于真实数据下的断本点。**

### 7.4 异质成本：按流动性定价（`--cost-model liquidity`）

统一 bps 有一个隐含假设：**所有股票的交易成本相同**。真实市场里高波动、低成交量的票
明显更贵，而这正好是某些因子偏好的方向。

启用后成本改为逐股逐日：

    one_way_bps_i = (ATR_i / close_i) × vol_factor × 1e4
                  + (order_shares_i / avg_volume_i) × size_factor × 1e4

`ATR` 用 Wilder 递归平均（与主流 TA 库一致），`avg_volume` 用滚动均值（ADV 标准定义）。

**合成数据上的结果（同一份信号、同一套选股）：**

| 策略 | 统一 10bp | 异质成本 | 等效单边 | 年成本 |
|---|---|---|---|---|
| long_only | 22.69% | **19.38%** | 13.4 bp | 3.36% |
| long_short | 2.01% | **−0.65%** | 13.0 bp | 3.26% |

多空组合从「勉强正收益」翻成亏损：年成本从 0.63% 涨到 3.3%（5 倍），
而多空的毛收益只有 2.65%。**统一成本假设正好卡在是/否的判决线上。**

#### 因子成本倾斜（`factor_cost_tilt.csv`）

逐日横截面 `Spearman(信号, ATR/价格)` 的均值，回答「谁最容易被成本吃掉」：

| 因子族 | 倾斜 | 含义 |
|---|---|---|
| `Volatility*` | **+0.41 ~ +0.63** | 系统性买高成本票，毛 alpha 要跨更高门槛 |
| `LogVolumeRatio*` | +0.01 ~ +0.03 | 中性 |
| `Momentum*` | −0.12 ~ −0.23 | 成本友好 |
| `Return*` | −0.26 ~ −0.33 | 成本友好 |

> `Volatility*` 的高倾斜是**定义使然**：它的信号本身就是波动率，与成本模型的波动项
> 近乎同义。这不是估计偏差，而是一个经济事实 —— 按波动率选股，就等于按交易成本选股。
> 结合 §7.1：`Volatility20` 是唯一通过 DK 检验的因子（t=3.30），**同时也是成本最贵的**。
> 这两个事实必须一起看，单独看任何一个都会得出错误结论。

#### 规模项在散户/中型账户下不起作用

规模项 = `(成交股数 / 日均成交量) × size_factor`。实测 $100,000 账户下
占总成本 **0.0024%**，$10,000,000 下也只到约 1bp。**要到机构规模才显著。**
运行时会打印分量占比，避免误以为它在工作。

#### 正确性保证

`backtest_hetero.py` 复制了上游 `backtest` 的逐日逻辑（漂移、execution_lag、
换手口径全部照抄），只替换成本项。**当成本退化为常数时，必须与上游逐位相同** ——
这条等价性由自检自动校验，最大绝对差 ~1e-19（纯浮点误差）。

---

## 8. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `YFRateLimitError: Too Many Requests` | Yahoo 限流 | 等几分钟到几小时；或先放一份 `_fb_cache/raw_{start}_{end}.pkl` 再跑 |
| `票池太小：单个交易日最多只有 N 只…` | 票池 < 30，无法做风格回归 | 用完整 51 只票池 |
| `train 终点 … 不早于 valid 起点 …` | 三段切分重叠 | 调整 `--train-end` / `--valid-start` / `--test-start` |
| 风格残差 IC 全是 NaN | （已加守卫）任何一段无可用日期 | 现在会直接报错并说明原因 |
| `找不到 FactorBench 仓库` | 仓库不在默认路径 | `set FACTORBENCH_REPO=D:\path\to\FactorBench-Release` |
| `cov_type 必须是 (...) 之一` | `LearningEngine` 传了未知的协方差类型 | 用 `driscoll_kraay`（默认）/ `two_way` / `cluster_date` / `iid` |

---

## 9. 与论文的差异（可比性提醒）

| 项 | 论文 | 本项目 |
|---|---|---|
| Universe | 5 市场 782 只 | 美股 51 只（S&P100 子集） |
| 目标 | 原始 20 日收益 | 相对 SPY 超额（可切 `--target raw`） |
| 切分 | 2017-2021 / 2022-2023 / 2024-2025 | 对齐 AI_finance 参数 |
| 合成 | ridge，20 日调仓 | 同（ridge） |
| 因子来源 | 9 个自动挖掘方法 | 人工设计的 15 个 |

**所以：本项目产出的绝对数字不要与论文表格横向比。**
可比的是**同一批因子内部的相对结构**，以及风格中性化前后的落差。

---

## 10. 状态

- ✅ 适配层、运行器、自检、合成数据生成器：已实现并验证
- ✅ 四级评估全链路：合成数据跑通（风格 R² 与因子性质吻合，验证了实现正确性）
- ✅ 统计推断修正：`LearningEngine` 改用 Driscoll-Kraay，滞后按重叠长度取
- ✅ 经验贝叶斯收缩：IC 表自动输出 `ic_eb_shrunk.csv`
- ✅ 成本敏感性：解析计算，输出 `cost_sensitivity.csv` 与断本点
- ✅ 异质成本模型：`--cost-model liquidity`，ATR/成交量定价 + 因子成本倾斜
- ✅ 自检五项全过（`python verify_factorbench.py`，完全离线）
- ⏳ **真实数据的完整运行：待执行**（Yahoo 限流中）
- 跑法：限流恢复后
  `python run_factorbench_eval.py --stage all --cost-model liquidity`

### 产物一览（`factorbench_out/`）

| 文件 | 内容 |
|---|---|
| `ic_table.csv` | 每因子 × 每段 IC / RankIC / ICIR / HAC t / 准入 |
| `ic_eb_shrunk.csv` | 经验贝叶斯收缩后的 IC 与 `Var(t)` |
| `style_neutralisation.csv` | 原始 vs 剥风格后的残差 IC、ΔIC、风格 R² |
| `redundancy_spearman.csv` | 因子间 \|Spearman\| 矩阵 |
| `portfolio_performance.csv` | 多头/多空净 CAGR / Sharpe / 回撤 / 换手 |
| `cost_sensitivity.csv` | 各成本档净指标与断本点 |
| `cost_model_comparison.csv` | 统一成本 vs 异质成本对照 |
| `factor_cost_tilt.csv` | 因子成本倾斜 |
| `backtest_daily_*.csv` | 逐日净值/成本/换手明细 |

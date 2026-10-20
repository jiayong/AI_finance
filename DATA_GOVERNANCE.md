# 数据治理说明

本文件回答一个问题：**从 Yahoo 上抓下来的原始行情，是怎么变成能进模型的样本矩阵的，
每一步做了哪些取舍、这些取舍又会在哪里引入偏差。**

代码位置：

| 阶段 | 文件 | 关键函数 |
|---|---|---|
| 抓取与目标构造 | `data_engine.py` | `DataEngine.download_data()` |
| 因子计算 | `factor_engine.py` | `add_momentum` / `add_volatility` / `add_volume` / `add_shortterm_return` |
| 标准化与切分 | `preprocess_engine.py` | `normalize()` / `split_data()` |
| 因子面板（FactorBench 侧） | `factorbench_adapter.py` | `download_raw()` / `to_panel()` / `add_target()` / `make_split()` |

---

## 1. 数据源

| 项 | 取值 | 说明 |
|---|---|---|
| 行情源 | Yahoo Finance，经 `yfinance` | 唯一外部数据源 |
| 标的 | 50 只美股（`tickers.py`） | 按板块手工挑选的大盘股，非指数成分 |
| 基准 | `SPY` | 唯一基准，用于构造超额收益 |
| 区间 | `2010-01-01` ~ `2026-01-01` | 15 年 |
| 频率 | 日频 | 无盘中数据 |
| 复权 | `auto_adjust=True` | 价格已调整拆股与分红，因此**不需要**再单独处理复权 |
| 字段 | `Close`、`Volume` | 只用这两个；OHLC 其余三项未取 |

**为什么只有 Close 和 Volume。** `data_engine.py:19-20` 明确只 xs 出这两列。
后果见 §6「结构性缺口」——因子设计因此限定在「只依赖收盘价与成交量」的族里。
这不是疏忽，是刻意的约束：它同时保证了**没有前视泄漏**。

---

## 2. 数据形状与单位

### 2.1 原始（yfinance 返回）

`yf.download(..., group_by="columns")` 返回 **MultiIndex 列** DataFrame，
列索引形如 `(Price, Ticker)`，行索引为交易日。

### 2.2 引擎内部（长表）

`data_engine.py:22-28` 用 `stack()` 压成长表，再 merge：

```
Date (datetime64) | Ticker (str) | Close (float) | Volume (int64)
                     + DailyReturn | FutureReturn{H} | SPYClose | SPYReturn{H} | NetReturn{H}
```

每行 = 某个交易日 × 某只票。排序键 `["Ticker", "Date"]`。

### 2.3 FactorBench 面板

`factorbench_adapter.py` 输出 `(date, asset)` **MultiIndex 面板**，8 个标准字段：

`open` `high` `low` `close` `volume` `vwap` `returns` `market_cap`

其中 6 个由 §6 的规则派生。

---

## 3. 数据流（逐阶段）

```
yfinance ──auto_adjust──> MultiIndex(Price,Ticker)
                              │ xs("Close"/"Volume") + stack
                              v
                         长表 (Date,Ticker,Close,Volume)
                              │ groupby(Ticker)
                              ├─> DailyReturn      = Close.pct_change()
                              └─> FutureReturn{H}  = Close.shift(-H)/Close − 1
                              │ merge SPY on Date
                              ├─> SPYReturn{H}     = SPYClose.shift(-H)/SPYClose − 1
                              └─> NetReturn{H}     = FutureReturn{H} − SPYReturn{H}   ← 目标
                              │ FactorEngine 追加特征列
                              v
                         长表 + 15 个因子列
                              │ groupby(Date) 横截面标准化
                              ├─> {feature}_z = (x − mean_date) / std_date
                              └─> inf → NaN，再 dropna(特征 + 目标)
                              │ split_data 按日期切分 + purge
                              v
                    train / validation / test
```

### 3.1 目标变量的定义（最关键的一条）

```
NetReturn{H} = FutureReturn{H} − SPYReturn{H}
```

即**基准相对超额收益**，不是原始收益。

- `FutureReturn{H}` = 未来 H 个交易日的**价格涨幅**，不含分红再投资（价格已因 `auto_adjust` 调整过历史，故等价于总收益的前向实现）
- `SPYReturn{H}` 用同一个 H 对齐，按 `Date` 左连接

**推断影响**：超额化会把 β≈1 的大盘股的市场成分扣掉，因子 IC 因此**机械性变小**。
FactorBench 原论文用的是原始收益，所以 `--target net` 与 `--target raw` 会给出不同量级，
两者都跑才能和论文对齐。`run_factorbench_eval.py` 的 `--target` 就是为此而设。

### 3.2 因子（15 个）

| 族 | 定义 | 参数 |
|---|---|---|
| `Momentum{L}_{S}` | 过去 L 日收益，跳过最近 S 日 | L∈{100,60,20,10}, S=5 |
| `Volatility{H}` | 过去 H 日收益的滚动标准差 | H∈{100,60,20,10} |
| `LogVolumeRatio{H}` | 当前成交量 / H 日均量，取对数 | H∈{100,60,20,10} |
| `Return{H}` | 过去 H 日收益 | H∈{20,10,5} |

跳过最近 5 日是动量研究的标准做法（避免短期反转污染）。见 `FACTOR_TUTORIAL.md`。

因子集由 `main.py` 的 `FACTOR_SPECS` 表驱动，`FactorEngine.add_factor(kind, params)`
按配置分发，无需改引擎代码。

---

## 4. 清洗规则

| 规则 | 位置 | 说明 |
|---|---|---|
| 横截面 z-score | `preprocess_engine.py:19` | `groupby("Date")` 做 `(x−mean)/std`，**逐日**，不跨日 |
| `inf → NaN` | `preprocess_engine.py:22` | 除以零体积等情形 |
| `dropna` | `preprocess_engine.py:24` | 丢掉特征或目标缺失的行 |
| 缩尾 winsorize | `factorbench_adapter.py`（`WINSOR=(0.01,0.99)`） | 仅 FactorBench 侧面板；**逐日横截面** |
| 退化解过滤 | `DEGENERATE_REL=1e-9` / `FLAT_REL=1e-12` | 剔除常数因子与全零截面 |
| 最小截面数 | `MIN_OBS_PER_DATE=30` | 单日不足 30 只票则整日不参与 IC / 风格回归 |

**注意 z-score 与 winsorize 的顺序。** AI_finance 侧只做 z-score，不做缩尾；
FactorBench 侧先缩尾后中性化。两条链的数值因此不完全相同，这是已知差异，
不是 bug——它们各自服务不同的评估口径。

---

## 5. 切分与 purge（防泄漏的核心）

### 5.1 AI_finance 侧（`preprocess_engine.py:27-68`）

按**日期**切，且在边界处挖掉 `forecast_horizon` 天：

```python
train_cutoff      = dates_before_validation.iloc[-forecast_horizon]   # 2020-01-01 前推 H 天
validation_cutoff = dates_before_test.iloc[-max_horizon]              # test_date 前推 max_horizon 天
```

- train: `Date < train_cutoff`
- validation: `start_date ≤ Date < validation_cutoff`
- test: `Date ≥ validation_date`

因为目标是 H 日**前瞻**收益，边界前最后 H 天的标签依赖跨越边界的数据。
不 purge 就等于让训练集看到验证期的价格。**这是本管线里最容易被忽略、后果最严重的一步。**

### 5.2 FactorBench 侧（`factorbench_adapter.py: make_split`）

日期区间切分 + 显式重叠校验：

| 段 | 区间 |
|---|---|
| train | `2010-01-01` ~ `2019-12-31` |
| valid | `2020-01-01` ~ `2023-12-31` |
| test | `2024-01-01` ~ `2026-10-01` |

任一段落重叠或非时序时直接抛 `ValueError`，不静默通过。

**为什么不直接用 FactorBench 的 `DEFAULT_SPLIT`**：那套切分与本项目
`validation_date=2020-01-01` 的设定不一致，且没有考虑 2024 年后才有的数据。
统一到 AI_finance 的参数下，两条链的结论才可比。

### 5.3 IC 方向的确定

`sign_for()` 包一层 `train_sign(daily_ic(...))`：**方向只在训练集上确定，
原样应用到 valid/test**。绝不用全样本定方向——那是最典型的隐性前视。

---

## 6. 结构性缺口（必须知道的偏差来源）

### 6.1 只有 Close / Volume，其余 6 个字段是派生的

`factorbench_adapter.py` 的派生规则：

| 字段 | 派生方式 | 风险 |
|---|---|---|
| `open` | `= close` | **近似**。任何依赖开盘价的因子（隔夜跳空、AST 类）都不可信 |
| `high` / `low` | `= close` | **近似**。真实振幅不可得，ATR 只能由收益波动替代 |
| `vwap` | `= (high+low+close)/3` | 典型价公式，**不是**真实 VWAP（无成交额数据） |
| `returns` | `close.pct_change()` | 可靠 |
| `market_cap` | 未提供 | 依赖市值的因子不可用 |

**但本项目 15 个因子只用到 close 与 volume**，因此上述近似**不构成泄漏或错误**，
只影响未来扩展。`open=close` 的实际影响范围严格限定在「将来要加的、用开盘价的因子」。

### 6.2 静态 universe（幸存者偏差）

50 只票是**今天**的名单，回填到 2010 年。真实世界中 2010 年持有这些票的人
并不知道它们能活到今天。这会让所有回测系统性偏乐观。

这是本项目**最大的、无法靠代码修掉的**偏差。缓解手段只能是：把结论当作
「在这 50 只票上的相对排序」，而不是「策略的绝对收益」。

### 6.3 无退市、无停牌、无涨跌停

`yfinance` 不返回这些。退市票直接缺席 = 幸存者偏差的又一来源。

### 6.4 数据修订不可得

Yahoo 给的是当前修订后的历史。真实交易时看到的是当时的值（含错误）。
本管线无法区分「当时发布值」与「事后修订值」。

---

## 7. 缓存与失效

| 层 | 位置 | 键 | 失效 |
|---|---|---|---|
| AI_finance 行情 | `data_cache/`（另仓） | 文件名含起止日 | 手删 |
| FactorBench 原始 | `_fb_cache/raw_{start}_{end}.pkl` | 起止日期 | 换日期即换文件 |
| FactorBench 结果 | `factorbench_out/` | 每次运行覆盖 | 无 |

`download_raw()` 的策略（`factorbench_adapter.py:113`）：

1. 先查 `raw_{start}_{end}.pkl`，命中直接返回
2. 未命中则下载，失败按 5s / 10s / 20s 指数退避，最多 4 次
3. 4 次全败 → **回退到旧缓存**（若有），并打印警告
4. 无缓存可回退 → 抛异常，绝不返回空表

第 3 步的设计意图：Yahoo 限流有很强的时段性，在一次成功的下载之后，
短时重跑不应该因为限流而整体失败。但它也意味着**缓存可能比你以为的旧**——
所以警告必须看。

`.gitignore` 已排除 `_fb_cache*/`、`_fb_out*/`、`*.parquet`、`*.pkl`。
**数据不入库**，新机器必须自己拉。

---

## 8. 可复现性的边界

| 能复现 | 不能复现 |
|---|---|
| 代码逻辑与全部计算步骤 | Yahoo 在 2026-10 当天返回的行情（会变、会限流） |
| 因子定义与参数 | 真实交易时的历史快照（§6.4） |
| 切分与 purge 规则 | 当时的 universe 成分（§6.2） |
| 合成数据上的完整链路（`_make_synthetic_cache.py`，固定 seed） | 真实数据上的精确数值 |
| 统计推断方法（`hac_inference.py`） | |

**结论：另一台机器能复现「方法与结论方向」，不能复现「小数点后三位的数字」。**
这是所有基于免费行情的研究的共同边界，不是本项目的实现缺陷。

---

## 9. 校验清单

新机器上按顺序跑，任一步失败就不要往下走：

```bash
# 1. 环境
python -m pip install -r requirements-frozen.txt

# 2. 离线全链路自检（不需要网络，5 项）
python verify_factorbench.py

# 3. 单元测试（15 项因子一致性）
pytest test_factor_engine.py test_factor_specs.py -q

# 4. 真实数据（需要 Yahoo 可用）
python run_factorbench_eval.py --stage all --cost-model liquidity
```

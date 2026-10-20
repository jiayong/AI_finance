# 换机复现指南

本仓库是 **AI_finance**（[Yazdan-Babazadeh/AI_finance](https://github.com/Yazdan-Babazadeh/AI_finance)）
的研究扩展。在原五层管线之上，加了四件事：

1. **统计推断修正** —— 把 iid / 单维聚类标准误换成 Driscoll-Kraay
2. **FactorBench 评估管线接入** —— IC 表 / 风格中性化 / 冗余 / 组合回测
3. **经验贝叶斯收缩** —— 按 Chen & Welch (2026) 的公式校正 t 统计量
4. **交易成本建模** —— 统一 bps、按流动性定价的异质成本、成本敏感性

目标：**在另一台机器上，从零把上述工作全部跑通。**

---

## 0. 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **3.12.x** | 实测 3.12.4；3.13 未验证 |
| 操作系统 | Windows / Linux / macOS 均可 | 本项目在 Windows 11 开发；脚本给了 PowerShell 与 bash 两版 |
| 磁盘 | ≥ 2 GB | 依赖约 800 MB，行情缓存约 20 MB，FactorBench 约 1 MB |
| 网络 | 需要 GitHub；行情部分需要 Yahoo | Yahoo 会限流，见 §5 |
| git | 任意近版本 | 需 SSH key 已加到 GitHub |

---

## 1. 克隆本仓库

```bash
git clone git@github.com:jiayong/AI_finance.git
cd AI_finance
git checkout medium-blog
```

> 上游原作者仓库是 `https://github.com/Yazdan-Babazadeh/AI_finance.git`。
> `origin` 指向**本 fork**，可以自由推送。若要拉上游更新：
> `git remote add upstream https://github.com/Yazdan-Babazadeh/AI_finance.git && git fetch upstream`

---

## 2. 安装依赖

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-frozen.txt
```

**用 `requirements-frozen.txt`，不要用 `requirements.txt`。**
后者是上游自带的（对应作者的环境），版本与本项目实测通过的不一致，且缺少
`statsmodels` / `scikit-learn` / `matplotlib` / `cvxpy`。两个文件都保留，
是为了不污染 fork 的上游历史。

关键版本（`requirements-frozen.txt` 已锁定）：

```
numpy 2.1.3   pandas 2.2.3   statsmodels 0.15.0   scikit-learn 1.5.2
scipy 1.17.1  pyarrow 24.0.0  yfinance 1.2.0      cvxpy 1.8.2
```

> **`linearmodels` 不需要装。** Driscoll-Kraay 是手写的（`hac_inference.py`），
> 刻意不引入该依赖。如果你装了也可以，两者互不干扰。

---

## 3. 获取 FactorBench（第二个仓库）

FactorBench 是评估管线的核心依赖，**不在本仓库内**，需要单独克隆到固定提交：

```bash
git clone https://github.com/ZhuoHan1998/FactorBench-Release.git
cd FactorBench-Release
git checkout 8dbc5b42e005a4038dd52e4790b30b467a5c1575
```

**务必 checkout 到上面这个 SHA。** 本项目的适配层是按这个版本的内部 API 写的
（`factor_bench.portfolio.backtest`、`BacktestConfig`、`performance_metrics`
的签名与语义）。上游若改了这些，`verify_factorbench.py` 会立刻报错——这正是自检存在的意义。

然后告诉本项目它在哪里，二选一：

```powershell
# PowerShell（临时，仅当前窗口）
$env:FACTORBENCH_REPO = "D:\workspace\FactorBench-Release"
```

```bash
# bash（临时）
export FACTORBENCH_REPO=/path/to/FactorBench-Release
```

**永久生效**（Windows）：

```powershell
[Environment]::SetEnvironmentVariable("FACTORBENCH_REPO", "D:\workspace\FactorBench-Release", "User")
```

不设也行——`factorbench_adapter.py:40` 的默认值是
`D:\workspace\FactorBench-Release`。但换机后路径几乎肯定不同，建议显式设。

---

## 4. 先跑离线自检（不需要网络）

**这是最重要的一步。** 五项全过，才说明代码与 FactorBench 的对接是完好的：

```bash
python verify_factorbench.py
```

输出应当以「全部通过」结尾。五项分别是：

| 检查 | 验的是什么 |
|---|---|
| 因子名一致性 | `factorbench_adapter.py` 生成的 15 个列名与 `factor_engine.py` 逐一对齐 |
| 合成数据全链路 | 在内存里造行情 → 因子 → IC → 风格中性化 → 冗余 → 回测，端到端跑通 |
| 风格门槛守卫 | 截面不足 30 只票时必须显式报错，而不是静默产出 NaN |
| 推断与成本不变量 | 零成本回测下 `net == gross`；成本对净收益严格线性 |
| 异质成本等价性 | `backtest_hetero.py` 在常数成本下与上游 `backtest` 逐位相同 |

**完全离线**，用的是 `_make_synthetic_cache.py` 的固定 seed 合成行情，
所以结果可重复。

再跑因子定义测试：

```bash
python test_factor_engine.py
python test_factor_specs.py
```

期望 `test_factor_engine.py` 输出 `passed 15/15`，
`test_factor_specs.py` 以 `FACTOR_SPECS 与分发器一致 ✓` 结尾。两者退出码都应为 0。

> **这两个是独立脚本，不是 pytest 用例。** 请直接 `python test_xxx.py` 运行；
> 用 `pytest test_factor_engine.py` 会显示 `no tests ran`——文件内部没有
> 命名为 `test_*` 的函数（断言函数都叫 `_`），pytest 收集不到。
> 这是刻意保持的：脚本可在无 pytest 的机器上直接跑，自检结果自解释。

---

## 5. 真实数据（需要 Yahoo 可用）

### 5.1 已知问题：Yahoo 限流

2026-10 期间，从本机 IP 下载 50 只票曾**持续 1 小时以上**返回
`YFRateLimitError: Too Many Requests`。这不是代码问题，是 IP 级限流。

应对顺序：

1. **换网络**（手机热点通常立刻可用）—— 最快
2. 隔几小时再试 —— 限流有窗口
3. 用离线合成数据跑通流程（§6），先确认代码没问题

适配层有 4 次指数退避重试（5s/10s/20s），并在全败时回退到旧缓存；
**但旧缓存会让结果和你的预期不一致**，看到警告就去查缓存日期。

### 5.2 完整评估

```bash
python run_factorbench_eval.py --stage all --cost-model liquidity
```

常用变体：

```bash
# 只跑 IC 表（最快，先确认数据能进来）
python run_factorbench_eval.py --stage ic

# 换预测周期（20 日是默认；5 日可区分「窗口重叠」与「真的没信息」）
python run_factorbench_eval.py --stage all --horizon 5

# 用原始收益而非超额收益（与 FactorBench 论文口径对齐）
python run_factorbench_eval.py --stage all --target raw

# 成本敏感性单看（10/20/25 bp 三档 + 断本点）
python run_factorbench_eval.py --stage portfolio --cost-bps 10
```

全部参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--stage` | `all` | `ic` / `style` / `redundancy` / `portfolio` / `all` |
| `--cache-dir` | `_fb_cache` | 行情缓存目录 |
| `--out-dir` | `factorbench_out` | 结果输出目录 |
| `--start` / `--end` | `2010-01-01` / `2026-10-01` | 数据区间 |
| `--train-end` | `2019-12-31` | 训练段结束 |
| `--valid-start` / `--valid-end` | `2020-01-01` / `2023-12-31` | 验证段 |
| `--test-start` | `2024-01-01` | 测试段开始 |
| `--horizon` | `20` | 预测周期（交易日） |
| `--target` | `net` | `net`=超额 / `raw`=原始收益 |
| `--rebalance-days` | `20` | 调仓间隔 |
| `--cost-bps` | `10.0` | 单边成本（`flat` 模式） |
| `--cost-model` | `flat` | `flat` / `liquidity`（按 ATR+成交量定价） |
| `--equity` | `100000` | 账户规模，`liquidity` 模式用 |

> `--train-end 2021-12-31` 这类改动必须同时给 `--valid-start`，
> 否则段会重叠，程序会显式抛错（这是刻意设计，不静默通过）。

### 5.3 原 AI_finance 主管线

```bash
python main.py
```

跑 17 个预测周期 × 8 个 alpha 的完整网格，耗时很长（小时级）。

⚠️ **已知问题**：`main.py` 里 `test_date = "2026-01-01"`，而 `end_date` 也是
`2026-01-01`，**测试集会为空**。要得到非空测试结果，把 `test_date` 改到
`end_date` 之前（例如 `2024-01-01`）。这是上游遗留的配置问题，本项目未改动它
以保持与上游可比；FactorBench 侧已用 `--test-start 2024-01-01` 绕开。

---

## 6. 离线跑通（不需要 Yahoo）

造一份固定 seed 的合成行情，验证整条链路：

```bash
# 生成 50 只票、2020-2021 的合成行情，写入缓存
python _make_synthetic_cache.py _fb_cache 2020-01-01 2021-12-31

# 用它跑完整评估
python run_factorbench_eval.py --stage all --cache-dir _fb_cache --out-dir _fb_out

# 跑经验贝叶斯收缩（读某个 ic_table.csv）
python eb_shrinkage.py _fb_out/ic_table.csv 20
```

合成数据的结果**只用于验证流程正确，不能用于投资判断**——
它的因子结构与真实行情无关。

---

## 7. 产物在哪

`factorbench_out/`（或 `--out-dir` 指定的目录）：

| 文件 | 内容 |
|---|---|
| `ic_table.csv` | 每因子 × 每段的 IC / RankIC / ICIR / HAC t 值 / 准入标记 |
| `ic_eb_shrunk.csv` | 经验贝叶斯收缩后的 IC 与 `Var(t)` |
| `style_neutralisation.csv` | 原始 vs 剥风格后的残差 IC、ΔIC、各风格 R² |
| `redundancy_spearman.csv` | 因子间 \|Spearman\| 相关矩阵 |
| `portfolio_performance.csv` | 多头/多空净 CAGR、Sharpe、最大回撤、换手 |
| `cost_sensitivity.csv` | 各成本档净指标 + 断本点 |
| `cost_model_comparison.csv` | 统一成本 vs 异质成本对照 |
| `factor_cost_tilt.csv` | 因子成本倾斜（信号 vs ATR/价格 的横截面相关） |
| `backtest_daily_*.csv` | 逐日净值 / 成本 / 换手明细 |

**先看 `ic_table.csv` 的 HAC t 列，不要看 ICIR。** 在重叠样本上 ICIR 会严重高估
显著性——这正是本项目做统计推断修正的原因。详见 `FACTORBENCH_README.md` §7.1。

---

## 8. 文档索引

| 文件 | 读它来了解 |
|---|---|
| `REPRODUCE.md` | 换机复现（本文件） |
| `DATA_GOVERNANCE.md` | 数据从哪来、怎么洗、有哪些偏差 |
| `FACTORBENCH_README.md` | FactorBench 接入的完整说明、口径差异、成本模型 |
| `FACTOR_TUTORIAL.md` | 因子工作模式的详细教程（约 1700 行） |
| `papers/README.md` | 两篇核心论文的要点与对应代码 |
| `requirements-frozen.txt` | 实测通过的依赖版本 |

---

## 9. 复现失败的常见原因

| 现象 | 原因 | 处理 |
|---|---|---|
| `FactorBench 仓库不存在` | 没克隆或 `FACTORBENCH_REPO` 没设 | §3 |
| `YFRateLimitError` | Yahoo 限流 | §5.1，换网络 |
| 风格中性化全 NaN | 截面股票数 < 30 | 用 `tickers.py` 的完整 50 只，别只跑几只 |
| `no module named factor_bench` | FactorBench 装了但没在 `sys.path` | 确认 `FACTORBENCH_REPO` 指向**仓库根目录**（含 `factor_bench/`） |
| 段重叠 ValueError | 改了 `--train-end` 没改 `--valid-start` | 按 §5.2 表格配套修改 |
| pandas 版本报错 | 装了 `requirements.txt` 而非 frozen | §2 |
| 中文输出乱码 | 控制台编码 | `set PYTHONIOENCODING=utf-8` |

---

## 10. 这台机器上的现状

| 项 | 状态 |
|---|---|
| 代码 | 已提交，可推送 |
| 离线自检 | **5/5 通过** |
| 单元测试 | **15/15 通过** |
| 真实数据完整运行 | **待执行**（Yahoo 限流中，截至本次整理） |

> 真实数据跑出来的数字，才是能写进结论的数字。合成数据上的一切结论
> 只说明「管线是通的」，不说明「因子是有效的」。

# ETF 两阶段选股实验项目

## 项目概述

本项目是一个研究 ETF 横截面选股中“两阶段排序”流程的实验代码库。核心流程为：

1. **第一阶段**：使用传统机器学习模型对 ETF 的未来收益进行预测与初筛。
2. **第二阶段**：使用大语言模型（LLM）对第一阶段候选池进行再次排序。
3. **回测与汇总**：对比 baseline（仅第一阶段）与 LLM 两阶段策略的表现，生成实验报告。

当前版本固定以 `Y_future_5d_return`（未来5日收益率）为核心预测目标，支持四种传统模型和两种 LLM，并包含最小稳健性检验 scenario。

## 环境要求

- Python 3.10+
- 依赖包见 `requirements.txt`

## 安装与配置

```bash
# 克隆项目
git clone https://github.com/MogekoQwQ/etf.git
cd etf-main

# 创建虚拟环境（Windows PowerShell 示例）
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt

# 配置 LLM API 密钥
cp config/llm_config.example.yaml config/llm_config.local.yaml
# 编辑 config/llm_config.local.yaml，填入实际的 API 密钥
```

## 项目结构（主流程相关）

```text
etf-main/
├─ data/                    # 数据目录
│  ├─ raw/                 # 原始数据
│  │  ├─ etf_list.csv      # ETF 列表
│  │  └─ etf_daily/        # 日频行情（按代码存储）
│  ├─ processed/           # 处理后的数据
│  │  ├─ factor_data/      # 单只 ETF 因子文件
│  │  └─ all_etf_factors.csv  # 合并后的因子数据集
│  └─ predictions/         # 第一阶段预测结果
│     ├─ random_forest/
│     ├─ linear/
│     ├─ lightgbm/
│     └─ xgboost/
│
├─ runs/                   # 活动运行结果
│  ├─ training/           # 第一阶段训练评估
│  │  ├─ random_forest/
│  │  ├─ linear/
│  │  ├─ lightgbm/
│  │  └─ xgboost/
│  └─ backtests/          # 两阶段回测结果
│     ├─ <传统模型>/       # 如 lightgbm/
│     │  ├─ deepseek_chat/    # LLM 变体目录
│     │  │  ├─ default/       # scenario: top50/top10
│     │  │  ├─ top30_top10/   # scenario: top30/top10
│     │  │  └─ top50_top5/    # scenario: top50/top5
│     │  ├─ gemini_2_5_flash_lite/  # 另一 LLM
│     │  └─ logs/           # LLM 请求日志与缓存
│     └─ benchmarks/        # 基准策略（等权、单因子）
│
├─ experiments/            # 实验汇总与快照
│  ├─ summary/            # 跨运行汇总报告
│  └─ snapshots/          # 实验快照归档（按 exp_main/ 组织）
│
├─ config/                # 配置文件
│  ├─ llm_config.example.yaml
│  ├─ llm_config.local.yaml   # 本地 LLM 配置（优先读取）
│  ├─ backtest_defaults.yaml
│  └─ experiment_defaults.yaml
│
├─ scripts/               # 主流程脚本（见下方说明）
├─ requirements.txt       # Python 依赖
└─ README.md             # 本文档
```

**说明**：`scripts/` 目录下还包含若干分析、解释、迁移等辅助脚本，用于调试或特定实验，不列为主流程文件。

## 核心功能模块（主流程脚本）

### 1. 数据下载与因子生成

- `download_etf.py`：从 akshare 下载 ETF 列表与日频行情，输出至 `data/raw/`。
- `compute_etf_factors.py`：对单只 ETF 计算技术因子（动量、波动率、成交量等），输出至 `data/processed/factor_data/`。
- `merge_etf_factors.py`：合并所有 ETF 因子，生成 `data/processed/all_etf_factors.csv`。

### 2. 第一阶段传统模型训练

- `train_traditional_multi_target_backtest.py`：训练传统模型，输出预测结果与评估指标。默认目标为 `Y_future_5d_return`，支持 `--target` 指定目标列。
- **支持的传统模型**（通过 `--traditional-model` 指定）：
  - `random_forest`（随机森林）
  - `linear`（线性回归）
  - `lightgbm`
  - `xgboost`

### 3. 第二阶段 LLM 排序
- `llm_ranking.py`：提供 LLM 排序与解释生成的底层函数。**当前使用的 Prompt 模板内置于该文件**（`RANKING_PROMPT_TEMPLATE`），输出格式已修改为**只返回 Top N 的评分结果**（详见下方 Prompt 设计说明）。
- **支持的 LLM**（通过 `--second-stage-llm` 指定）：
  - `deepseek-chat`
  - `gemini-2.5-flash-lite`

#### 解释性输出分支
该分支用于生成 LLM 排序的**解释性输出**。通过 `--enable-explanations` 参数启用后，除常规排序结果外，还会为每个调仓日生成：
- **候选 ETF 排序分值**：LLM 对 top N 候选 ETF 的综合评分。
- **信号标签**（signal_tags）：如“动量”、“趋势”、“价值”等，反映排序时关注的信息类别。
- **风险标签**（risk_tags）：如“高波动”、“流动性风险”等，标识潜在风险因素。

**用途**：
- 展示 LLM 排序的依据与决策线索。
- 揭示候选资产间的相对差异。
- 建立风险/收益线索的对应关系，增强结果可解释性。

**输出文件示例**：
```text
runs/backtests/<traditional_model>/<second_stage_llm>/<scenario>/reports/explanations/
├── YYYYMMDD_explanation.json          # 单日解释结果（含标签）
├── YYYYMMDD_explanation_summary.md    # 解释摘要
└── explanations_manifest.json         # 解释生成元数据
```

### 4. 两阶段回测
- `two_stage_backtest.py`：核心回测脚本。读取第一阶段预测结果，执行 baseline（仅传统模型排序）与 LLM 两阶段策略的回测，输出绩效指标、换手率、净值曲线等。
- 通过 `--top-n-first` 与 `--top-n-final` 控制候选池大小与最终持仓数，自动映射为 scenario 目录（如 `default`、`top30_top10`、`top50_top5`）。

### 5. 单次回测报告生成
- `two_stage_reporter.py`：根据单次回测目录生成 HTML 报告，包含净值对比、收益分布、换手率等图表。

### 6. 实验套件与汇总
- `run_pipeline.py`：端到端执行单个模型+LLM 的实验流程，可跳过已完成的步骤（数据下载、因子计算、训练）。
- `run_experiment_suite.py`：批量运行多个传统模型与变体（baseline、LLM、mock_llm），并将结果归档至 `experiments/snapshots/`。
- `experiment_summary.py`：读取 `runs/backtests/` 下的结果，生成跨模型汇总报告（CSV、JSON、HTML）。



## 脚本一览

**主流程脚本**

- download_etf.py：从 data/raw/etf_list.csv 读取 ETF 列表，用 akshare 下载每只 ETF 的日线行情，保存到 data/raw/etf_daily/。带断点续传，进度写在 data/raw/download_progress.txt。

- compute_etf_factors.py：逐只读取 etf_daily/*.csv，计算技术因子和标签列，输出到 data/processed/factor_data/*_factor.csv。当前主要因子是 momentum_20、volatility_20、volume_mean_20、return_mean_20、amplitude_mean_20、turnover_mean_20、MA_5、MA_10；标签是未来收益/波动变化。

- merge_etf_factors.py：把所有单 ETF 因子文件合并成 data/processed/all_etf_factors.csv，并写一个 dataset_manifest.json。

- train_traditional_multi_target_backtest.py：第一阶段传统模型训练脚本。支持 random_forest、linear、lightgbm、xgboost。按日期 80/20 切训练/测试，额外做按日截面 z-score 特征增强，输出训练评估、特征重要性图、预测文件、split 信息、ranking metrics。

- two_stage_backtest.py

  ：核心回测脚本。读取第一阶段预测结果，做两组对比：

  - 基线：按传统模型预测值直接选 TopN。
  - 两阶段：先取 Top top_n_first 候选池，再交给 LLM 或 Mock LLM 重排，取最终 Top top_n_final。
    还会计算换手、交易成本、滑点、净收益，并额外生成 3 类 benchmark：510300 市场基准、全样本等权、简单单因子规则。

- two_stage_reporter.py：把单次回测目录里的 performance_report.json、performance_metrics.csv、backtest_results.csv 和图片整理成一个离线 HTML 报告。

**LLM 与解释相关**

- llm_ranking.py

  ：LLM 排序与解释的底层库。负责：

  - 读取 llm_config；
  - 调 DeepSeek / Gemini / OpenRouter 兼容接口；
  - 构造 ranking / explanation prompt；
  - 解析 JSON；
  - 保存 prompt、response、result 日志；
  - 生成 explanation 结构化文件。

- explanation_reporter.py：把 logs/.../explanations/YYYYMMDD/ 里的解释结果渲染成单日 HTML 报告和汇总页。它更像解释结果浏览器。

- rebuild_explanation_tags.py：不是重新排序，而是对“已经固定的两阶段最终选股结果”补生成结构化解释标签，如 signal_tags、risk_tags、style_tag、confidence_tag，并输出标签频率、共现统计和 Markdown 报告。

- explanation_utils.py：解释结果的存储/读取/校验工具。llm_ranking.py 里实际会用它把 explanation 落盘。

**配置与通用工具**

- traditional_model_config.py：整个项目的路径规范、模型名规范化、模型工厂和输出目录约定都在这里。很多脚本都依赖它。
- market_data_utils.py：CSV 编码兜底读取、日期列识别、行情列名标准化、必需列校验。

**分析、统计、迁移类脚本**

- analyze_candidate_pool_quality.py：离线分析不同第一阶段模型的候选池质量。核心看候选池预测分数离散度、候选池真实收益离散度、Top10 相对 Top50 的收益梯度。
- pairwise_significance_test.py：对 default 场景下各模型 + 各 LLM 的“基线 vs 两阶段”逐期净收益做配对样本 t 检验，输出表 5-2 风格结果。
- plot_net_cumulative_return_curves.py：从已有 backtest_results.csv 中推导净累计收益曲线，按模型画 baseline / DeepSeek / Gemini 三条线，不重跑实验。
- raw_data_stats.py：一次性描述性统计脚本。直接读 data/processed/all_etf_factors.csv，输出 raw_data_descriptive_stats.csv/md 到项目根目录。

**汇总实验相关**

- run_pipeline.py：单模型端到端编排器。串起下载、因子、训练、两阶段回测，可用 --skip-* 跳过已完成步骤。
- run_experiment_suite.py：批量实验编排器。循环跑多个模型 / 变体（baseline、llm、mock_llm），并把结果归档到 experiments/snapshots/...，最后调用 experiment_summary.py 做总汇总。
- experiment_summary.py：跨实验汇总脚本。会同时扫描“当前活动结果”和“快照归档结果”，抽取默认场景下 baseline / DeepSeek / Gemini 的指标，生成 CSV / JSON / Markdown / HTML 汇总，以及图表；还单独做 LightGBM+DeepSeek 的 scenario 稳健性汇总。

## 实验设计

### 主实验框架
- **传统模型**：4 个（random_forest、linear、lightgbm、xgboost）。
- **对比策略**：baseline（仅第一阶段排序） vs. LLM 两阶段策略。
- **预测目标**：`Y_future_5d_return`。
- **调仓频率**：默认每周调仓（`--rebalancing-freq W`）。

### LLM 支持
- 当前支持 `deepseek-chat` 与 `gemini-2.5-flash-lite`。
- 不同 LLM 的结果完全隔离存储，避免覆盖。

### 最小稳健性检验（scenario）
通过 `--top-n-first` 与 `--top-n-final` 参数定义三组 scenario：

1. **default**：`top_n_first=50`，`top_n_final=10`。
2. **top30_top10**：`top_n_first=30`，`top_n_final=10`。
3. **top50_top5**：`top_n_first=50`，`top_n_final=5`。

scenario 名称自动生成，结果分别存入对应的子目录。

### 基准策略
回测中包含两种简单基准：
- 样本池等权组合。
- 单因子动量规则（20日动量）。

基准结果输出至 `runs/backtests/benchmarks/<scenario>/`。

## 使用方法

### 1. 完整端到端流程（单个模型+LLM）
```powershell
python scripts/run_pipeline.py --traditional-model lightgbm --target Y_future_5d_return --second-stage-llm deepseek-chat --llm-config config/llm_config.local.yaml
```
可通过 `--skip-download`、`--skip-factors`、`--skip-traditional` 跳过已完成的步骤。

### 2. 指定 scenario
```powershell
python scripts/run_pipeline.py --traditional-model lightgbm --target Y_future_5d_return --second-stage-llm deepseek-chat --llm-config config/llm_config.local.yaml --top-n-first 30 --top-n-final 10
```

### 3. 批量实验套件（论文实验）
```powershell
python scripts/run_experiment_suite.py --baseline-models random_forest linear lightgbm xgboost --llm-models lightgbm --llm-config config/llm_config.local.yaml
```
该命令将运行 baseline 策略（四个模型）和 LLM 两阶段策略（仅 lightgbm），结果自动归档至 `experiments/snapshots/`。

### 4. 生成单次回测报告
```powershell
python scripts/two_stage_reporter.py --input-dir runs/backtests/lightgbm/deepseek_chat/default --output runs/backtests/lightgbm/deepseek_chat/default/reports/report_summary.html
```

### 5. 生成实验汇总报告
```powershell
python scripts/experiment_summary.py
```

## 输出结果

### 第一阶段训练输出
- **目录**：`runs/training/<traditional_model>/`
- **主要文件**：
  - `model_results.txt`：模型性能摘要。
  - `ranking_metrics.csv`：排序指标。
  - `pred_vs_actual_*.png`：预测 vs. 实际散点图。
  - `feature_importance_*.png`：特征重要性图。
  - `run_manifest.json`：运行元数据。

### 第一阶段预测输出
- **目录**：`data/predictions/<traditional_model>/`
- **主要文件**：
  - `test_set_with_predictions.csv`：测试集预测值。
  - `split_info.json`：数据集划分信息。
  - `prediction_manifest.json`：预测元数据。

### 第二阶段回测输出
- **目录**：`runs/backtests/<traditional_model>/<second_stage_llm>/<scenario>/`
- **主要文件**：
  - `backtest_results.csv`：每日持仓与收益。
  - `performance_metrics.csv`：绩效指标汇总。
  - `performance_report.json`：结构化绩效报告。
  - `cumulative_returns.png`：净值曲线对比图。
  - `returns_distribution.png`：收益分布图。
  - `run_manifest.json`：回测元数据。
  - `reports/report_summary.html`：HTML 报告。

### LLM 日志与缓存
- **目录**：`runs/backtests/<traditional_model>/logs/<second_stage_llm>/`
- **主要文件**：
  - `prompt_*.txt`：发送给 LLM 的完整 prompt。
  - `response_*.txt`：LLM 原始响应。
  - `result_*.csv`：解析后的排序结果。
  - `rebalance_YYYYMMDD_<fingerprint12>.csv`：单日缓存文件（基于候选池 fingerprint）。

### 实验汇总输出
- **目录**：`experiments/summary/`
- **主要文件**：
  - `experiment_overview.csv`：跨模型绩效概览。
  - `experiment_summary.json`：结构化汇总数据。
  - `experiment_summary.md` / `.html`：可读报告。
  - `charts/*.png`：汇总图表。

## Prompt 设计说明

当前使用的 Prompt 模板位于 `scripts/llm_ranking.py` 中的 `RANKING_PROMPT_TEMPLATE`。核心设计要求如下：

### 1. 角色与任务
- 扮演 ETF 横截面排序的投资研究助手。
- 仅根据输入的 ETF 因子数值进行横向比较，不进行时间序列预测，不引入外部知识。

### 2. 输入数据格式
- 包含调仓日期、候选 ETF 数量及各 ETF 的因子数值（已标准化）。
- 因子包括动量、波动率、成交量、移动平均线等技术指标。

### 3. 输出格式
- **只返回 Top N 的评分结果**，其中 N 由 `top_n_final` 参数指定。
- 输出必须为合法 JSON，仅包含 `rankings` 字段，无额外解释文本。
- `rankings` 数组中每个元素包含 `code`（ETF 代码）和 `score`（综合评分，0~1）。
- 当前模板（`RANKING_PROMPT_TEMPLATE`）已明确要求只返回指定数量的 top ETF，与运行时附加的指令一致。
- 示例：
  ```json
  {
    "rankings": [
      {"code": "510300", "score": 0.95},
      {"code": "510500", "score": 0.87}
    ]
  }
  ```

### 4. 关键约束
- 禁止联网搜索或使用外部知识。
- 禁止进行时间序列预测。
- 排序应综合考虑收益潜力与风险控制，追求风险调整后收益。

### 5. 实际实现
在 `llm_ranking.py` 的 `rank_etfs_by_llm` 函数中，基础 prompt 后会自动附加：
```text
Only return the top {int(top_n_final)} ETFs in `rankings`. Do not include the remaining candidates.
```
确保 LLM 仅输出所需数量的 ETF，减少 token 消耗与解析复杂度。

### 6. Prompt使用场景
根据实验需求，系统支持三种不同的 Prompt 使用场景：

#### 场景一：仅启用排序（基础实验）
- **使用的模板**：`RANKING_PROMPT_TEMPLATE`（位于 `scripts/llm_ranking.py`）
- **调用函数**：`rank_etfs_by_llm()`，参数 `enable_explanations=False`
- **输出**：仅包含 ETF 代码与评分的排序结果
- **启用方式**：运行两阶段回测时不添加 `--enable-explanations` 参数

#### 场景二：排序 + 解释生成（完整实验）
- **使用的模板**：
  - 排序：`RANKING_PROMPT_TEMPLATE`
  - 解释：`EXPLANATION_PROMPT_TEMPLATE`（位于同一文件）
- **调用流程**：
  1. `rank_etfs_by_llm(..., enable_explanations=True)` 生成排序结果
  2. `generate_explanations_for_date()` 为排序结果生成解释与标签
- **输出**：排序结果 + 解释性 JSON（含 signal_tags、risk_tags 等）
- **启用方式**：运行两阶段回测时添加 `--enable-explanations` 参数

#### 场景三：基于现有结果的解释生成（后处理）
- **使用的模板**：
  - 因子说明：`EXPLANATION_PROMPT_TEMPLATE`
  - 标签生成：`EXPLANATION_TAG_PROMPT_TEMPLATE`（位于 `scripts/rebuild_explanation_tags.py`）
- **调用函数**：
  - `generate_explanations_for_date()`：基于已有 `llm_score` 生成解释
  - `rebuild_explanation_tags.py`：为已有解释结果生成结构化标签
- **用途**：为已完成排序的实验结果添加解释性输出，无需重新调用 LLM 排序
- **启用方式**：
  - 直接调用 `generate_explanations_for_date()` 函数
  - 运行 `python scripts/rebuild_explanation_tags.py --models lightgbm`

## 限制与注意事项

### 1. 当前主目标
- 实验主要围绕 `Y_future_5d_return` 组织，其他目标列（如 `Y_next_day_return`）代码虽保留，但未在主线流程中充分测试。

### 2. 回测简化
- 包含基础交易成本（0.1%）、滑点（0.05%）与换手率计算。
- 未考虑流动性冲击、盘口级成交模拟等高级市场摩擦。

### 3. 缓存机制
- LLM 结果缓存基于候选池 fingerprint（SHA256），确保仅当输入完全一致时才复用。
- 不同 scenario（`top_n_first`/`top_n_final`）不会错误命中缓存。

### 4. 实验汇总与单次运行分离
- 单次运行详情查看 `runs/backtests/.../reports/report_summary.html`。
- 跨模型汇总查看 `experiments/summary/experiment_summary.html`。

### 5. 辅助脚本
- `scripts/` 目录下存在若干分析、解释、迁移脚本（如 `analyze_candidate_pool_quality.py`、`explanation_reporter.py`、`migrate_legacy_deepseek_cache.py`），用于特定调试或实验，不属主流程。

## 故障排除

### 1. 第一阶段预测文件缺失
```powershell
python scripts/train_traditional_multi_target_backtest.py --traditional-model lightgbm --target Y_future_5d_return
```

### 2. LLM API 失败
- 检查 `config/llm_config.local.yaml` 中的 API 密钥与 URL。
- 查看 `runs/backtests/.../logs/` 下的 `response_*.txt` 与 `parse_debug_*.txt`。

### 3. scenario 目录不符
- 确认命令中 `--top-n-first` 与 `--top-n-final` 参数是否正确。
- 不同参数组合会写入不同的 scenario 子目录。

### 4. 实验汇总未更新
```powershell
python scripts/experiment_summary.py
```

## 附录

### 常用路径速查
- 合并因子数据：`data/processed/all_etf_factors.csv`
- 第一阶段预测：`data/predictions/<traditional_model>/test_set_with_predictions.csv`
- 两阶段回测结果：`runs/backtests/<traditional_model>/<second_stage_llm>/<scenario>/`
- LLM 日志：`runs/backtests/<traditional_model>/logs/<second_stage_llm>/`
- 实验汇总：`experiments/summary/`

### 常用命令速查
```powershell
# 数据准备
python scripts/download_etf.py
python scripts/compute_etf_factors.py
python scripts/merge_etf_factors.py

# 单模型训练
python scripts/train_traditional_multi_target_backtest.py --traditional-model lightgbm --target Y_future_5d_return

# 两阶段回测（默认 scenario）
python scripts/two_stage_backtest.py --traditional-model lightgbm --target Y_future_5d_return --second-stage-llm deepseek-chat --llm-config config/llm_config.local.yaml

# 实验套件（lightgbm + deepseek）
python scripts/run_experiment_suite.py --baseline-models lightgbm --llm-models lightgbm --llm-config config/llm_config.local.yaml
```
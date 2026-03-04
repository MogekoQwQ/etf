LLM可解释性模块实施计划
上下文
用户询问："如何修改prompt，让它输出结果的同时，附带人能理解的解释?"
当前ETF量化分析系统使用大语言模型(DeepSeek API)对ETF进行横向排序，但prompt明确禁止LLM输出任何额外解释（prompt_template.txt第25行："不得输出任何额外的自然语言解释、评论或发散性内容"）。用户希望增强策略透明度，让模型在给出预测结果的同时生成解释性报告。
问题分析
1. 当前限制：Prompt严格限制输出格式，只允许JSON格式的排序结果
2. 向后兼容需求：现有回测逻辑依赖llm_score列，不能破坏现有功能
3. 成本考虑：解释性内容会增加API调用token使用量
4. 存储需求：解释内容需要合理组织存储
用户偏好澄清
基于用户问答，明确了以下需求：
1. 解释详细程度：标准分析（每只ETF多因子分析，包含优势劣势和风险提示）
2. 成本控制：成本优先（限制解释长度，只对高分ETF生成详细解释，默认关闭解释功能）
3. 报告格式：先生成JSON格式或纯文本，在最后的结果（绩效指标对比阶段）统一生成一份HTML报告
建议方案
基于用户偏好，设计一个平衡的方案：
混合方案：分阶段可解释性系统
1. 数据层：JSON格式存储结构化解释，便于程序处理
2. 处理层：成本优先的生成策略，默认关闭，按需开启
3. 展示层：在绩效指标对比阶段生成统一的HTML报告
核心设计原则
1. 成本控制优先：默认关闭解释功能，需要时通过配置开启
2. 选择性生成：只对高分ETF（score > 0.7）生成详细解释
3. 长度限制：每只ETF解释不超过300字符
4. 双格式输出：JSON存储 + HTML报告展示
5. 向后兼容：现有功能完全不受影响
详细实施计划
第一阶段：Prompt与解析逻辑修改（1-2天）
1.1 创建扩展的prompt模板（成本优化版）
在llm_ranking.py中添加新prompt模板，包含成本控制提示：
PROMPT_TEMPLATE_WITH_EXPLANATION = """
[保留原有任务定义、因子说明、约束条件1-2和4-5]

## 约束条件（修改）
3. 输出格式必须严格遵守下方要求。请为每只ETF提供简洁的分析解释（不超过100字），并添加简要的整体市场分析。

## 特别要求（成本控制）
- 对每只ETF的解释请控制在100字以内
- 重点关注评分前10名的ETF，其他ETF可以简要说明
- 整体分析摘要控制在200字以内
- 使用简洁的语言，避免冗余描述

## 输出格式（扩展）
你必须输出一个合法的JSON对象，格式如下：
{
  "rankings": [
    {
      "code": "ETF代码1",
      "score": 0.95,
      "explanation": "简洁分析：关键优势因子...主要风险点..."
    },
    ...
  ],
  "summary": {
    "market_context": "当前市场环境简要分析...",
    "key_factors": ["动量因子", "波动率", "流动性"],
    "top_etf_count": 10,
    "risk_considerations": "主要风险因素..."
  },
  "metadata": {
    "explanation_length_limit": 100,
    "generation_time": "2025-01-15T10:30:00"
  }
}
"""
1.2 扩展解析逻辑
修改rank_etfs_by_llm函数：
- 添加enable_explanations参数（默认False）
- 支持解析扩展的JSON格式
- 提取解释内容并保存到日志目录
1.3 保持向后兼容
- 默认行为不变（enable_explanations=False）
- 现有代码继续使用llm_score列
第二阶段：解释内容存储与组织（1天）
2.1 存储结构设计
results/two_stage/llm_logs/
├── explanations/
│   ├── 20250115/
│   │   ├── full_response.json      # 完整API响应
│   │   ├── summary.txt             # 摘要文本
│   │   └── etf_details/            # 单只ETF详细解释
│   │       ├── 510300.json
│   │       └── ...
│   └── index.csv                   # 文件索引
2.2 索引管理
创建CSV索引文件，方便检索和汇总：
date,etf_code,score,explanation_file,has_explanation
2025-01-15,510300,0.95,20250115/etf_details/510300.json,true
第三阶段：报告生成与集成（2-3天）
3.1 创建解释报告生成器
新文件：scripts/explanation_reporter.py
功能：
- 生成HTML格式的可读报告
- 提取关键洞察和模式
- 集成到现有报告系统
3.2 双格式输出设计
JSON格式（程序处理）：
{
  "date": "2025-01-15",
  "rankings": [
    {
      "code": "510300",
      "score": 0.95,
      "explanation": {
        "strengths": ["动量强劲", "波动率稳定"],
        "weaknesses": ["换手率偏低"],
        "key_factors": {"momentum_20": 0.85},
        "risk_level": "中等"
      }
    }
  ],
  "summary": {
    "market_analysis": "简要市场分析...",
    "top_factors": ["momentum_20", "volatility_20"],
    "risk_warnings": ["注意高波动率ETF"]
  }
}
HTML报告（用户查看）：
在绩效指标对比阶段统一生成，包含：
1. 报告封面：日期、策略名称、关键指标
2. 执行摘要：调仓日的关键决策和理由
3. 市场分析：LLM对整体市场的判断（带图表）
4. ETF详细分析：前10名ETF的排序理由（表格+文字）
5. 因子重要性：LLM关注的关键因子（条形图）
6. 风险提示：LLM识别的风险因素
7. 策略建议：基于解释的优化建议
文本摘要：
=== ETF可解释性分析报告 ===
日期：2025-01-15

[市场分析]
当前市场动量因子有效性较高...

[前3名ETF分析]
1. 510300 (0.95)
   优势：动量强劲(0.85)，波动率低(0.12)
   风险：估值偏高，换手率偏低
   建议：核心持仓

2. 510500 (0.87)
   ...
3.3 集成到回测流程
在two_stage_backtest.py中添加选项：
# 配置参数
GENERATE_EXPLANATIONS = True  # 是否生成解释
EXPLANATION_DIR = os.path.join(OUTPUT_DIR, "explanations")
第四阶段：配置与优化（1天）
4.1 配置管理（成本优先）
添加配置文件支持，强调成本控制：
class ExplanationConfig:
    # 核心控制（成本优先）
    enable = False  # 默认关闭，需要时手动开启
    enable_by_default = False  # 永远默认关闭，防止意外成本

    # 内容控制
    explanation_level = "standard"  # standard/brief
    max_explanation_length = 300  # 每只ETF最多300字符
    summary_max_length = 500  # 摘要最多500字符

    # 选择性生成（成本优化）
    score_threshold = 0.7  # 只对评分>0.7的ETF生成详细解释
    top_n_detailed = 10  # 最多为前10名生成详细解释
    skip_low_score = True  # 跳过低分ETF

    # 格式控制
    save_json = True  # 保存JSON格式（程序处理）
    generate_html = True  # 生成HTML报告（用户查看）
    save_text_summary = True  # 保存文本摘要

    # 成本跟踪
    enable_cost_tracking = True
    max_tokens_per_call = 8000  # 每次API调用token限制
    estimated_cost_per_call = 0.002  # 估计每次调用成本（美元）
4.2 成本控制
- 限制解释长度
- 选择性生成（只对高分ETF生成详细解释）
- 缓存重复解释
关键文件路径
需要修改的文件
1. e:\ClaudeProject\etf\etf-main\scripts\llm_ranking.py
  - 添加新的prompt模板
  - 扩展解析逻辑
  - 添加解释提取功能
2. e:\ClaudeProject\etf\etf-main\scripts\two_stage_backtest.py
  - 添加解释功能配置
  - 调用解释报告生成
需要创建的新文件
1. e:\ClaudeProject\etf\etf-main\scripts\explanation_reporter.py
  - 报告生成模块
2. e:\ClaudeProject\etf\etf-main\scripts\explanation_utils.py
  - 解释内容处理工具
3. e:\ClaudeProject\etf\etf-main\templates\explanation_report.html
  - HTML报告模板
验证方案
测试步骤
1. 单元测试：
python -m pytest tests/test_llm_explanation.py -v
1. 成本控制测试：
# 测试默认关闭
python scripts/two_stage_backtest.py
# 验证：解释功能未启用，无额外API成本

# 测试按需开启
python scripts/two_stage_backtest.py --enable-explanations --score-threshold 0.7
# 验证：只对高分ETF生成解释，成本可控
1. 格式验证测试：
# 验证JSON格式
python -c "
import json, os
with open('results/two_stage/explanations/20250115/full_response.json') as f:
    data = json.load(f)
print('JSON验证通过' if 'rankings' in data and 'summary' in data else '失败')
"

# 验证HTML报告生成
python scripts/explanation_reporter.py --date 2025-01-15 --format html
# 验证：HTML文件正确生成，包含图表和表格
1. 成本监控验证：
  - 检查日志中的token使用统计
  - 验证解释长度不超过配置限制
  - 确认低分ETF被跳过（score < 0.7）
  - 计算每调仓日的估计API成本
2. 向后兼容验证：
  - 现有回测结果完全一致
  - llm_score列格式和值不变
  - 现有图表和报告正常生成
成功标准
1. ✅ 现有回测功能正常工作
2. ✅ 解释内容正确生成和存储
3. ✅ 报告可读且信息丰富
4. ✅ API成本增加在可控范围内
5. ✅ 用户能够理解LLM的决策逻辑
风险评估与缓解
技术风险
1. API响应格式不一致：LLM可能不遵守输出格式
  - 缓解：添加严格的格式验证和容错处理
2. 存储空间增长：大量解释内容占用存储
  - 缓解：压缩存储、定期清理旧数据
3. 性能影响：额外的解析增加处理时间
  - 缓解：异步处理、批量操作
业务风险
1. 解释质量参差不齐：LLM可能生成不准确解释
  - 缓解：添加置信度评分、人工审核选项
2. 信息过载：用户被过多细节淹没
  - 缓解：提供摘要视图、关键洞察提取
时间估算
- 第一阶段：1-2天（核心功能）
- 第二阶段：1天（存储系统）
- 第三阶段：2-3天（报告生成）
- 第四阶段：1天（配置优化）
- 总计：5-7个工作日
建议的起始点
从**方案1（最小修改）**开始，快速验证可行性：
1. 先修改prompt模板，允许解释字段
2. 扩展解析逻辑支持新格式
3. 测试基本功能
4. 根据反馈决定是否实施完整方案
这提供了最小的改动，同时满足了用户"输出结果的同时附带解释"的核心需求。

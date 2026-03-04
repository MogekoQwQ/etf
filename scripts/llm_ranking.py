"""
大语言模型排序模块
调用 DeepSeek API 对 ETF 因子截面数据进行横向排序
"""

import os
import json
import datetime
import requests
import time
from typing import List, Dict, Any, Optional
import pandas as pd

# ------------------ 配置 ------------------
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")  # 从环境变量读取
if not DEEPSEEK_API_KEY:
    # 也可以从配置文件读取，这里简单硬编码（不推荐，仅示例）
    DEEPSEEK_API_KEY = "sk-e19935e4e0c24900b7992576c3768d29"
    print("警告：请设置环境变量 DEEPSEEK_API_KEY 或修改脚本中的 API_KEY")

MODEL_NAME = "deepseek-chat"  # 或其他可用模型
MAX_RETRY = 3
RETRY_DELAY = 5  # 秒
TIMEOUT = 60  # 请求超时时间（秒）

# ------------------ Prompt 模板 ------------------
PROMPT_TEMPLATE = """你是一个专业的量化投资分析助手，你的任务是对给定的ETF进行横向比较和排序。你将获得某个交易日多只ETF的量化因子数据，请仅依据这些数值进行综合分析，输出排序结果。

## 任务定义
基于提供的ETF因子截面数据，对它们进行横向比较，按照“未来收益潜力与风险调整后收益”的综合评估进行排序。你需要扮演一个纯粹的“横向比较与排序器”，不进行时间序列预测，也不引入任何外部知识或历史经验。排序必须完全基于当前提供的因子数值。

## 因子说明
以下是每个因子的含义，请理解后用于比较：
- momentum_20：20日动量，反映过去20日的价格趋势强度。
- volatility_20：20日波动率，反映过去20日收益率的波动程度，越低代表越稳定。
- volume_mean_20：20日成交量均值，反映近期交易活跃度。
- return_mean_20：20日平均涨跌幅，反映近期平均日收益。
- amplitude_mean_20：20日振幅均值，反映近期价格波动范围。
- turnover_mean_20：20日换手率均值，反映近期筹码换手频率。
- MA_5：5日移动平均线，反映极短期价格趋势。
- MA_10：10日移动平均线，反映短期价格趋势。
- 涨跌幅：当日涨跌幅，反映当日价格变动百分比。
- 振幅：当日振幅，反映当日价格波动范围。
- 换手率：当日换手率，反映当日筹码换手频率。
- 成交量：当日成交量，反映当日交易活跃度。
- 成交额：当日成交额，反映当日交易金额规模。

## 约束条件
1. 禁止联网搜索或使用任何外部知识库，仅依据输入数值进行比较。
2. 禁止进行时间序列预测或趋势外推，仅做横截面比较。
3. 输出格式必须严格遵守下方要求，不得输出任何额外的自然语言解释、评论或发散性内容。
4. 排序应综合考虑收益潜力（如动量、涨跌幅、移动平均线等）与风险控制（如波动率、振幅等），追求风险调整后的收益。
5. 由于ETF数量较多（50只），建议先分组比较：可将50只ETF分为5组，每组10只，先选出每组前2名，再对10只优胜者进行最终排序。

## 输出格式
你必须输出一个合法的JSON对象，且只包含该JSON对象，格式如下：
{{
  "rankings": [
    {{"code": "ETF代码1", "score": 0.95}},
    {{"code": "ETF代码2", "score": 0.87}},
    ...
  ]
}}
其中：
- "rankings" 是一个数组，按综合评分从高到低排列。
- 每个元素包含 "code"（ETF代码，字符串）和 "score"（综合评分，浮点数，范围0~1，评分越高代表综合表现越好）。
- 数组长度必须等于输入ETF的数量。
- 评分应体现相对优劣，可以基于数值归一化或主观加权，但需保持一致性。

## 输入数据
以下是当前交易日（{date}）的ETF因子数据（数值已标准化处理，可直接比较）：
{data}

请严格遵守任务定义、约束条件和输出格式，输出排序JSON对象。"""

# ------------------ API 调用函数 ------------------
def call_deepseek_api(messages: List[Dict[str, str]], max_retry: int = MAX_RETRY) -> Optional[str]:
    """
    调用 DeepSeek API，支持重试机制

    Args:
        messages: OpenAI 格式的消息列表
        max_retry: 最大重试次数

    Returns:
        API 返回的文本内容，失败时返回 None
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,  # 低温度以保证输出确定性
        "max_tokens": 8000
    }

    for attempt in range(1, max_retry + 1):
        try:
            print(f"正在调用 DeepSeek API (尝试 {attempt}/{max_retry})...")
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=TIMEOUT)
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print("API 调用成功")
            return content

        except requests.exceptions.RequestException as e:
            print(f"API 请求失败: {e}")
            if attempt < max_retry:
                print(f"等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
            else:
                print("达到最大重试次数，放弃本次调用")
                return None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"API 响应解析失败: {e}")
            print(f"原始响应: {response.text if 'response' in locals() else '无响应'}")
            return None

    return None

# ------------------ 排序主函数 ------------------
def rank_etfs_by_llm(etf_data: pd.DataFrame, date: str, log_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    使用大语言模型对给定日期的 ETF 数据进行排序

    Args:
        etf_data: DataFrame，包含多只ETF的因子数据，必须包含 'code' 列
        date: 交易日字符串，用于提示词
        log_dir: 日志目录，如果提供则保存prompt、响应和结果

    Returns:
        排序后的 DataFrame，新增 'llm_score' 列，按分数降序排列；失败时返回 None
    """
    # 确保有 code 列
    if "code" not in etf_data.columns:
        print("错误：etf_data 必须包含 'code' 列")
        return None

    # 创建副本并确保 code 列为字符串类型（避免与 LLM 返回的字符串代码不匹配）
    etf_data = etf_data.copy()
    etf_data["code"] = etf_data["code"].astype(str)

    # 准备数据：选择用于排序的因子列（排除代码、名称、日期等）
    exclude_cols = ["code", "name", "日期"]
    # 如果存在目标变量列，也排除
    target_cols = ["Y_next_day_return", "Y_future_5d_return", "Y_future_10d_return",
                   "Y_future_5d_vol_change", "Y_future_10d_vol_change"]
    exclude_cols += [col for col in target_cols if col in etf_data.columns]

    factor_cols = [col for col in etf_data.columns if col not in exclude_cols]

    # 构建数据表格字符串
    data_lines = []
    for _, row in etf_data.iterrows():
        code = row["code"]
        # 只保留因子数值，保留一定精度
        factor_values = {col: round(row[col], 6) for col in factor_cols}
        line = f"ETF代码 {code}: " + ", ".join([f"{k}: {v}" for k, v in factor_values.items()])
        data_lines.append(line)

    data_str = "\n".join(data_lines)

    # 填充提示词模板
    prompt = PROMPT_TEMPLATE.format(date=date, data=data_str)

    # 日志记录：保存 prompt
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_file = os.path.join(log_dir, f"prompt_{timestamp}.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Prompt 保存至 {prompt_file}")

    # 构建 API 消息
    messages = [
        {"role": "system", "content": "你是一个专业的量化投资分析助手，严格遵循用户指令。"},
        {"role": "user", "content": prompt}
    ]

    # 调用 API
    response_text = call_deepseek_api(messages)

    # 日志记录：保存原始响应
    if log_dir and response_text is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        response_file = os.path.join(log_dir, f"response_{timestamp}.txt")
        with open(response_file, "w", encoding="utf-8") as f:
            f.write(response_text)
        print(f"Response 保存至 {response_file}")

    if response_text is None:
        print("API 调用失败，无法获取排序结果")
        return None

    # 解析响应
    try:
        # 尝试提取 JSON 部分（可能响应包含 markdown 代码块）
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
        else:
            json_str = response_text

        result = json.loads(json_str)
        rankings = result.get("rankings", [])

        # 验证 rankings 格式
        if not isinstance(rankings, list):
            raise ValueError("rankings 不是列表")

        # 创建代码到分数的映射，确保代码为字符串类型
        code_to_score = {str(item["code"]): item["score"] for item in rankings if "code" in item and "score" in item}

        # 调试：打印映射结果
        print(f"  LLM返回 {len(code_to_score)} 个ETF的评分")
        if code_to_score:
            sample_codes = list(code_to_score.keys())[:3]
            print(f"    示例映射: {sample_codes} -> {[code_to_score[c] for c in sample_codes]}")

        # 检查是否所有 ETF 都有分数
        missing_codes = set(etf_data["code"]) - set(code_to_score.keys())
        print(f"  ETF数据共 {len(etf_data)} 只，LLM返回 {len(code_to_score)} 只，缺失 {len(missing_codes)} 只")
        if missing_codes:
            print(f"警告：部分 ETF 未在排名中找到: {missing_codes}")
            # 为缺失的代码分配最低分数
            min_score = min(code_to_score.values()) if code_to_score else 0
            for code in missing_codes:
                code_to_score[code] = min_score - 0.01  # 稍低于最低分

        # 添加分数列并排序
        etf_data = etf_data.copy()
        etf_data["llm_score"] = etf_data["code"].map(code_to_score)
        etf_data_sorted = etf_data.sort_values("llm_score", ascending=False).reset_index(drop=True)

        # 日志记录：保存解析结果
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(log_dir, f"result_{timestamp}.csv")
            etf_data_sorted[["code", "name", "llm_score"] + factor_cols].to_csv(result_file, index=False)
            print(f"解析结果保存至 {result_file}")

        return etf_data_sorted

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"解析 API 响应失败: {e}")
        print(f"原始响应文本:\n{response_text}")
        return None

# ------------------ 批量处理函数 ------------------
def process_rebalancing_dates(data_path: str, output_dir: str,
                              target: str = "Y_future_5d_return",
                              top_n_first: int = 50,
                              top_n_final: int = 10,
                              rebalancing_freq: str = "W",
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None):
    """
    处理整个回测期的调仓日，执行两阶段排序

    Args:
        data_path: 因子数据文件路径
        output_dir: 输出目录
        target: 预测目标列名
        top_n_first: 第一阶段选取的 ETF 数量
        top_n_final: 最终组合的 ETF 数量
        rebalancing_freq: 调仓频率，'D' 每日，'W' 每周，'M' 每月
        start_date: 开始日期（字符串）
        end_date: 结束日期（字符串）

    Returns:
        包含每次调仓结果的 DataFrame
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    print(f"读取数据从 {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["日期"])

    # 过滤日期范围
    if start_date:
        df = df[df["日期"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["日期"] <= pd.to_datetime(end_date)]

    # 确保有预测列
    pred_col = f"y_pred_{target}"
    if pred_col not in df.columns:
        print(f"警告：数据中没有预测列 {pred_col}，需要先运行传统模型预测")
        # 这里可以添加预测逻辑，但为了简化，假设已有预测列
        return None

    # 获取唯一的调仓日期
    dates = df["日期"].unique()
    dates = pd.Series(dates).sort_values().reset_index(drop=True)

    # 按频率筛选调仓日
    if rebalancing_freq == "W":
        # 每周一次（取每周最后一个交易日）
        rebalance_dates = dates[dates.isin(dates + pd.offsets.Week(weekday=4))]  # 周五
    elif rebalancing_freq == "M":
        # 每月一次（取每月最后一个交易日）
        rebalance_dates = dates.groupby(pd.Grouper(key="日期", freq="M")).last()
    else:
        # 每日
        rebalance_dates = dates

    rebalance_dates = pd.Series(rebalance_dates).dropna().unique()
    rebalance_dates = pd.to_datetime(rebalance_dates)

    print(f"共 {len(rebalance_dates)} 个调仓日")

    # 存储每次调仓结果
    all_results = []

    for i, date in enumerate(rebalance_dates):
        print(f"\n处理调仓日 {i+1}/{len(rebalance_dates)}: {date.date()}")

        # 获取当日所有 ETF 数据
        day_data = df[df["日期"] == date].copy()

        if len(day_data) < top_n_first:
            print(f"  警告：当日只有 {len(day_data)} 只 ETF，小于第一阶段所需 {top_n_first}，跳过")
            continue

        # 第一阶段：按传统模型预测排序，选 top_n_first
        day_data_sorted = day_data.sort_values(pred_col, ascending=False).reset_index(drop=True)
        top_etfs = day_data_sorted.head(top_n_first).copy()

        # 第二阶段：大语言模型排序
        ranked_etfs = rank_etfs_by_llm(top_etfs, str(date.date()))

        if ranked_etfs is None:
            print(f"  大语言模型排序失败，使用传统模型排序结果")
            ranked_etfs = top_etfs.copy()
            ranked_etfs["llm_score"] = ranked_etfs[pred_col]  # 用预测值作为分数

        # 选取最终组合
        final_etfs = ranked_etfs.head(top_n_final)

        # 保存本次调仓结果
        result = {
            "rebalance_date": date,
            "traditional_top_codes": list(top_etfs["code"]),
            "traditional_top_scores": list(top_etfs[pred_col]),
            "llm_ranked_codes": list(ranked_etfs["code"]),
            "llm_scores": list(ranked_etfs["llm_score"]),
            "final_codes": list(final_etfs["code"]),
            "final_scores": list(final_etfs["llm_score"]),
            "target_values": list(final_etfs[target]) if target in final_etfs.columns else []
        }
        all_results.append(result)

        # 保存详细结果到文件
        result_df = pd.DataFrame({
            "date": date,
            "code": final_etfs["code"],
            "name": final_etfs["name"] if "name" in final_etfs.columns else "",
            "llm_score": final_etfs["llm_score"],
            "traditional_score": final_etfs[pred_col],
            "target_value": final_etfs[target] if target in final_etfs.columns else None
        })

        result_file = os.path.join(output_dir, f"rebalance_{date.strftime('%Y%m%d')}.csv")
        result_df.to_csv(result_file, index=False)
        print(f"  结果保存至 {result_file}")

    # 汇总所有调仓结果
    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(output_dir, "rebalancing_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n所有调仓结果汇总保存至 {summary_file}")

    return summary_df

# ------------------ 主函数 ------------------
if __name__ == "__main__":
    # 示例用法
    data_path = "../data/all_etf_factors.csv"
    output_dir = "../results/llm_ranking"

    # 注意：需要先运行传统模型预测，生成 y_pred_* 列
    # 这里假设数据已经包含预测列

    process_rebalancing_dates(
        data_path=data_path,
        output_dir=output_dir,
        target="Y_future_5d_return",
        top_n_first=50,  # 实际ETF数量100只，第一阶段选50只
        top_n_final=10,
        rebalancing_freq="W",
        start_date="2023-01-01",
        end_date="2024-12-31"
    )
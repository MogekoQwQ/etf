"""
LLM ranking utilities for ETF cross-sectional sorting.
"""

import datetime
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = "sk-e19935e4e0c24900b7992576c3768d29"
    print("警告：请设置环境变量 DEEPSEEK_API_KEY，当前仍使用脚本中的默认值。")

MODEL_NAME = "deepseek-chat"
MAX_RETRY = 3
RETRY_DELAY = 5
DEFAULT_TIMEOUT = 60
MIN_TIMEOUT = 60
MAX_TIMEOUT = 300
RANKING_MAX_COMPLETION_TOKENS = 1200
EXPLANATION_MAX_COMPLETION_TOKENS = 2400


RANKING_PROMPT_TEMPLATE = """你是一名专业的量化投资分析助手。
任务：基于同一交易日的 ETF 横截面因子数据，对候选 ETF 做排序。

要求：
1. 只能依据输入的数值型信息比较，不得引入外部知识。
2. 不做时间序列外推，只做同日横截面排序。
3. 综合考虑收益潜力与风险约束。
4. 输出必须是合法 JSON，且只能输出 JSON。
5. 不要输出任何自然语言解释。

输出格式：
{{
  "rankings": [
    {{"code": "ETF代码", "score": 0.95}}
  ]
}}

当前交易日：{date}
ETF 因子数据：
{data}
"""


EXPLANATION_PROMPT_TEMPLATE = """你是一名专业的量化投资分析助手。
任务：对同一交易日已经完成排序的少量 ETF 生成展示性解释。

要求：
1. 只能依据输入的因子、排序分数和横截面对比信息说明，不得引入外部知识。
2. 不做时间序列预测，只解释当前已给出的排序结果。
3. explanation 需要简洁，聚焦排序依据与风险提示。
4. 输出必须是合法 JSON，且只能输出 JSON。

输出格式：
{{
  "rankings": [
    {{"code": "ETF代码", "score": 0.95, "explanation": "简洁解释"}}
  ],
  "summary": {{
    "market_context": "整体说明",
    "key_factors": ["因子1", "因子2"],
    "risk_considerations": "风险提示"
  }}
}}

当前交易日：{date}
待解释 ETF 数据：
{data}
"""


def estimate_deepseek_timeout(
    messages: List[Dict[str, str]],
    max_tokens: int,
    etf_count: Optional[int] = None,
    mode: str = "ranking",
) -> int:
    """
    Estimate a practical timeout based on request size.
    """
    total_chars = sum(len(message.get("content", "")) for message in messages)
    estimated_input_tokens = max(1, total_chars // 4)

    if etf_count is None:
        etf_count = max(1, total_chars // 300)

    if mode == "explanation":
        per_etf_output_tokens = 180
        summary_tokens = 420
    else:
        per_etf_output_tokens = 35
        summary_tokens = 120

    estimated_output_tokens = min(max_tokens, etf_count * per_etf_output_tokens + summary_tokens)

    estimated_seconds = DEFAULT_TIMEOUT
    estimated_seconds += estimated_input_tokens / 80.0
    estimated_seconds += estimated_output_tokens / 50.0

    return max(MIN_TIMEOUT, min(MAX_TIMEOUT, int(round(estimated_seconds))))


def call_deepseek_api(
    messages: List[Dict[str, str]],
    max_retry: int = MAX_RETRY,
    timeout: Optional[int] = None,
    max_tokens: int = RANKING_MAX_COMPLETION_TOKENS,
    etf_count: Optional[int] = None,
    mode: str = "ranking",
) -> Optional[str]:
    """
    Call the DeepSeek API with retries.
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    if timeout is None:
        effective_timeout = estimate_deepseek_timeout(
            messages=messages,
            max_tokens=max_tokens,
            etf_count=etf_count,
            mode=mode,
        )
        print(
            f"DeepSeek timeout estimate: {effective_timeout}s "
            f"(mode={mode}, chars={sum(len(message.get('content', '')) for message in messages)})"
        )
    else:
        effective_timeout = timeout
        print(f"DeepSeek timeout override: {effective_timeout}s")

    for attempt in range(1, max_retry + 1):
        try:
            print(f"正在调用 DeepSeek API ({mode}, 尝试 {attempt}/{max_retry})...")
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=effective_timeout,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print("API 调用成功")
            return content
        except requests.exceptions.RequestException as exc:
            print(f"API 请求失败: {exc}")
            if attempt < max_retry:
                print(f"等待 {RETRY_DELAY} 秒后重试。")
                time.sleep(RETRY_DELAY)
            else:
                print("达到最大重试次数，放弃本次调用")
                return None
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"API 响应解析失败: {exc}")
            print(f"原始响应: {response.text if 'response' in locals() else '无响应'}")
            return None

    return None


def _normalized_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return pd.Series(0.5, index=series.index, dtype=float)

    ranked = numeric.rank(method="average", pct=True, ascending=not higher_is_better)
    return ranked.fillna(0.5).astype(float)


def _extract_factor_columns(etf_data: pd.DataFrame) -> List[str]:
    exclude_cols = ["code", "name", "日期", "llm_score", "llm_explanation"]
    target_cols = [
        "Y_next_day_return",
        "Y_future_5d_return",
        "Y_future_10d_return",
        "Y_future_5d_vol_change",
        "Y_future_10d_vol_change",
    ]
    exclude_cols.extend([col for col in target_cols if col in etf_data.columns])
    return [col for col in etf_data.columns if col not in exclude_cols]


def _serialize_etf_payload(
    etf_data: pd.DataFrame,
    factor_cols: List[str],
    include_llm_score: bool = False,
) -> str:
    data_lines: List[str] = []
    for _, row in etf_data.iterrows():
        parts = []
        if include_llm_score and "llm_score" in row.index:
            parts.append(f"llm_score: {round(float(row['llm_score']), 6)}")

        for col in factor_cols:
            value = row[col]
            if isinstance(value, (float, np.floating, int, np.integer)):
                display_value = round(float(value), 6)
            else:
                display_value = value
            parts.append(f"{col}: {display_value}")

        data_lines.append(f"ETF代码 {row['code']}: " + ", ".join(parts))
    return "\n".join(data_lines)


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    json_str = json_match.group(0) if json_match else response_text
    return json.loads(json_str)


def _build_mock_llm_result(
    etf_data: pd.DataFrame,
    score_reference_col: Optional[str] = None,
) -> Dict[str, Any]:
    df = etf_data.copy()
    score_parts: List[pd.Series] = []

    if score_reference_col and score_reference_col in df.columns:
        score_parts.append(_normalized_rank(df[score_reference_col], higher_is_better=True) * 0.45)

    weighted_factors = [
        ("momentum_20", 0.20, True),
        ("return_mean_20", 0.15, True),
        ("MA_5", 0.05, True),
        ("MA_10", 0.05, True),
        ("volatility_20", 0.10, False),
        ("amplitude_mean_20", 0.05, False),
    ]

    for column, weight, higher_is_better in weighted_factors:
        if column in df.columns:
            score_parts.append(_normalized_rank(df[column], higher_is_better=higher_is_better) * weight)

    if score_parts:
        combined_score = sum(score_parts)
    else:
        combined_score = pd.Series(np.linspace(1.0, 0.0, len(df)), index=df.index, dtype=float)

    min_score = float(combined_score.min())
    max_score = float(combined_score.max())
    if max_score > min_score:
        normalized_score = (combined_score - min_score) / (max_score - min_score)
    else:
        normalized_score = pd.Series(0.5, index=df.index, dtype=float)

    rankings: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        rankings.append(
            {
                "code": str(row["code"]),
                "score": round(float(normalized_score.loc[idx]), 6),
            }
        )

    rankings.sort(key=lambda item: item["score"], reverse=True)
    return {"rankings": rankings}


def _build_mock_explanation_result(etf_data: pd.DataFrame) -> Dict[str, Any]:
    df = etf_data.copy().reset_index(drop=True)
    rankings: List[Dict[str, Any]] = []
    key_factors: List[str] = []

    if "momentum_20" in df.columns:
        key_factors.append("动量因子")
    if "volatility_20" in df.columns:
        key_factors.append("波动率约束")
    if "return_mean_20" in df.columns:
        key_factors.append("收益均值")
    if "llm_score" in df.columns:
        key_factors.append("排序分数")

    for _, row in df.iterrows():
        explanation_parts = []
        if "llm_score" in df.columns:
            explanation_parts.append(f"排序分数为 {float(row['llm_score']):.3f}")
        if "momentum_20" in df.columns:
            explanation_parts.append("动量因子参与比较")
        if "volatility_20" in df.columns:
            explanation_parts.append("同时考虑波动率约束")

        rankings.append(
            {
                "code": str(row["code"]),
                "score": round(float(row.get("llm_score", 0.0)), 6),
                "explanation": "模拟解释：" + "，".join(explanation_parts[:3]) + "。",
            }
        )

    return {
        "rankings": rankings,
        "summary": {
            "market_context": "该解释由本地 mock 逻辑生成，仅用于展示和链路验证。",
            "key_factors": key_factors[:5] or ["排序分数"],
            "risk_considerations": "mock 解释不代表真实大语言模型判断，仅用于离线验证。",
        },
    }


def _save_text_artifact(log_dir: Optional[str], filename: str, content: str) -> None:
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"文件已保存至 {path}")


def _save_ranking_artifacts(
    result: Dict[str, Any],
    response_text: str,
    etf_data: pd.DataFrame,
    factor_cols: List[str],
    log_dir: Optional[str],
) -> pd.DataFrame:
    rankings = result.get("rankings", [])
    if not isinstance(rankings, list):
        raise ValueError("rankings 不是列表")

    code_to_score: Dict[str, float] = {}
    for item in rankings:
        if "code" in item and "score" in item:
            code_to_score[str(item["code"])] = float(item["score"])

    print(f"  LLM 返回 {len(code_to_score)} 只 ETF 的评分")

    missing_codes = set(etf_data["code"].astype(str)) - set(code_to_score.keys())
    if missing_codes:
        print(f"警告：部分 ETF 未在排名中找到，将补最低分: {missing_codes}")
        min_score = min(code_to_score.values()) if code_to_score else 0.0
        for code in missing_codes:
            code_to_score[code] = min_score - 0.01

    ranked = etf_data.copy()
    ranked["code"] = ranked["code"].astype(str)
    ranked["llm_score"] = ranked["code"].map(code_to_score)
    ranked = ranked.sort_values("llm_score", ascending=False).reset_index(drop=True)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_text_artifact(log_dir, f"response_{timestamp}.txt", response_text)

        result_file = os.path.join(log_dir, f"result_{timestamp}.csv")
        cols_to_save = ["code", "name", "llm_score"]
        cols_to_save.extend([col for col in factor_cols if col not in cols_to_save])
        ranked[cols_to_save].to_csv(result_file, index=False)
        print(f"解析结果保存至 {result_file}")

    return ranked


def _merge_explanations_into_dataframe(
    result: Dict[str, Any],
    etf_data: pd.DataFrame,
) -> pd.DataFrame:
    rankings = result.get("rankings", [])
    if not isinstance(rankings, list):
        raise ValueError("rankings 不是列表")

    code_to_explanation: Dict[str, str] = {}
    code_to_score: Dict[str, float] = {}

    for item in rankings:
        code = str(item.get("code", ""))
        if not code:
            continue
        code_to_explanation[code] = str(item.get("explanation", "") or "")
        if "score" in item:
            code_to_score[code] = float(item["score"])

    explained = etf_data.copy()
    explained["code"] = explained["code"].astype(str)
    explained["llm_explanation"] = explained["code"].map(code_to_explanation).fillna("")

    if "llm_score" not in explained.columns and code_to_score:
        explained["llm_score"] = explained["code"].map(code_to_score)

    return explained


def _save_explanation_artifacts(
    result: Dict[str, Any],
    response_text: str,
    etf_data: pd.DataFrame,
    log_dir: Optional[str],
    date: str,
) -> None:
    if not log_dir:
        return

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_text_artifact(log_dir, f"explanation_response_{timestamp}.txt", response_text)

    explanation_file = os.path.join(log_dir, f"explanation_{timestamp}.json")
    with open(explanation_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"完整解释响应保存至 {explanation_file}")

    from explanation_utils import ExplanationStorage

    storage = ExplanationStorage(log_dir)
    storage.save_explanation(
        date=date,
        result=result,
        etf_data=etf_data,
        enable_explanations=True,
    )


def rank_etfs_by_llm(
    etf_data: pd.DataFrame,
    date: str,
    log_dir: Optional[str] = None,
    enable_explanations: bool = False,
    mock: bool = False,
    score_reference_col: Optional[str] = None,
    api_timeout: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Use the LLM to rank ETFs for a single rebalancing date.
    """
    if "code" not in etf_data.columns:
        print("错误：etf_data 必须包含 'code' 列")
        return None

    if enable_explanations:
        print("提示：排序调用已与解释解耦，本次排序会忽略 enable_explanations。")

    ranked_input = etf_data.copy()
    ranked_input["code"] = ranked_input["code"].astype(str)

    factor_cols = _extract_factor_columns(ranked_input)
    data_str = _serialize_etf_payload(ranked_input, factor_cols)
    prompt = RANKING_PROMPT_TEMPLATE.format(date=date, data=data_str)

    if log_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_text_artifact(log_dir, f"prompt_{timestamp}.txt", prompt)

    if mock:
        print("使用本地模拟 LLM 排序，不调用外部 API")
        result = _build_mock_llm_result(
            etf_data=ranked_input,
            score_reference_col=score_reference_col,
        )
        response_text = json.dumps(result, ensure_ascii=False, indent=2)
        return _save_ranking_artifacts(
            result=result,
            response_text=response_text,
            etf_data=ranked_input,
            factor_cols=factor_cols,
            log_dir=log_dir,
        )

    messages = [
        {"role": "system", "content": "你是一名专业的量化投资分析助手，严格遵循用户要求并只返回 JSON。"},
        {"role": "user", "content": prompt},
    ]

    response_text = call_deepseek_api(
        messages=messages,
        timeout=api_timeout,
        max_tokens=RANKING_MAX_COMPLETION_TOKENS,
        etf_count=len(ranked_input),
        mode="ranking",
    )
    if response_text is None:
        print("API 调用失败，无法获取排序结果")
        return None

    try:
        result = _parse_json_response(response_text)
        return _save_ranking_artifacts(
            result=result,
            response_text=response_text,
            etf_data=ranked_input,
            factor_cols=factor_cols,
            log_dir=log_dir,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"解析 API 响应失败: {exc}")
        print(f"原始响应文本:\n{response_text}")
        return None


def generate_explanations_for_date(
    etf_data: pd.DataFrame,
    date: str,
    log_dir: Optional[str] = None,
    mock: bool = False,
    api_timeout: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Generate display-only explanations for a single date.
    """
    if etf_data.empty:
        print("提示：没有可解释的 ETF 数据，跳过解释生成。")
        return None

    explained_input = etf_data.copy().reset_index(drop=True)
    explained_input["code"] = explained_input["code"].astype(str)

    if "llm_score" not in explained_input.columns:
        print("警告：解释输入缺少 llm_score，无法生成解释。")
        return None

    factor_cols = _extract_factor_columns(explained_input)
    data_str = _serialize_etf_payload(explained_input, factor_cols, include_llm_score=True)
    prompt = EXPLANATION_PROMPT_TEMPLATE.format(date=date, data=data_str)

    if log_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_text_artifact(log_dir, f"explanation_prompt_{timestamp}.txt", prompt)

    if mock:
        print("使用本地模拟解释逻辑，不调用外部 API")
        result = _build_mock_explanation_result(explained_input)
        response_text = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        messages = [
            {"role": "system", "content": "你是一名专业的量化投资分析助手，严格遵循用户要求并只返回 JSON。"},
            {"role": "user", "content": prompt},
        ]
        response_text = call_deepseek_api(
            messages=messages,
            timeout=api_timeout,
            max_tokens=EXPLANATION_MAX_COMPLETION_TOKENS,
            etf_count=len(explained_input),
            mode="explanation",
        )
        if response_text is None:
            print("API 调用失败，无法获取解释结果")
            return None

        try:
            result = _parse_json_response(response_text)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"解析解释响应失败: {exc}")
            print(f"原始响应文本:\n{response_text}")
            return None

    explained_output = _merge_explanations_into_dataframe(result, explained_input)

    try:
        _save_explanation_artifacts(
            result=result,
            response_text=response_text,
            etf_data=explained_output,
            log_dir=log_dir,
            date=date,
        )
    except Exception as exc:
        print(f"警告：保存解释结果时出错: {exc}")
        return None

    return explained_output


def process_rebalancing_dates(
    data_path: str,
    output_dir: str,
    target: str = "Y_future_5d_return",
    top_n_first: int = 50,
    top_n_final: int = 10,
    rebalancing_freq: str = "W",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    enable_explanations: bool = False,
    mock: bool = False,
    api_timeout: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Batch-process rebalancing dates for LLM ranking.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"读取数据自 {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["日期"])

    if start_date:
        df = df[df["日期"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["日期"] <= pd.to_datetime(end_date)]

    pred_col = f"y_pred_{target}"
    if pred_col not in df.columns:
        print(f"警告：数据中没有预测列 {pred_col}，请先运行传统模型预测")
        return None

    dates = pd.Series(pd.to_datetime(df["日期"].unique())).sort_values().reset_index(drop=True)
    if rebalancing_freq == "W":
        rebalance_dates = dates.groupby(dates.dt.to_period("W-FRI")).max()
    elif rebalancing_freq == "M":
        rebalance_dates = dates.groupby(dates.dt.to_period("M")).max()
    else:
        rebalance_dates = dates

    rebalance_dates = pd.to_datetime(pd.Series(rebalance_dates).dropna().unique())
    print(f"共 {len(rebalance_dates)} 个调仓日")

    all_results: List[Dict[str, Any]] = []

    for i, date in enumerate(rebalance_dates):
        print(f"\n处理调仓日 {i + 1}/{len(rebalance_dates)}: {date.date()}")
        day_data = df[df["日期"] == date].copy()

        if len(day_data) < top_n_first:
            print(f"  警告：当日仅有 {len(day_data)} 只 ETF，小于第一阶段所需 {top_n_first}，跳过")
            continue

        day_data_sorted = day_data.sort_values(pred_col, ascending=False).reset_index(drop=True)
        top_etfs = day_data_sorted.head(top_n_first).copy()

        ranked_etfs = rank_etfs_by_llm(
            top_etfs,
            str(date.date()),
            log_dir=output_dir,
            enable_explanations=False,
            mock=mock,
            score_reference_col=pred_col,
            api_timeout=api_timeout,
        )

        if ranked_etfs is None:
            print("  LLM 排序失败，使用传统模型排序结果")
            ranked_etfs = top_etfs.copy()
            ranked_etfs["llm_score"] = ranked_etfs[pred_col]

        final_etfs = ranked_etfs.head(top_n_final).copy()

        if enable_explanations:
            explanation_result = generate_explanations_for_date(
                final_etfs,
                str(date.date()),
                log_dir=output_dir,
                mock=mock,
                api_timeout=api_timeout,
            )
            print(f"  解释生成状态: {'成功' if explanation_result is not None else '失败'}")

        result = {
            "rebalance_date": date,
            "traditional_top_codes": list(top_etfs["code"]),
            "traditional_top_scores": list(top_etfs[pred_col]),
            "llm_ranked_codes": list(ranked_etfs["code"]),
            "llm_scores": list(ranked_etfs["llm_score"]),
            "final_codes": list(final_etfs["code"]),
            "final_scores": list(final_etfs["llm_score"]),
            "target_values": list(final_etfs[target]) if target in final_etfs.columns else [],
        }
        all_results.append(result)

        result_df = pd.DataFrame(
            {
                "date": date,
                "code": final_etfs["code"],
                "name": final_etfs["name"] if "name" in final_etfs.columns else "",
                "llm_score": final_etfs["llm_score"],
                "traditional_score": final_etfs[pred_col],
                "target_value": final_etfs[target] if target in final_etfs.columns else None,
            }
        )

        result_file = os.path.join(output_dir, f"rebalance_{date.strftime('%Y%m%d')}.csv")
        result_df.to_csv(result_file, index=False)
        print(f"  结果保存至 {result_file}")

    summary_df = pd.DataFrame(all_results)
    summary_file = os.path.join(output_dir, "rebalancing_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n所有调仓结果汇总保存至 {summary_file}")

    return summary_df


if __name__ == "__main__":
    process_rebalancing_dates(
        data_path="../data/all_etf_factors.csv",
        output_dir="../results/traditional_models/random_forest/two_stage/llm_logs",
        target="Y_future_5d_return",
        top_n_first=50,
        top_n_final=10,
        rebalancing_freq="W",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )

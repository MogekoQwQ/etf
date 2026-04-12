"""Rebuild structured explanation tags for saved two-stage selections."""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from llm_ranking import _parse_json_response, call_llm_api
from traditional_model_config import (
    PLANNED_TRADITIONAL_MODELS,
    get_llm_config_local_file,
    get_traditional_model_label,
    get_traditional_prediction_file,
    get_traditional_two_stage_output_dir,
    normalize_second_stage_llm_name,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_TARGET = "Y_future_5d_return"
DEFAULT_TOP_N_FIRST = 50
DEFAULT_TOP_N_FINAL = 10
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "analysis", "explanation_tags")

DATE_COLUMN_CANDIDATES = ("date", "trade_date", "日期")
ASSET_COLUMN_CANDIDATES = ("etf_code", "code", "symbol")
PREDICTION_COLUMN_CANDIDATES = ("prediction", "y_pred", "pred")
BACKTEST_CODE_COLUMN = "strategy_b_codes"

SIGNAL_TAGS = [
    "动量",
    "波动",
    "流动性",
    "趋势",
    "均线",
    "成交活跃度",
    "收益稳定性",
    "相对强弱",
    "风险收益权衡",
]
RISK_TAGS = [
    "高波动",
    "流动性不足",
    "短期过热",
    "趋势不稳",
    "收益回撤风险",
    "信号分化不足",
]
STYLE_TAGS = [
    "动量优先",
    "风险控制优先",
    "流动性优先",
    "趋势确认优先",
    "多因子平衡",
]
CONFIDENCE_TAGS = ["高", "中", "低"]

EXPLANATION_TAG_PROMPT_TEMPLATE = """你是一个ETF两阶段排序结果解释标签生成器。

任务目标不是重新排序，也不是修改最终买入结果，而是对已经固定的最终Top{top_n_final}结果进行“解释性复现”。

你将得到：
1. 当日第一阶段Top{top_n_first}候选池的结构化摘要；
2. 已固定的最终Top{top_n_final}入选代码；
3. 第二阶段排序的基本约束。

请根据这些信息，判断该次最终选择更可能体现了哪些“信息类别标签”和“风险标签”，并给出一个整体风格标签和置信度标签。

严格要求：
1. 最终Top{top_n_final}结果已经固定，你不能改动结果，也不能补充新的ETF代码。
2. 只能从给定标签集合中选择，不允许自造标签。
3. 输出必须是合法JSON。
4. 不要输出markdown代码块，不要输出任何JSON之外的文字。
5. 即使信息不充分，也必须从给定标签中选出最合理的标签，不要留空。

标签集合：
- signal_tags 可多选：{signal_tags}
- risk_tags 可多选：{risk_tags}
- style_tag 单选：{style_tags}
- confidence_tag 单选：{confidence_tags}

输出JSON结构固定为：
{{
  "signal_tags": ["动量", "趋势"],
  "risk_tags": ["高波动"],
  "style_tag": "多因子平衡",
  "confidence_tag": "中"
}}

第二阶段解释约束：
- 仅解释“为什么最终选择这{top_n_final}只”，不重新做买卖决策。
- 应结合候选池内部的相对比较，而不是脱离候选池单独评价。
- 标签应尽量反映排序时可能关注的高频信息类别，而非编造不存在的因果机制。

输入数据：
{payload_json}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="补生成ETF两阶段排序的结构化解释标签。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(PLANNED_TRADITIONAL_MODELS),
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="需要处理的第一阶段模型列表。",
    )
    parser.add_argument(
        "--llms",
        nargs="+",
        default=["deepseek-chat"],
        help="需要处理的第二阶段LLM列表，例如 deepseek-chat gemini-2.5-flash-lite。",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["default"],
        help="需要处理的scenario列表。",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="目标列名称。当前默认且建议使用 Y_future_5d_return。",
    )
    parser.add_argument(
        "--top-n-first",
        type=int,
        default=DEFAULT_TOP_N_FIRST,
        help="保留兼容参数；实际以scenario目录对应的候选池口径为准。",
    )
    parser.add_argument(
        "--top-n-final",
        type=int,
        default=DEFAULT_TOP_N_FINAL,
        help="保留兼容参数；实际以scenario目录对应的最终持仓口径为准。",
    )
    parser.add_argument(
        "--llm-config",
        default=get_llm_config_local_file(PROJECT_ROOT),
        help="LLM配置文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="单日期标签生成的最大额外重试次数。",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=None,
        help="LLM接口超时时间（秒）。",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("-", "_")


def resolve_exact_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized_map = {normalize_text(column): column for column in columns}
    for candidate in candidates:
        matched = normalized_map.get(normalize_text(candidate))
        if matched:
            return matched
    return None


def infer_prediction_column(columns: List[str], target: str) -> Optional[str]:
    exact_candidates = (
        f"y_pred_{target}",
        f"prediction_{target}",
        f"pred_{target}",
        f"{target}_prediction",
        f"{target}_pred",
    )
    exact_match = resolve_exact_column(columns, exact_candidates)
    if exact_match:
        return exact_match

    normalized_target = normalize_text(target)
    compatible_columns = []
    for column in columns:
        normalized_column = normalize_text(column)
        if normalized_target in normalized_column and (
            normalized_column.startswith("y_pred")
            or normalized_column.startswith("prediction")
            or normalized_column.startswith("pred")
        ):
            compatible_columns.append(column)
    if len(compatible_columns) == 1:
        return compatible_columns[0]
    if len(compatible_columns) > 1:
        raise ValueError(f"Ambiguous prediction columns for `{target}`: {compatible_columns}")

    return resolve_exact_column(columns, PREDICTION_COLUMN_CANDIDATES)


def infer_prediction_columns(df: pd.DataFrame, target: str) -> Dict[str, str]:
    columns = list(df.columns)
    date_column = resolve_exact_column(columns, DATE_COLUMN_CANDIDATES)
    asset_column = resolve_exact_column(columns, ASSET_COLUMN_CANDIDATES)
    target_column = resolve_exact_column(columns, [target])
    prediction_column = infer_prediction_column(columns, target)

    if not date_column:
        raise ValueError(f"Failed to infer date column from {DATE_COLUMN_CANDIDATES}.")
    if not asset_column:
        raise ValueError(f"Failed to infer asset identifier column from {ASSET_COLUMN_CANDIDATES}.")
    if not target_column:
        raise ValueError(f"Required target column is missing: `{target}`")
    if not prediction_column:
        raise ValueError("Failed to infer prediction column for explanation-tag rebuild.")

    return {
        "date": date_column,
        "asset": asset_column,
        "target": target_column,
        "prediction": prediction_column,
    }


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def parse_scenario_top_n(scenario: str) -> Tuple[int, int]:
    if scenario == "default":
        return DEFAULT_TOP_N_FIRST, DEFAULT_TOP_N_FINAL
    match = re.fullmatch(r"top(\d+)_top(\d+)", str(scenario or "").strip().lower())
    if not match:
        raise ValueError(
            f"Unsupported scenario naming: {scenario}. Expected `default` or `top{{n}}_top{{m}}`."
        )
    return int(match.group(1)), int(match.group(2))


def load_prediction_df(model_name: str) -> Tuple[pd.DataFrame, str, Dict[str, str]]:
    prediction_path = get_traditional_prediction_file(PROJECT_ROOT, model_name)
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Prediction file not found for `{model_name}`: {prediction_path}")

    df = pd.read_csv(prediction_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Prediction file is empty for `{model_name}`: {prediction_path}")

    column_names = infer_prediction_columns(df, DEFAULT_TARGET)
    df[column_names["date"]] = pd.to_datetime(df[column_names["date"]], errors="coerce")
    df = df.dropna(subset=[column_names["date"], column_names["prediction"], column_names["target"]]).copy()
    if df.empty:
        raise ValueError(f"No valid prediction rows remain for `{model_name}` after dropping missing values.")

    df[column_names["date"]] = df[column_names["date"]].dt.normalize()
    df[column_names["asset"]] = df[column_names["asset"]].astype(str)
    return df, prediction_path, column_names


def get_run_input_dir(model_name: str, llm_name: str, scenario: str) -> str:
    top_n_first, top_n_final = parse_scenario_top_n(scenario)
    if scenario == "default":
        return get_traditional_two_stage_output_dir(
            PROJECT_ROOT,
            model_name,
            llm_name,
            top_n_first,
            top_n_final,
        )
    return os.path.join(
        PROJECT_ROOT,
        "runs",
        "backtests",
        model_name,
        normalize_second_stage_llm_name(llm_name),
        scenario,
    )


def load_backtest_selection_df(model_name: str, llm_name: str, scenario: str) -> Tuple[pd.DataFrame, str]:
    run_dir = get_run_input_dir(model_name, llm_name, scenario)
    backtest_path = os.path.join(run_dir, "backtest_results.csv")
    if not os.path.exists(backtest_path):
        raise FileNotFoundError(
            f"Saved backtest result not found for model={model_name}, llm={llm_name}, scenario={scenario}: {backtest_path}"
        )

    df = pd.read_csv(backtest_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Backtest result is empty: {backtest_path}")
    if "rebalance_date" not in df.columns:
        raise ValueError(f"`rebalance_date` column is missing: {backtest_path}")
    if BACKTEST_CODE_COLUMN not in df.columns:
        raise ValueError(f"`{BACKTEST_CODE_COLUMN}` column is missing: {backtest_path}")

    df["rebalance_date"] = pd.to_datetime(df["rebalance_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["rebalance_date"]).copy()
    return df, backtest_path


def parse_code_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "momentum_20",
        "volatility_20",
        "volume_mean_20",
        "return_mean_20",
        "amplitude_mean_20",
        "turnover_mean_20",
        "MA_5",
        "MA_10",
    ]
    return [column for column in preferred if column in df.columns]


def round_number(value: Any, digits: int = 6) -> Optional[float]:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def compute_feature_summary(top50_df: pd.DataFrame, selected_df: pd.DataFrame, feature_columns: Sequence[str]) -> Dict[str, Any]:
    feature_summary: Dict[str, Any] = {}
    for column in feature_columns:
        top50_mean = float(top50_df[column].mean()) if column in top50_df.columns else float("nan")
        top50_std = float(top50_df[column].std()) if column in top50_df.columns else float("nan")
        selected_mean = float(selected_df[column].mean()) if column in selected_df.columns else float("nan")
        feature_summary[column] = {
            "top50_mean": round_number(top50_mean),
            "top50_std": round_number(top50_std),
            "selected_top10_mean": round_number(selected_mean),
            "selected_minus_top50_mean": round_number(selected_mean - top50_mean) if pd.notna(top50_mean) else None,
        }
    return feature_summary


def build_prompt_payload(
    model_name: str,
    llm_name: str,
    scenario: str,
    date_value: pd.Timestamp,
    top50_df: pd.DataFrame,
    selected_codes: Sequence[str],
    column_names: Dict[str, str],
    top_n_first: int,
) -> Dict[str, Any]:
    asset_column = column_names["asset"]
    prediction_column = column_names["prediction"]
    target_column = column_names["target"]

    feature_columns = select_feature_columns(top50_df)
    ranked_top50 = top50_df.sort_values(prediction_column, ascending=False).head(top_n_first).copy()
    ranked_top50["rank_in_top50"] = range(1, len(ranked_top50) + 1)

    selected_df = ranked_top50[ranked_top50[asset_column].isin(selected_codes)].copy()
    selected_df["selected_order"] = selected_df[asset_column].apply(lambda code: list(selected_codes).index(code))
    selected_df = selected_df.sort_values("selected_order").copy()

    selected_records: List[Dict[str, Any]] = []
    for _, row in selected_df.iterrows():
        record = {
            "code": str(row[asset_column]),
            "rank_in_top50": int(row["rank_in_top50"]),
            "prediction": round_number(row[prediction_column]),
            "realized_return_5d": round_number(row[target_column]),
        }
        for feature in feature_columns:
            record[feature] = round_number(row[feature])
        selected_records.append(record)

    top50_summary = {
        "candidate_size": int(len(ranked_top50)),
        "prediction_mean": round_number(ranked_top50[prediction_column].mean()),
        "prediction_std": round_number(ranked_top50[prediction_column].std()),
        "realized_return_mean": round_number(ranked_top50[target_column].mean()),
        "realized_return_std": round_number(ranked_top50[target_column].std()),
    }

    return {
        "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
        "model": model_name,
        "model_label": get_traditional_model_label(model_name),
        "llm": llm_name,
        "scenario": scenario,
        "target": DEFAULT_TARGET,
        "top50_candidate_pool_summary": top50_summary,
        "selected_top10_codes_fixed": list(selected_codes),
        "selected_top10_records": selected_records,
        "feature_comparison_summary": compute_feature_summary(ranked_top50, selected_df, feature_columns),
    }


def build_explanation_prompt(payload: Dict[str, Any], top_n_first: int, top_n_final: int) -> str:
    return EXPLANATION_TAG_PROMPT_TEMPLATE.format(
        top_n_first=top_n_first,
        top_n_final=top_n_final,
        signal_tags="、".join(SIGNAL_TAGS),
        risk_tags="、".join(RISK_TAGS),
        style_tags="、".join(STYLE_TAGS),
        confidence_tags="、".join(CONFIDENCE_TAGS),
        payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
    )


def validate_tag_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    signal_tags = parsed.get("signal_tags")
    risk_tags = parsed.get("risk_tags")
    style_tag = parsed.get("style_tag")
    confidence_tag = parsed.get("confidence_tag")

    if not isinstance(signal_tags, list) or not signal_tags:
        raise ValueError("`signal_tags` must be a non-empty list.")
    if not isinstance(risk_tags, list):
        raise ValueError("`risk_tags` must be a list.")
    if not isinstance(style_tag, str) or not style_tag.strip():
        raise ValueError("`style_tag` must be a non-empty string.")
    if not isinstance(confidence_tag, str) or not confidence_tag.strip():
        raise ValueError("`confidence_tag` must be a non-empty string.")

    normalized_signal_tags = []
    for tag in signal_tags:
        tag_text = str(tag).strip()
        if tag_text not in SIGNAL_TAGS:
            raise ValueError(f"Out-of-range signal tag: {tag_text}")
        if tag_text not in normalized_signal_tags:
            normalized_signal_tags.append(tag_text)

    normalized_risk_tags = []
    for tag in risk_tags:
        tag_text = str(tag).strip()
        if tag_text not in RISK_TAGS:
            raise ValueError(f"Out-of-range risk tag: {tag_text}")
        if tag_text not in normalized_risk_tags:
            normalized_risk_tags.append(tag_text)

    style_text = str(style_tag).strip()
    confidence_text = str(confidence_tag).strip()
    if style_text not in STYLE_TAGS:
        raise ValueError(f"Out-of-range style tag: {style_text}")
    if confidence_text not in CONFIDENCE_TAGS:
        raise ValueError(f"Out-of-range confidence tag: {confidence_text}")

    return {
        "signal_tags": normalized_signal_tags,
        "risk_tags": normalized_risk_tags,
        "style_tag": style_text,
        "confidence_tag": confidence_text,
    }


def build_record_key(model_name: str, llm_name: str, scenario: str, date_value: str) -> str:
    return f"{model_name}|{llm_name}|{scenario}|{date_value}"


def is_valid_completed_record(record: Dict[str, Any]) -> bool:
    try:
        validate_tag_payload(record)
    except Exception:
        return False
    required_fields = {"date", "model", "llm", "scenario", "selected_count"}
    return required_fields.issubset(record.keys())


def load_existing_records(jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(jsonl_path):
        return records

    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not is_valid_completed_record(record):
                continue
            key = build_record_key(
                str(record["model"]),
                str(record["llm"]),
                str(record["scenario"]),
                str(record["date"]),
            )
            records[key] = record
    return records


def append_failure_row(
    failures: List[Dict[str, Any]],
    model_name: str,
    llm_name: str,
    scenario: str,
    date_value: pd.Timestamp,
    message: str,
) -> None:
    failures.append(
        {
            "model": model_name,
            "llm": llm_name,
            "scenario": scenario,
            "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
            "message": message,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )


def request_explanation_tags(
    payload: Dict[str, Any],
    top_n_first: int,
    top_n_final: int,
    llm_name: str,
    llm_config_path: str,
    api_timeout: Optional[int],
    max_retries: int,
) -> Dict[str, Any]:
    prompt = build_explanation_prompt(payload, top_n_first, top_n_final)
    messages = [
        {"role": "system", "content": "你是一个结构化解释标签生成器。只输出合法JSON。"},
        {"role": "user", "content": prompt},
    ]

    attempts = 0
    last_error: Optional[Exception] = None
    while attempts <= max_retries:
        attempts += 1
        try:
            response_text = call_llm_api(
                messages=messages,
                llm_model=llm_name,
                config_path=llm_config_path,
                timeout=api_timeout,
                max_retry=2,
                max_tokens=800,
                etf_count=int(payload.get("selected_count", DEFAULT_TOP_N_FINAL)),
                mode="explanation",
            )
            if response_text is None:
                raise ValueError("LLM returned no response.")
            parsed = _parse_json_response(response_text)
            return validate_tag_payload(parsed)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"Failed to rebuild explanation tags after retries: {last_error}")


def expand_tag_columns(records_df: pd.DataFrame) -> pd.DataFrame:
    expanded = records_df.copy()
    for tag in SIGNAL_TAGS:
        expanded[f"has_signal_{tag}"] = expanded["signal_tags"].apply(lambda tags: tag in tags)
    for tag in RISK_TAGS:
        expanded[f"has_risk_{tag}"] = expanded["risk_tags"].apply(lambda tags: tag in tags)

    expanded["signal_tags_json"] = expanded["signal_tags"].apply(lambda tags: json.dumps(tags, ensure_ascii=False))
    expanded["risk_tags_json"] = expanded["risk_tags"].apply(lambda tags: json.dumps(tags, ensure_ascii=False))
    return expanded


def summarize_tags(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    total_dates = len(records)
    summary_rows: List[Dict[str, Any]] = []

    signal_counter = Counter(tag for record in records for tag in record["signal_tags"])
    risk_counter = Counter(tag for record in records for tag in record["risk_tags"])
    style_counter = Counter(record["style_tag"] for record in records)
    confidence_counter = Counter(record["confidence_tag"] for record in records)

    for dimension, tags, counter in (
        ("signal_tags", SIGNAL_TAGS, signal_counter),
        ("risk_tags", RISK_TAGS, risk_counter),
        ("style_tag", STYLE_TAGS, style_counter),
        ("confidence_tag", CONFIDENCE_TAGS, confidence_counter),
    ):
        for tag in tags:
            count = int(counter.get(tag, 0))
            summary_rows.append(
                {
                    "dimension": dimension,
                    "tag": tag,
                    "count": count,
                    "ratio": count / total_dates if total_dates else 0.0,
                    "total_dates": total_dates,
                }
            )
    return pd.DataFrame(summary_rows)


def compute_signal_cooccurrence(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    total_dates = len(records)
    counter: Counter[Tuple[str, str]] = Counter()
    for record in records:
        unique_tags = sorted(set(record["signal_tags"]))
        for tag_a, tag_b in itertools.combinations(unique_tags, 2):
            counter[(tag_a, tag_b)] += 1

    rows = []
    for tag_a, tag_b in sorted(counter.keys()):
        count = int(counter[(tag_a, tag_b)])
        rows.append(
            {
                "tag_a": tag_a,
                "tag_b": tag_b,
                "count": count,
                "ratio": count / total_dates if total_dates else 0.0,
                "total_dates": total_dates,
            }
        )
    return pd.DataFrame(rows)


def format_ratio(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2%}"


def build_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_report_frequency_rows(summary_df: pd.DataFrame, dimension: str) -> List[Dict[str, Any]]:
    rows = []
    subset = summary_df[summary_df["dimension"] == dimension].sort_values(["count", "tag"], ascending=[False, True])
    for _, row in subset.iterrows():
        rows.append(
            {
                "标签": row["tag"],
                "出现次数": int(row["count"]),
                "出现比例": format_ratio(row["ratio"]),
            }
        )
    return rows


def write_markdown_report(
    records: Sequence[Dict[str, Any]],
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    model_labels = sorted({get_traditional_model_label(record["model"]) for record in records})
    llm_names = sorted({record["llm"] for record in records})
    scenario_names = sorted({record["scenario"] for record in records})

    signal_rows = build_report_frequency_rows(summary_df, "signal_tags")
    risk_rows = build_report_frequency_rows(summary_df, "risk_tags")

    lines = [
        "# 解释标签补统计简报",
        "",
        f"- 共统计 {len(records)} 个调仓日结果",
        f"- 覆盖模型数：{len(model_labels)}（{', '.join(model_labels)}）",
        f"- 覆盖第二阶段LLM：{', '.join(llm_names)}",
        f"- 覆盖scenario：{', '.join(scenario_names)}",
        "",
        "## Signal Tags 频率表",
        "",
        build_markdown_table(signal_rows, ["标签", "出现次数", "出现比例"]),
        "",
        "## Risk Tags 频率表",
        "",
        build_markdown_table(risk_rows, ["标签", "出现次数", "出现比例"]),
        "",
        "## 说明",
        "",
        "解释标签统计用于描述第二阶段重排序中高频关注的信息类型分布，从而为论文中的解释性分析提供全样本层面的统计支撑。",
        "这些标签结果属于“在相同最终结果与相同约束下的解释性复现”，用于增强整体解释性分析，而不应被表述为模型内部真实推理过程的直接记录。",
    ]

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def prepare_tasks(
    models: Sequence[str],
    llms: Sequence[str],
    scenarios: Sequence[str],
) -> List[Tuple[str, str, str]]:
    tasks = []
    for model_name in models:
        for llm_name in llms:
            for scenario in scenarios:
                tasks.append((model_name, llm_name, scenario))
    return tasks


def rebuild_tags_for_run(
    model_name: str,
    llm_name: str,
    scenario: str,
    prediction_df: pd.DataFrame,
    column_names: Dict[str, str],
    backtest_df: pd.DataFrame,
    completed_records: Dict[str, Dict[str, Any]],
    failures: List[Dict[str, Any]],
    llm_config_path: str,
    max_retries: int,
    api_timeout: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    asset_column = column_names["asset"]
    prediction_column = column_names["prediction"]
    top_n_first, top_n_final = parse_scenario_top_n(scenario)

    run_results: Dict[str, Dict[str, Any]] = {}
    for _, row in backtest_df.iterrows():
        date_value = pd.Timestamp(row["rebalance_date"]).normalize()
        date_text = date_value.strftime("%Y-%m-%d")
        record_key = build_record_key(model_name, llm_name, scenario, date_text)

        if record_key in completed_records:
            run_results[record_key] = completed_records[record_key]
            continue

        selected_codes = parse_code_list(row.get(BACKTEST_CODE_COLUMN))
        if len(selected_codes) != top_n_final:
            append_failure_row(
                failures,
                model_name,
                llm_name,
                scenario,
                date_value,
                f"Invalid saved final selection size: {len(selected_codes)}",
            )
            continue

        day_df = prediction_df[prediction_df[column_names["date"]] == date_value].copy()
        if day_df.empty:
            append_failure_row(
                failures,
                model_name,
                llm_name,
                scenario,
                date_value,
                "Prediction file has no rows for this rebalance date.",
            )
            continue

        top50_df = day_df.sort_values(prediction_column, ascending=False).head(top_n_first).copy()
        if len(top50_df) < top_n_first:
            append_failure_row(
                failures,
                model_name,
                llm_name,
                scenario,
                date_value,
                f"Candidate pool is incomplete: {len(top50_df)} / {top_n_first}",
            )
            continue

        selected_in_top50 = top50_df[top50_df[asset_column].isin(selected_codes)][asset_column].astype(str).tolist()
        if sorted(selected_in_top50) != sorted(selected_codes):
            append_failure_row(
                failures,
                model_name,
                llm_name,
                scenario,
                date_value,
                "Saved final Top10 cannot be fully matched back to reconstructed Top50 candidate pool.",
            )
            continue

        payload = build_prompt_payload(
            model_name=model_name,
            llm_name=llm_name,
            scenario=scenario,
            date_value=date_value,
            top50_df=top50_df,
            selected_codes=selected_codes,
            column_names=column_names,
            top_n_first=top_n_first,
        )
        payload["selected_count"] = len(selected_codes)

        try:
            tags = request_explanation_tags(
                payload=payload,
                top_n_first=top_n_first,
                top_n_final=top_n_final,
                llm_name=llm_name,
                llm_config_path=llm_config_path,
                api_timeout=api_timeout,
                max_retries=max_retries,
            )
        except Exception as exc:  # noqa: BLE001
            append_failure_row(
                failures,
                model_name,
                llm_name,
                scenario,
                date_value,
                str(exc),
            )
            continue

        run_results[record_key] = {
            "date": date_text,
            "model": model_name,
            "llm": llm_name,
            "scenario": scenario,
            "selected_count": len(selected_codes),
            "signal_tags": tags["signal_tags"],
            "risk_tags": tags["risk_tags"],
            "style_tag": tags["style_tag"],
            "confidence_tag": tags["confidence_tag"],
        }

    return run_results


def main() -> int:
    args = parse_args()
    if args.target != DEFAULT_TARGET:
        raise ValueError(f"Current script only supports `{DEFAULT_TARGET}`.")

    output_dir = ensure_output_dir(args.output_dir)
    jsonl_path = os.path.join(output_dir, "explanation_tags_by_date.jsonl")
    csv_path = os.path.join(output_dir, "explanation_tags_by_date.csv")
    summary_path = os.path.join(output_dir, "explanation_tag_summary.csv")
    cooccurrence_path = os.path.join(output_dir, "explanation_tag_cooccurrence.csv")
    report_path = os.path.join(output_dir, "explanation_tag_report.md")
    failure_log_path = os.path.join(output_dir, "failure_log.csv")

    completed_records = load_existing_records(jsonl_path)
    failures: List[Dict[str, Any]] = []
    new_records: Dict[str, Dict[str, Any]] = {}

    for model_name, llm_name, scenario in prepare_tasks(args.models, args.llms, args.scenarios):
        print(f"Rebuilding explanation tags for model={model_name}, llm={llm_name}, scenario={scenario}...")
        prediction_df, prediction_path, column_names = load_prediction_df(model_name)
        backtest_df, backtest_path = load_backtest_selection_df(model_name, llm_name, scenario)
        print(f"  Prediction source: {prediction_path}")
        print(f"  Selection source: {backtest_path}")

        run_records = rebuild_tags_for_run(
            model_name=model_name,
            llm_name=llm_name,
            scenario=scenario,
            prediction_df=prediction_df,
            column_names=column_names,
            backtest_df=backtest_df,
            completed_records=completed_records,
            failures=failures,
            llm_config_path=args.llm_config,
            max_retries=args.max_retries,
            api_timeout=args.api_timeout,
        )
        new_records.update(run_records)

    all_records_map = dict(completed_records)
    all_records_map.update(new_records)
    ordered_records = sorted(
        all_records_map.values(),
        key=lambda record: (record["model"], record["llm"], record["scenario"], record["date"]),
    )

    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for record in ordered_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    records_df = pd.DataFrame(ordered_records)
    if records_df.empty:
        raise ValueError("No valid explanation-tag results were generated.")

    expanded_df = expand_tag_columns(records_df)
    expanded_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_df = summarize_tags(ordered_records)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    cooccurrence_df = compute_signal_cooccurrence(ordered_records)
    cooccurrence_df.to_csv(cooccurrence_path, index=False, encoding="utf-8-sig")

    failure_df = pd.DataFrame(failures)
    if failure_df.empty:
        failure_df = pd.DataFrame(columns=["model", "llm", "scenario", "date", "message", "timestamp"])
    failure_df.to_csv(failure_log_path, index=False, encoding="utf-8-sig")

    write_markdown_report(
        records=ordered_records,
        summary_df=summary_df,
        output_path=report_path,
    )

    print("Explanation-tag rebuild completed.")
    print(f"- jsonl: {jsonl_path}")
    print(f"- csv: {csv_path}")
    print(f"- summary: {summary_path}")
    print(f"- cooccurrence: {cooccurrence_path}")
    print(f"- report: {report_path}")
    print(f"- failures: {failure_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

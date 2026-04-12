"""Analyze candidate-pool quality from existing stage-one prediction files."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from traditional_model_config import (
    PLANNED_TRADITIONAL_MODELS,
    get_traditional_model_label,
    get_traditional_prediction_file,
    get_traditional_training_eval_dir,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_TARGET = "Y_future_5d_return"
DEFAULT_TOP_N_FIRST = 50
DEFAULT_TOP_N_FINAL = 10
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "analysis", "candidate_pool_quality")

DATE_COLUMN_CANDIDATES = ("date", "trade_date", "日期")
ASSET_COLUMN_CANDIDATES = ("etf_code", "code", "symbol")
PREDICTION_COLUMN_CANDIDATES = ("prediction", "y_pred", "pred")
SUMMARY_METRIC_ORDER = (
    "candidate_pred_std",
    "candidate_return_std",
    "top10_mean_return",
    "top50_mean_return",
    "top10_excess_within_top50",
)
CORE_REPORT_METRICS = (
    "candidate_pred_std",
    "top10_excess_within_top50",
    "candidate_return_std",
)
REPORT_METRIC_LABELS = {
    "candidate_pred_std": "候选池预测分数离散度",
    "candidate_return_std": "候选池真实收益离散度",
    "top10_mean_return": "Top10真实收益均值",
    "top50_mean_return": "Top50真实收益均值",
    "top10_excess_within_top50": "Top10相对Top50收益梯度",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计不同传统模型的候选池质量差异。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(PLANNED_TRADITIONAL_MODELS),
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="需要统计的第一阶段模型列表。",
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
        help="候选池大小。",
    )
    parser.add_argument(
        "--top-n-final",
        type=int,
        default=DEFAULT_TOP_N_FINAL,
        help="Top10分层大小。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="分析结果输出目录。",
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
        raise ValueError(
            f"Ambiguous prediction columns for target `{target}`: {compatible_columns}"
        )

    fallback_match = resolve_exact_column(columns, PREDICTION_COLUMN_CANDIDATES)
    if fallback_match:
        return fallback_match
    return None


def infer_column_names(df: pd.DataFrame, target: str) -> Dict[str, str]:
    columns = list(df.columns)
    date_column = resolve_exact_column(columns, DATE_COLUMN_CANDIDATES)
    if not date_column:
        raise ValueError(
            f"Failed to infer date column. Supported names: {', '.join(DATE_COLUMN_CANDIDATES)}"
        )

    asset_column = resolve_exact_column(columns, ASSET_COLUMN_CANDIDATES)
    if not asset_column:
        raise ValueError(
            f"Failed to infer asset identifier column. Supported names: {', '.join(ASSET_COLUMN_CANDIDATES)}"
        )

    target_column = resolve_exact_column(columns, [target])
    if not target_column:
        raise ValueError(f"Required target column is missing: `{target}`")

    prediction_column = infer_prediction_column(columns, target)
    if not prediction_column:
        raise ValueError(
            "Failed to infer prediction column. "
            f"Supported generic names: {', '.join(PREDICTION_COLUMN_CANDIDATES)}, "
            f"or target-specific forms such as `y_pred_{target}`."
        )

    return {
        "date": date_column,
        "asset": asset_column,
        "target": target_column,
        "prediction": prediction_column,
    }


def load_model_prediction_file(model_name: str) -> Tuple[pd.DataFrame, str]:
    prediction_path = get_traditional_prediction_file(PROJECT_ROOT, model_name)
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Prediction file not found for `{model_name}`: {prediction_path}")

    df = pd.read_csv(prediction_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Prediction file is empty for `{model_name}`: {prediction_path}")
    return df, prediction_path


def load_reusable_ranking_metrics_by_date(
    model_name: str,
    target: str,
) -> Optional[pd.DataFrame]:
    metrics_path = os.path.join(
        get_traditional_training_eval_dir(PROJECT_ROOT, model_name),
        "ranking_metrics_by_date.csv",
    )
    if not os.path.exists(metrics_path):
        return None

    metrics_df = pd.read_csv(metrics_path, encoding="utf-8-sig")
    if metrics_df.empty:
        return None

    required_columns = {"date", "top10_mean_target", "top50_mean_target"}
    if not required_columns.issubset(metrics_df.columns):
        return None

    if "target" in metrics_df.columns:
        metrics_df = metrics_df[metrics_df["target"].astype(str) == target].copy()
        if metrics_df.empty:
            return None

    metrics_df["date"] = pd.to_datetime(metrics_df["date"], errors="coerce")
    metrics_df = metrics_df.dropna(subset=["date"]).copy()
    if metrics_df.empty:
        return None

    metrics_df["date"] = metrics_df["date"].dt.normalize()
    metrics_df["top10_excess_within_top50"] = (
        metrics_df["top10_mean_target"] - metrics_df["top50_mean_target"]
    )
    return metrics_df[
        ["date", "top10_mean_target", "top50_mean_target", "top10_excess_within_top50"]
    ].drop_duplicates(subset=["date"])


def build_issue_record(
    model_name: str,
    date_value: Optional[pd.Timestamp],
    level: str,
    message: str,
) -> Dict[str, str]:
    return {
        "model": model_name,
        "date": "" if date_value is None else pd.Timestamp(date_value).strftime("%Y-%m-%d"),
        "level": level,
        "message": message,
    }


def compute_candidate_pool_metrics(
    model_name: str,
    df: pd.DataFrame,
    column_names: Dict[str, str],
    target: str,
    top_n_first: int,
    top_n_final: int,
    ranking_metrics_by_date: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    working_df = df.copy()
    working_df[column_names["date"]] = pd.to_datetime(working_df[column_names["date"]], errors="coerce")
    working_df = working_df.dropna(
        subset=[column_names["date"], column_names["prediction"], column_names["target"]]
    ).copy()
    if working_df.empty:
        raise ValueError(f"No valid rows remain after dropping missing values for `{model_name}`.")

    working_df[column_names["date"]] = working_df[column_names["date"]].dt.normalize()
    working_df[column_names["asset"]] = working_df[column_names["asset"]].astype(str)

    reusable_map: Dict[pd.Timestamp, Dict[str, float]] = {}
    if ranking_metrics_by_date is not None and not ranking_metrics_by_date.empty:
        reusable_map = {
            row["date"]: {
                "top10_mean_return": float(row["top10_mean_target"]),
                "top50_mean_return": float(row["top50_mean_target"]),
                "top10_excess_within_top50": float(row["top10_excess_within_top50"]),
            }
            for _, row in ranking_metrics_by_date.iterrows()
        }

    detail_rows: List[Dict[str, Any]] = []
    issues: List[Dict[str, str]] = []

    grouped = working_df.groupby(column_names["date"], sort=True)
    for date_value, day_df in grouped:
        ranked_day_df = day_df.sort_values(column_names["prediction"], ascending=False)
        top50_df = ranked_day_df.head(top_n_first).copy()
        top10_df = ranked_day_df.head(top_n_final).copy()

        if len(top50_df) < top_n_first:
            issues.append(
                build_issue_record(
                    model_name,
                    date_value,
                    "warning",
                    f"Candidate pool size is {len(top50_df)}, below required top_n_first={top_n_first}.",
                )
            )
            continue
        if len(top10_df) < top_n_final:
            issues.append(
                build_issue_record(
                    model_name,
                    date_value,
                    "warning",
                    f"Top10 size is {len(top10_df)}, below required top_n_final={top_n_final}.",
                )
            )
            continue

        reused_metrics = reusable_map.get(pd.Timestamp(date_value))
        if reused_metrics is None:
            top10_mean_return = float(top10_df[column_names["target"]].mean())
            top50_mean_return = float(top50_df[column_names["target"]].mean())
            top10_excess_within_top50 = top10_mean_return - top50_mean_return
        else:
            top10_mean_return = reused_metrics["top10_mean_return"]
            top50_mean_return = reused_metrics["top50_mean_return"]
            top10_excess_within_top50 = reused_metrics["top10_excess_within_top50"]

        detail_rows.append(
            {
                "model": model_name,
                "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "candidate_pred_std": float(top50_df[column_names["prediction"]].std()),
                "candidate_return_std": float(top50_df[column_names["target"]].std()),
                "top10_mean_return": top10_mean_return,
                "top50_mean_return": top50_mean_return,
                "top10_excess_within_top50": top10_excess_within_top50,
                "candidate_size": int(len(top50_df)),
                "top10_size": int(len(top10_df)),
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        raise ValueError(f"No valid candidate-pool metric rows were generated for `{model_name}`.")

    return detail_df, issues


def summarize_metrics(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    for model_name, model_df in detail_df.groupby("model", sort=False):
        for metric in SUMMARY_METRIC_ORDER:
            metric_series = pd.to_numeric(model_df[metric], errors="coerce").dropna()
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": float(metric_series.mean()) if not metric_series.empty else float("nan"),
                    "std": float(metric_series.std()) if not metric_series.empty else float("nan"),
                    "min": float(metric_series.min()) if not metric_series.empty else float("nan"),
                    "max": float(metric_series.max()) if not metric_series.empty else float("nan"),
                    "count": int(metric_series.count()),
                }
            )
    return pd.DataFrame(summary_rows)


def format_float(value: Any, digits: int = 6) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def build_markdown_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_core_report_rows(summary_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_name in PLANNED_TRADITIONAL_MODELS:
        model_summary = summary_df[summary_df["model"] == model_name].copy()
        if model_summary.empty:
            continue
        metric_lookup = {
            row["metric"]: row for _, row in model_summary.iterrows()
        }
        rows.append(
            {
                "模型": get_traditional_model_label(model_name),
                "候选池预测分数离散度均值": format_float(
                    metric_lookup.get("candidate_pred_std", {}).get("mean")
                ),
                "Top10相对Top50收益梯度均值": format_float(
                    metric_lookup.get("top10_excess_within_top50", {}).get("mean")
                ),
                "候选池真实收益离散度均值": format_float(
                    metric_lookup.get("candidate_return_std", {}).get("mean")
                ),
                "有效日期数": metric_lookup.get("candidate_pred_std", {}).get("count", ""),
            }
        )
    return rows


def build_metric_ranking_lines(summary_df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    for metric in CORE_REPORT_METRICS:
        metric_summary = summary_df[summary_df["metric"] == metric].sort_values("mean", ascending=False)
        if metric_summary.empty:
            continue
        ranking_text = " > ".join(
            f"{get_traditional_model_label(row['model'])}({format_float(row['mean'])})"
            for _, row in metric_summary.iterrows()
        )
        lines.append(f"- {REPORT_METRIC_LABELS[metric]}按均值从高到低排序：{ranking_text}")
    return lines


def write_markdown_report(
    summary_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    output_path: str,
    target: str,
    top_n_first: int,
    top_n_final: int,
) -> None:
    core_rows = build_core_report_rows(summary_df)
    ranking_lines = build_metric_ranking_lines(summary_df)
    issue_count = 0 if issues_df.empty else len(issues_df)

    lines = [
        "# 候选池质量差异离线统计报告",
        "",
        f"- 目标列：`{target}`",
        f"- 候选池口径：按日期分组后取 Top{top_n_first} 作为候选池，再取 Top{top_n_final} 作为 Top{top_n_final} 分层",
        f"- 模型范围：{', '.join(get_traditional_model_label(model) for model in PLANNED_TRADITIONAL_MODELS)}",
        "",
        "## 四个模型核心指标汇总",
        "",
        build_markdown_table(
            core_rows,
            ["模型", "候选池预测分数离散度均值", "Top10相对Top50收益梯度均值", "候选池真实收益离散度均值", "有效日期数"],
        ),
        "",
        "## 均值排序说明",
        "",
        *ranking_lines,
        "",
        "## 说明",
        "",
        "上述指标分别用于刻画候选池内部的预测分化程度、收益分层程度和真实收益波动空间，可为“不同传统模型下大模型重排序效果差异”提供描述性数据支撑。",
        "这些统计结果可以用于强化论文中的机制解释，但它们本身不直接构成因果证明。",
        "",
        f"本次分析记录的异常或跳过条目数量：{issue_count}。",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main() -> int:
    args = parse_args()
    if args.target != DEFAULT_TARGET:
        raise ValueError(f"Current analysis only supports `{DEFAULT_TARGET}`.")
    if args.top_n_first <= 0 or args.top_n_final <= 0:
        raise ValueError("top_n_first and top_n_final must be positive integers.")
    if args.top_n_first < args.top_n_final:
        raise ValueError("top_n_first must be greater than or equal to top_n_final.")

    output_dir = ensure_output_dir(args.output_dir)
    detail_frames: List[pd.DataFrame] = []
    issue_rows: List[Dict[str, str]] = []

    for model_name in args.models:
        print(f"Analyzing candidate-pool quality for {model_name}...")
        prediction_df, prediction_path = load_model_prediction_file(model_name)
        print(f"  Prediction file: {prediction_path}")

        column_names = infer_column_names(prediction_df, target=args.target)
        ranking_metrics_by_date = load_reusable_ranking_metrics_by_date(model_name, target=args.target)
        if ranking_metrics_by_date is not None:
            print("  Reusing top10/top50 mean-return fields from ranking_metrics_by_date.csv")
        else:
            print("  ranking_metrics_by_date.csv is unavailable or incompatible; computing all metrics directly")

        detail_df, issues = compute_candidate_pool_metrics(
            model_name=model_name,
            df=prediction_df,
            column_names=column_names,
            target=args.target,
            top_n_first=args.top_n_first,
            top_n_final=args.top_n_final,
            ranking_metrics_by_date=ranking_metrics_by_date,
        )
        detail_frames.append(detail_df)
        issue_rows.extend(issues)

    detail_output = pd.concat(detail_frames, ignore_index=True)
    summary_output = summarize_metrics(detail_output)
    issues_output = pd.DataFrame(issue_rows, columns=["model", "date", "level", "message"])

    detail_path = os.path.join(output_dir, "candidate_pool_quality_by_date.csv")
    summary_path = os.path.join(output_dir, "candidate_pool_quality_summary.csv")
    report_path = os.path.join(output_dir, "candidate_pool_quality_report.md")
    issues_path = os.path.join(output_dir, "candidate_pool_quality_issues.csv")

    detail_output.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary_output.to_csv(summary_path, index=False, encoding="utf-8-sig")
    issues_output.to_csv(issues_path, index=False, encoding="utf-8-sig")
    write_markdown_report(
        summary_df=summary_output,
        issues_df=issues_output,
        output_path=report_path,
        target=args.target,
        top_n_first=args.top_n_first,
        top_n_final=args.top_n_final,
    )

    print("Candidate-pool quality analysis completed.")
    print(f"- by_date: {detail_path}")
    print(f"- summary: {summary_path}")
    print(f"- report: {report_path}")
    print(f"- issues: {issues_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

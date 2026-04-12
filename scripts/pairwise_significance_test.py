"""Generate pairwise paired-sample t-tests for Table 5-2 comparisons."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
from scipy import stats

from market_data_utils import read_csv_with_fallback
from traditional_model_config import (
    get_experiment_summary_dir,
    get_traditional_model_label,
    get_traditional_two_stage_output_dir,
    normalize_second_stage_llm_name,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_SCENARIO = "default"
MODEL_ORDER = ("random_forest", "linear", "lightgbm", "xgboost")
LLM_ORDER = ("deepseek-chat", "gemini-2.5-flash-lite")
BACKTEST_DATE_COLUMNS = ("rebalance_date", "date")
NET_RETURN_COLUMNS = {
    "baseline": ("strategy_a_net_return", "strategy_a_return"),
    "llm": ("strategy_b_net_return", "strategy_b_return"),
}
NET_METRIC_NAMES = ("annual_return", "sharpe_ratio", "max_drawdown")
LLM_DISPLAY_LABELS = {
    "deepseek-chat": "DeepSeek",
    "gemini-2.5-flash-lite": "Gemini",
}


def to_project_relative_path(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path_obj)


@dataclass(frozen=True)
class ComparisonSpec:
    traditional_model: str
    llm_model: str

    @property
    def model_label(self) -> str:
        return get_traditional_model_label(self.traditional_model)

    @property
    def llm_label(self) -> str:
        return LLM_DISPLAY_LABELS.get(self.llm_model, self.llm_model)

    @property
    def normalized_llm(self) -> str:
        return normalize_second_stage_llm_name(self.llm_model)

    @property
    def run_dir(self) -> Path:
        return get_traditional_two_stage_output_dir(PROJECT_ROOT, self.traditional_model, self.llm_model)

    @property
    def backtest_results_path(self) -> Path:
        return self.run_dir / "backtest_results.csv"

    @property
    def performance_report_path(self) -> Path:
        return self.run_dir / "performance_report.json"

    @property
    def performance_metrics_path(self) -> Path:
        return self.run_dir / "performance_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成表 5-2 的成对样本 t 检验结果。")
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO, help="回测场景名称，默认 default。")
    parser.add_argument(
        "--output-dir",
        default=str(get_experiment_summary_dir(PROJECT_ROOT)),
        help="结果输出目录，默认 experiments/summary。",
    )
    return parser.parse_args()


def coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isnan(number):
            return None
        return number
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
        try:
            return float(text) / 100
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def resolve_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def load_json_file(file_path: Path) -> Optional[dict[str, Any]]:
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_metric_snapshot(spec: ComparisonSpec) -> tuple[dict[str, Optional[float]], dict[str, Optional[float]], list[str]]:
    warnings: list[str] = []
    report = load_json_file(spec.performance_report_path)

    if isinstance(report, dict):
        baseline_section = report.get("performance_strategy_a_net", {})
        llm_section = report.get("performance_strategy_b_net", {})
        if isinstance(baseline_section, dict) and isinstance(llm_section, dict):
            baseline_metrics = {metric_name: coerce_number(baseline_section.get(metric_name)) for metric_name in NET_METRIC_NAMES}
            llm_metrics = {metric_name: coerce_number(llm_section.get(metric_name)) for metric_name in NET_METRIC_NAMES}
            if any(value is not None for value in baseline_metrics.values()) and any(value is not None for value in llm_metrics.values()):
                return baseline_metrics, llm_metrics, warnings
        warnings.append(f"performance_report.json 缺少净值绩效指标: {to_project_relative_path(spec.performance_report_path)}")
    else:
        warnings.append(f"performance_report.json 不存在: {to_project_relative_path(spec.performance_report_path)}")

    if not spec.performance_metrics_path.exists():
        raise FileNotFoundError(
            f"缺少 performance_report.json 和 performance_metrics.csv: {to_project_relative_path(spec.run_dir)}"
        )

    metrics_df = read_csv_with_fallback(spec.performance_metrics_path, index_col=0)
    required_columns = ("strategy_a_net", "strategy_b_net")
    missing_columns = [column for column in required_columns if column not in metrics_df.columns]
    if missing_columns:
        raise KeyError(f"{to_project_relative_path(spec.performance_metrics_path)} 缺少列: {', '.join(missing_columns)}")

    missing_rows = [metric_name for metric_name in NET_METRIC_NAMES if metric_name not in metrics_df.index]
    if missing_rows:
        raise KeyError(f"{to_project_relative_path(spec.performance_metrics_path)} 缺少指标: {', '.join(missing_rows)}")

    warnings.append(f"已回退读取 performance_metrics.csv: {to_project_relative_path(spec.performance_metrics_path)}")
    baseline_metrics = {metric_name: coerce_number(metrics_df.at[metric_name, "strategy_a_net"]) for metric_name in NET_METRIC_NAMES}
    llm_metrics = {metric_name: coerce_number(metrics_df.at[metric_name, "strategy_b_net"]) for metric_name in NET_METRIC_NAMES}
    return baseline_metrics, llm_metrics, warnings


def load_paired_period_returns(spec: ComparisonSpec) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    if not spec.backtest_results_path.exists():
        raise FileNotFoundError(f"缺少 backtest_results.csv: {to_project_relative_path(spec.backtest_results_path)}")

    results_df = read_csv_with_fallback(spec.backtest_results_path)
    warnings: list[str] = []

    date_column = resolve_existing_column(results_df, BACKTEST_DATE_COLUMNS)
    if date_column is None:
        raise KeyError(
            f"{to_project_relative_path(spec.backtest_results_path)} 缺少调仓日期列，当前列为: {', '.join(results_df.columns)}"
        )

    baseline_column = resolve_existing_column(results_df, NET_RETURN_COLUMNS["baseline"])
    llm_column = resolve_existing_column(results_df, NET_RETURN_COLUMNS["llm"])
    if baseline_column is None or llm_column is None:
        raise KeyError(
            f"{to_project_relative_path(spec.backtest_results_path)} 缺少策略收益列，当前列为: {', '.join(results_df.columns)}"
        )

    paired = results_df[[date_column, baseline_column, llm_column]].copy()
    paired.rename(
        columns={
            date_column: "rebalance_date",
            baseline_column: "baseline_period_return",
            llm_column: "llm_period_return",
        },
        inplace=True,
    )
    paired["rebalance_date"] = pd.to_datetime(paired["rebalance_date"], errors="coerce")
    paired["baseline_period_return"] = pd.to_numeric(paired["baseline_period_return"], errors="coerce")
    paired["llm_period_return"] = pd.to_numeric(paired["llm_period_return"], errors="coerce")

    missing_date_count = int(paired["rebalance_date"].isna().sum())
    missing_baseline_count = int(paired["baseline_period_return"].isna().sum())
    missing_llm_count = int(paired["llm_period_return"].isna().sum())
    if missing_date_count:
        warnings.append(f"丢弃 {missing_date_count} 行无法解析的调仓日期")
    if missing_baseline_count:
        warnings.append(f"baseline 有 {missing_baseline_count} 行收益缺失")
    if missing_llm_count:
        warnings.append(f"LLM 策略有 {missing_llm_count} 行收益缺失")

    paired = paired.dropna(subset=["rebalance_date", "baseline_period_return", "llm_period_return"]).copy()
    paired.sort_values("rebalance_date", inplace=True)

    duplicate_count = int(paired.duplicated(subset=["rebalance_date"]).sum())
    if duplicate_count:
        warnings.append(f"发现 {duplicate_count} 个重复调仓日，已按均值合并")
        paired = (
            paired.groupby("rebalance_date", as_index=False)[["baseline_period_return", "llm_period_return"]]
            .mean()
            .sort_values("rebalance_date")
        )

    metadata = {
        "date_column": date_column,
        "baseline_column": baseline_column,
        "llm_column": llm_column,
        "raw_row_count": int(len(results_df)),
        "aligned_row_count": int(len(paired)),
        "missing_date_count": missing_date_count,
        "missing_baseline_count": missing_baseline_count,
        "missing_llm_count": missing_llm_count,
        "duplicate_date_count": duplicate_count,
    }
    return paired, metadata, warnings


def compute_paired_t_test(aligned_returns: pd.DataFrame) -> tuple[Optional[float], Optional[float], list[str]]:
    warnings: list[str] = []
    sample_size = len(aligned_returns)
    if sample_size < 2:
        warnings.append("有效样本数不足 2，无法计算成对样本 t 检验")
        return None, None, warnings

    differences = aligned_returns["llm_period_return"] - aligned_returns["baseline_period_return"]
    if differences.nunique(dropna=True) <= 1:
        if float(differences.iloc[0]) == 0.0:
            warnings.append("两策略逐期收益完全相同，按 t=0、p=1 处理")
            return 0.0, 1.0, warnings
        warnings.append("收益差分方差为 0，t 检验不可计算")
        return None, None, warnings

    test_result = stats.ttest_rel(
        aligned_returns["llm_period_return"],
        aligned_returns["baseline_period_return"],
        nan_policy="omit",
    )
    t_statistic = coerce_number(test_result.statistic)
    p_value = coerce_number(test_result.pvalue)
    if t_statistic is None or p_value is None:
        warnings.append("t 检验返回空结果")
    return t_statistic, p_value, warnings


def get_significance_marker(p_value: Optional[float]) -> str:
    if p_value is None:
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def build_comparison_specs(scenario: str) -> list[ComparisonSpec]:
    if scenario != DEFAULT_SCENARIO:
        raise ValueError(f"当前仅支持 default 场景，收到: {scenario}")
    return [ComparisonSpec(model_name, llm_name) for model_name in MODEL_ORDER for llm_name in LLM_ORDER]


def format_signed_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2%}"


def format_signed_float(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.4f}"


def format_p_value(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def format_warning_text(warnings: Sequence[str]) -> str:
    cleaned = [warning.strip() for warning in warnings if str(warning).strip()]
    return "；".join(cleaned)


def build_result_row(
    spec: ComparisonSpec,
    aligned_returns: pd.DataFrame,
    metadata: dict[str, Any],
    baseline_metrics: dict[str, Optional[float]],
    llm_metrics: dict[str, Optional[float]],
    warnings: Sequence[str],
    t_statistic: Optional[float],
    p_value: Optional[float],
) -> dict[str, Any]:
    annual_return_change = None if baseline_metrics.get("annual_return") is None or llm_metrics.get("annual_return") is None else llm_metrics["annual_return"] - baseline_metrics["annual_return"]
    sharpe_ratio_change = None if baseline_metrics.get("sharpe_ratio") is None or llm_metrics.get("sharpe_ratio") is None else llm_metrics["sharpe_ratio"] - baseline_metrics["sharpe_ratio"]
    max_drawdown_change = None if baseline_metrics.get("max_drawdown") is None or llm_metrics.get("max_drawdown") is None else llm_metrics["max_drawdown"] - baseline_metrics["max_drawdown"]

    return {
        "traditional_model": spec.traditional_model,
        "llm_name": spec.normalized_llm,
        "scenario_name": DEFAULT_SCENARIO,
        "第一阶段模型": spec.model_label,
        "大模型": spec.llm_label,
        "年化收益率变化": annual_return_change,
        "夏普比率变化": sharpe_ratio_change,
        "最大回撤变化": max_drawdown_change,
        "t统计量": t_statistic,
        "p值": p_value,
        "显著性": get_significance_marker(p_value),
        "有效样本数": int(len(aligned_returns)),
        "警告": format_warning_text(warnings),
        "backtest_results_path": to_project_relative_path(spec.backtest_results_path),
        "performance_report_path": to_project_relative_path(spec.performance_report_path),
        "performance_metrics_path": to_project_relative_path(spec.performance_metrics_path),
        "收益日期列": metadata.get("date_column"),
        "baseline收益列": metadata.get("baseline_column"),
        "llm收益列": metadata.get("llm_column"),
    }


def generate_results(specs: Iterable[ComparisonSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        aligned_returns, metadata, return_warnings = load_paired_period_returns(spec)
        baseline_metrics, llm_metrics, metric_warnings = load_metric_snapshot(spec)
        t_statistic, p_value, test_warnings = compute_paired_t_test(aligned_returns)
        rows.append(
            build_result_row(
                spec,
                aligned_returns=aligned_returns,
                metadata=metadata,
                baseline_metrics=baseline_metrics,
                llm_metrics=llm_metrics,
                warnings=[*return_warnings, *metric_warnings, *test_warnings],
                t_statistic=t_statistic,
                p_value=p_value,
            )
        )
    return rows


def write_csv_output(file_path: Path, rows: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(file_path, index=False, encoding="utf-8-sig")


def build_markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "第一阶段模型",
        "大模型",
        "年化收益率变化",
        "夏普比率变化",
        "最大回撤变化",
        "t统计量",
        "p值",
        "显著性",
        "有效样本数",
        "警告",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["第一阶段模型"]),
                    str(row["大模型"]),
                    format_signed_percent(coerce_number(row["年化收益率变化"])),
                    format_signed_float(coerce_number(row["夏普比率变化"])),
                    format_signed_percent(coerce_number(row["最大回撤变化"])),
                    format_signed_float(coerce_number(row["t统计量"])),
                    format_p_value(coerce_number(row["p值"])),
                    str(row["显著性"]),
                    str(int(row["有效样本数"])),
                    str(row["警告"] or ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_markdown_output(rows: list[dict[str, Any]]) -> str:
    note = (
        "显著性基于各调仓周期收益序列的成对样本 t 检验计算，"
        "检验对象为基线策略与对应两阶段策略在相同调仓期下的收益差异。"
        "*, **, *** 分别表示在 10%、5% 和 1% 显著性水平下显著。"
    )
    return "\n".join(
        [
            "# 表5-2 成对比较结果的统计显著性检验",
            "",
            build_markdown_table(rows),
            "",
            note,
            "",
        ]
    )


def write_markdown_output(file_path: Path, rows: list[dict[str, Any]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(build_markdown_output(rows), encoding="utf-8")


def print_console_summary(rows: list[dict[str, Any]], csv_path: Path, markdown_path: Path) -> None:
    print("表5-2 成对样本 t 检验结果")
    print("逐期收益口径来自 backtest_results.csv 的 strategy_a_net_return / strategy_b_net_return。")
    for row in rows:
        print(
            f"- {row['第一阶段模型']} + {row['大模型']}: "
            f"n={int(row['有效样本数'])}, "
            f"t={format_signed_float(coerce_number(row['t统计量']))}, "
            f"p={format_p_value(coerce_number(row['p值']))}, "
            f"sig={row['显著性'] or 'ns'}"
        )
        if row["警告"]:
            print(f"  警告: {row['警告']}")
    print(f"CSV: {to_project_relative_path(csv_path)}")
    print(f"Markdown: {to_project_relative_path(markdown_path)}")


def main() -> None:
    args = parse_args()
    specs = build_comparison_specs(args.scenario)
    rows = generate_results(specs)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / "pairwise_significance_results.csv"
    markdown_path = output_dir / "pairwise_significance_results.md"
    write_csv_output(csv_path, rows)
    write_markdown_output(markdown_path, rows)
    print_console_summary(rows, csv_path, markdown_path)


if __name__ == "__main__":
    main()

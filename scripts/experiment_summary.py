"""Generate a minimal Chinese experiments summary page for default-scenario runs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from traditional_model_config import (
    PLANNED_TRADITIONAL_MODELS,
    get_experiment_summary_dir,
    get_traditional_experiment_runs_dir,
    get_traditional_model_label,
    get_traditional_two_stage_dir,
    normalize_second_stage_llm_name,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_SCENARIO = "default"
MODEL_ORDER = ("random_forest", "linear", "lightgbm", "xgboost")
RUN_ORDER = ("baseline", "deepseek_chat", "gemini_2_5_flash_lite")
ROBUSTNESS_MODEL = "lightgbm"
ROBUSTNESS_LLM = "deepseek_chat"
ROBUSTNESS_SCENARIOS = ("default", "top30_top10", "top50_top5")

SUPPORTED_LLM_NAMES = {
    "baseline",
    "deepseek_chat",
    "gemini_2_5_flash_lite",
}

LLM_DISPLAY_LABELS = {
    "baseline": "baseline",
    "deepseek_chat": "DeepSeek",
    "gemini_2_5_flash_lite": "Gemini",
}

MODEL_CHART_TITLES = {
    "random_forest": "随机森林模型三组结果对比",
    "linear": "线性回归模型三组结果对比",
    "lightgbm": "LightGBM模型三组结果对比",
    "xgboost": "XGBoost模型三组结果对比",
}

MODEL_CHART_FILES = {
    "random_forest": "random_forest_comparison.png",
    "linear": "linear_comparison.png",
    "lightgbm": "lightgbm_comparison.png",
    "xgboost": "xgboost_comparison.png",
}

CHART_SERIES_COLORS = {
    "benchmark": "#2ca02c",
    "baseline": "#6c757d",
    "deepseek_chat": "#1f77b4",
    "gemini_2_5_flash_lite": "#ff7f0e",
}

SCENARIO_DISPLAY_LABELS = {
    "default": "Top50/Top10",
    "top30_top10": "Top30/Top10",
    "top50_top5": "Top50/Top5",
}

CORE_METRICS = (
    ("cumulative_return", "累计收益率", "percent"),
    ("annual_return", "年化收益率", "percent"),
    ("annual_volatility", "年化波动率", "percent"),
    ("sharpe_ratio", "夏普比率", "float"),
    ("max_drawdown", "最大回撤", "percent"),
    ("win_rate", "胜率", "percent"),
    ("total_periods", "调仓期数", "integer"),
)

NET_METRICS = (
    ("cumulative_return_net", "累计收益率（净）", "percent"),
    ("annual_return_net", "年化收益率（净）", "percent"),
    ("annual_volatility_net", "年化波动率（净）", "percent"),
    ("sharpe_ratio_net", "夏普比率（净）", "float"),
    ("max_drawdown_net", "最大回撤（净）", "percent"),
    ("win_rate_net", "胜率（净）", "percent"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成中文实验汇总页。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(PLANNED_TRADITIONAL_MODELS),
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="需要纳入汇总的传统模型。",
    )
    parser.add_argument(
        "--output-dir",
        default=get_experiment_summary_dir(PROJECT_ROOT),
        help="实验汇总输出目录。",
    )
    return parser.parse_args()


def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
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


def compute_cumulative_return_from_csv(file_path: str, column_name: str) -> Optional[float]:
    if not file_path or not column_name or not os.path.exists(file_path):
        return None
    cumulative = 1.0
    has_value = False
    with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = coerce_number(row.get(column_name))
            if value is None:
                continue
            cumulative *= 1 + value
            has_value = True
    if not has_value:
        return None
    return cumulative - 1


def format_metric(value: Any, value_type: str) -> str:
    number = coerce_number(value)
    if number is None:
        return "N/A"
    if value_type == "percent":
        return f"{number:.2%}"
    if value_type == "integer":
        return str(int(round(number)))
    return f"{number:.4f}"


def html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def iter_performance_reports(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    report_paths: List[str] = []
    for current_root, _, files in os.walk(root_dir):
        if "performance_report.json" in files:
            report_paths.append(os.path.join(current_root, "performance_report.json"))
    return sorted(report_paths)


def infer_variant(report: Dict[str, Any], manifest: Optional[Dict[str, Any]]) -> str:
    parameters = report.get("parameters", {}) if isinstance(report, dict) else {}
    use_llm = bool(parameters.get("use_llm"))
    mock_llm = bool(parameters.get("mock_llm"))
    if manifest:
        use_llm = bool(parameters.get("use_llm", use_llm))
        mock_llm = bool(parameters.get("mock_llm", mock_llm))
    if not use_llm:
        return "baseline"
    if mock_llm:
        return "mock_llm"
    return "llm"


def infer_dimensions_from_path(report_path: str) -> Dict[str, str]:
    normalized_path = os.path.normpath(report_path)
    parts = normalized_path.split(os.sep)
    result = {
        "traditional_model": "",
        "llm_name": "",
        "scenario_name": "",
    }
    if "backtests" in parts:
        index = parts.index("backtests")
        if len(parts) > index + 3:
            result["traditional_model"] = parts[index + 1]
            if parts[index + 2] != "logs":
                result["llm_name"] = parts[index + 2]
            result["scenario_name"] = parts[index + 3]
        return result
    if "exp_main" in parts:
        index = parts.index("exp_main")
        if len(parts) > index + 4:
            result["traditional_model"] = parts[index + 1]
            result["llm_name"] = parts[index + 3]
            result["scenario_name"] = parts[index + 4]
    return result


def normalize_llm_name(raw_llm_name: Any, variant: str, path_dimensions: Dict[str, str]) -> str:
    text = str(raw_llm_name or "").strip()
    if variant == "baseline":
        return "baseline"
    if variant == "mock_llm":
        return "mock_llm"
    if text:
        return normalize_second_stage_llm_name(text)
    if path_dimensions.get("llm_name"):
        return normalize_second_stage_llm_name(path_dimensions["llm_name"])
    return ""


def get_run_display_name(model_name: str, llm_name: str) -> str:
    model_label = get_traditional_model_label(model_name)
    llm_label = LLM_DISPLAY_LABELS.get(llm_name, llm_name)
    return f"{model_label} {llm_label}"


def extract_strategy_metrics(
    report: Dict[str, Any],
    gross_key: str,
    net_key: str,
) -> Dict[str, Optional[float]]:
    gross = report.get(gross_key, {})
    net = report.get(net_key, {})
    selected: Dict[str, Optional[float]] = {}
    for metric_key, _, _ in CORE_METRICS:
        selected[metric_key] = coerce_number(gross.get(metric_key))
    for metric_key, _, _ in NET_METRICS:
        base_metric_key = metric_key.replace("_net", "")
        selected[metric_key] = coerce_number(net.get(base_metric_key))
    return selected


def extract_equal_weight_benchmark_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    parameters = report.get("parameters", {}) if isinstance(report, dict) else {}
    benchmark_output_paths = parameters.get("benchmark_output_paths", {}) if isinstance(parameters, dict) else {}
    benchmark_metrics_paths = parameters.get("benchmark_metrics_paths", {}) if isinstance(parameters, dict) else {}
    benchmark_section = report.get("benchmarks", {}) if isinstance(report, dict) else {}
    equal_weight_section = benchmark_section.get("equal_weight", {}) if isinstance(benchmark_section, dict) else {}

    gross = report.get("performance_benchmark_equal_weight", {}) if isinstance(report, dict) else {}
    net = report.get("performance_benchmark_equal_weight_net", {}) if isinstance(report, dict) else {}
    if isinstance(equal_weight_section, dict):
        gross = equal_weight_section.get("performance", gross)
        net = equal_weight_section.get("performance_net", net)

    benchmark_output_path = str(benchmark_output_paths.get("equal_weight", "")).strip()
    cumulative_return = coerce_number(gross.get("cumulative_return"))
    cumulative_return_net = coerce_number(net.get("cumulative_return"))
    if cumulative_return is None:
        cumulative_return = compute_cumulative_return_from_csv(
            benchmark_output_path,
            "benchmark_equal_weight_return",
        )
    if cumulative_return_net is None:
        cumulative_return_net = compute_cumulative_return_from_csv(
            benchmark_output_path,
            "benchmark_equal_weight_net_return",
        )

    return {
        "benchmark_equal_weight_path": benchmark_output_path,
        "benchmark_equal_weight_metrics_path": str(benchmark_metrics_paths.get("equal_weight", "")).strip(),
        "benchmark_equal_weight_cumulative_return": cumulative_return,
        "benchmark_equal_weight_annual_return": coerce_number(gross.get("annual_return")),
        "benchmark_equal_weight_sharpe_ratio": coerce_number(gross.get("sharpe_ratio")),
        "benchmark_equal_weight_max_drawdown": coerce_number(gross.get("max_drawdown")),
        "benchmark_equal_weight_cumulative_return_net": cumulative_return_net,
        "benchmark_equal_weight_annual_return_net": coerce_number(net.get("annual_return")),
        "benchmark_equal_weight_sharpe_ratio_net": coerce_number(net.get("sharpe_ratio")),
        "benchmark_equal_weight_max_drawdown_net": coerce_number(net.get("max_drawdown")),
    }


def resolve_report_html_path(run_dir: str) -> str:
    preferred = os.path.join(run_dir, "reports", "report_summary.html")
    if os.path.exists(preferred):
        return preferred
    legacy = os.path.join(run_dir, "report_summary.html")
    if os.path.exists(legacy):
        return legacy
    return preferred


def build_run_record(report_path: str, source_type: str) -> Optional[Dict[str, Any]]:
    report = load_json_file(report_path)
    if report is None:
        return None

    run_dir = os.path.dirname(report_path)
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = load_json_file(manifest_path)
    parameters = report.get("parameters", {}) if isinstance(report, dict) else {}
    path_dimensions = infer_dimensions_from_path(report_path)

    variant = infer_variant(report, manifest)
    traditional_model = str(
        (manifest or {}).get("traditional_model")
        or parameters.get("traditional_model")
        or path_dimensions.get("traditional_model")
        or ""
    ).strip().lower()
    if not traditional_model:
        return None

    scenario_name = str(
        (manifest or {}).get("scenario")
        or parameters.get("scenario")
        or path_dimensions.get("scenario_name")
        or DEFAULT_SCENARIO
    ).strip()
    llm_name = normalize_llm_name(
        (manifest or {}).get("second_stage_llm") or parameters.get("second_stage_llm"),
        variant,
        path_dimensions,
    )

    record: Dict[str, Any] = {
        "traditional_model": traditional_model,
        "model_label": get_traditional_model_label(traditional_model),
        "llm_name": llm_name,
        "scenario_name": scenario_name,
        "variant": variant,
        "report_path": resolve_report_html_path(run_dir),
        "performance_report_path": report_path,
        "run_name": get_run_display_name(traditional_model, llm_name if variant != "baseline" else "baseline"),
    }
    if variant == "baseline":
        record.update(extract_strategy_metrics(report, "performance_strategy_a", "performance_strategy_a_net"))
    else:
        record.update(extract_strategy_metrics(report, "performance_strategy_b", "performance_strategy_b_net"))
        record.update(
            {
                f"baseline_snapshot_{metric_key}": metric_value
                for metric_key, metric_value in extract_strategy_metrics(
                    report,
                    "performance_strategy_a",
                    "performance_strategy_a_net",
                ).items()
            }
        )

    backtest_results_path = str(parameters.get("backtest_results_path", "")).strip()
    if coerce_number(record.get("cumulative_return")) is None:
        gross_column = "strategy_a_return" if variant == "baseline" else "strategy_b_return"
        record["cumulative_return"] = compute_cumulative_return_from_csv(backtest_results_path, gross_column)
    if coerce_number(record.get("cumulative_return_net")) is None:
        net_column = "strategy_a_net_return" if variant == "baseline" else "strategy_b_net_return"
        record["cumulative_return_net"] = compute_cumulative_return_from_csv(backtest_results_path, net_column)
    if variant != "baseline":
        if coerce_number(record.get("baseline_snapshot_cumulative_return")) is None:
            record["baseline_snapshot_cumulative_return"] = compute_cumulative_return_from_csv(
                backtest_results_path,
                "strategy_a_return",
            )
        if coerce_number(record.get("baseline_snapshot_cumulative_return_net")) is None:
            record["baseline_snapshot_cumulative_return_net"] = compute_cumulative_return_from_csv(
                backtest_results_path,
                "strategy_a_net_return",
            )
    record.update(extract_equal_weight_benchmark_metrics(report))
    return record


def choose_record(existing: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    existing_mtime = os.path.getmtime(existing["performance_report_path"])
    candidate_mtime = os.path.getmtime(candidate["performance_report_path"])
    if candidate_mtime > existing_mtime:
        return candidate
    if candidate_mtime < existing_mtime:
        return existing
    if candidate.get("variant") == "baseline":
        return candidate
    return existing


def collect_run_records(models: Iterable[str]) -> List[Dict[str, Any]]:
    requested_models = {str(model).strip().lower() for model in models}
    deduped: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    for model_name in requested_models:
        scan_roots = [
            (get_traditional_two_stage_dir(PROJECT_ROOT, model_name), "active"),
            (get_traditional_experiment_runs_dir(PROJECT_ROOT, model_name), "archived"),
        ]
        for root_dir, source_type in scan_roots:
            for report_path in iter_performance_reports(root_dir):
                record = build_run_record(report_path, source_type)
                if record is None:
                    continue
                if record["traditional_model"] not in requested_models:
                    continue
                dedupe_key = (
                    record["traditional_model"],
                    record["scenario_name"],
                    record["llm_name"] if record["variant"] != "baseline" else "baseline",
                )
                if dedupe_key in deduped:
                    deduped[dedupe_key] = choose_record(deduped[dedupe_key], record)
                else:
                    deduped[dedupe_key] = record

    return list(deduped.values())


def build_overview_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    default_rows = [
        dict(row)
        for row in records
        if row["scenario_name"] == DEFAULT_SCENARIO
        and (row["variant"] == "baseline" or row["llm_name"] in {"deepseek_chat", "gemini_2_5_flash_lite"})
    ]

    existing_baselines = {
        row["traditional_model"]
        for row in default_rows
        if row["variant"] == "baseline"
    }

    synthetic_baselines: List[Dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        if model_name in existing_baselines:
            continue
        source_row = next(
            (
                row
                for row in default_rows
                if row["traditional_model"] == model_name and row["variant"] == "llm"
            ),
            None,
        )
        if source_row is None:
            continue
        synthetic_row = {
            key: value
            for key, value in source_row.items()
            if not str(key).startswith("baseline_snapshot_")
        }
        synthetic_row.update(
            {
                "llm_name": "baseline",
                "variant": "baseline",
                "run_name": get_run_display_name(model_name, "baseline"),
            }
        )
        for metric_key, _, _ in CORE_METRICS:
            synthetic_row[metric_key] = source_row.get(f"baseline_snapshot_{metric_key}")
        for metric_key, _, _ in NET_METRICS:
            synthetic_row[metric_key] = source_row.get(f"baseline_snapshot_{metric_key}")
        synthetic_baselines.append(synthetic_row)

    combined_rows = default_rows + synthetic_baselines
    cleaned_rows: List[Dict[str, Any]] = []
    for row in combined_rows:
        cleaned_rows.append(
            {
                key: value
                for key, value in row.items()
                if not str(key).startswith("baseline_snapshot_")
            }
        )

    order_map = {name: index for index, name in enumerate(MODEL_ORDER)}
    llm_order_map = {name: index for index, name in enumerate(RUN_ORDER)}
    return sorted(
        cleaned_rows,
        key=lambda row: (
            order_map.get(row["traditional_model"], 999),
            llm_order_map.get(row["llm_name"], 999),
        ),
    )


def remove_legacy_summary_files(output_dir: str) -> None:
    for file_name in (
        "llm_stability_summary.csv",
        "scenario_stability_summary.csv",
        "paired_comparison.csv",
    ):
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)


def write_csv(file_path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not rows:
        with open(file_path, "w", encoding="utf-8-sig", newline="") as handle:
            handle.write("")
        return
    with open(file_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def relpath_for_output(path: str, output_dir: str) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, output_dir).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def build_markdown_table(rows: List[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows or not columns:
        return "暂无可展示的数据。"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def build_html_table(
    rows: List[Dict[str, Any]],
    columns: Sequence[str],
    output_dir: str,
) -> str:
    if not rows or not columns:
        return '<div class="empty-state">暂无可展示的数据。</div>'
    header_html = "".join(f"<th>{html_escape(column)}</th>" for column in columns)
    body_rows: List[str] = []
    for row in rows:
        cells: List[str] = []
        for column in columns:
            value = row.get(column)
            if column == "报告路径" and value:
                relative_path = relpath_for_output(str(value), output_dir)
                rendered = (
                    f'<a href="{html_escape(relative_path)}" target="_blank" rel="noopener noreferrer">'
                    "打开报告</a>"
                )
            else:
                rendered = html_escape(value)
            cells.append(f"<td>{rendered}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="table-wrapper"><table class="summary-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def build_display_rows(overview_rows: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    _ = output_dir
    display_rows: List[Dict[str, Any]] = []
    for row in overview_rows:
        display_rows.append(
            {
                "运行名称": row["run_name"],
                "场景": row["scenario_name"],
                "累计收益率（净）": format_metric(row.get("cumulative_return_net"), "percent"),
                "年化收益率": format_metric(row.get("annual_return"), "percent"),
                "年化收益率（净）": format_metric(row.get("annual_return_net"), "percent"),
                "夏普比率": format_metric(row.get("sharpe_ratio"), "float"),
                "夏普比率（净）": format_metric(row.get("sharpe_ratio_net"), "float"),
                "最大回撤": format_metric(row.get("max_drawdown"), "percent"),
                "市场基准累计收益率（净）": format_metric(
                    row.get("benchmark_equal_weight_cumulative_return_net"),
                    "percent",
                ),
                "市场基准年化收益率（净）": format_metric(
                    row.get("benchmark_equal_weight_annual_return_net"),
                    "percent",
                ),
                "市场基准夏普（净）": format_metric(
                    row.get("benchmark_equal_weight_sharpe_ratio_net"),
                    "float",
                ),
                "市场基准最大回撤（净）": format_metric(
                    row.get("benchmark_equal_weight_max_drawdown_net"),
                    "percent",
                ),
                "胜率": format_metric(row.get("win_rate"), "percent"),
                "报告路径": row.get("report_path", ""),
            }
        )
    return display_rows


def ensure_chart_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def generate_model_comparison_chart(
    model_name: str,
    rows: List[Dict[str, Any]],
    charts_dir: str,
) -> Optional[str]:
    ordered_rows = sorted(rows, key=lambda row: RUN_ORDER.index(row["llm_name"]))
    if not ordered_rows:
        return None

    labels = [LLM_DISPLAY_LABELS.get(row["llm_name"], row["llm_name"]) for row in ordered_rows]
    annual_return_net = [coerce_number(row.get("annual_return_net")) or 0.0 for row in ordered_rows]
    sharpe_ratio_net = [coerce_number(row.get("sharpe_ratio_net")) or 0.0 for row in ordered_rows]
    max_drawdown = [coerce_number(row.get("max_drawdown")) or 0.0 for row in ordered_rows]
    colors = [CHART_SERIES_COLORS.get(row["llm_name"], "#4c78a8") for row in ordered_rows]

    ensure_chart_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.suptitle(MODEL_CHART_TITLES[model_name], fontsize=14)

    series_specs = (
        (annual_return_net, "年化收益率（净）", "数值"),
        (sharpe_ratio_net, "夏普比率（净）", "数值"),
        (max_drawdown, "最大回撤", "数值"),
    )

    for axis, (values, title, ylabel) in zip(axes, series_specs):
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        for bar, value in zip(bars, values):
            text = f"{value:.2%}" if "收益率" in title or "回撤" in title else f"{value:.2f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                text,
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(charts_dir, exist_ok=True)
    chart_path = os.path.join(charts_dir, MODEL_CHART_FILES[model_name])
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def generate_charts(overview_rows: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, str]]:
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    chart_items: List[Dict[str, str]] = []

    for model_name in MODEL_ORDER:
        model_rows = [
            row
            for row in overview_rows
            if row["traditional_model"] == model_name and row["llm_name"] in RUN_ORDER
        ]
        if not model_rows:
            continue
        chart_path = generate_model_comparison_chart(model_name, model_rows, charts_dir)
        if not chart_path:
            continue
        chart_items.append(
            {
                "model_name": model_name,
                "title": MODEL_CHART_TITLES[model_name],
                "chart_path": chart_path,
            }
        )
    return chart_items


def build_lightgbm_deepseek_robustness_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    scenario_rows: List[Dict[str, Any]] = []
    for scenario_name in ROBUSTNESS_SCENARIOS:
        llm_row = next(
            (
                row
                for row in records
                if row["traditional_model"] == ROBUSTNESS_MODEL
                and row["scenario_name"] == scenario_name
                and row["variant"] == "llm"
                and row["llm_name"] == ROBUSTNESS_LLM
            ),
            None,
        )
        baseline_row = next(
            (
                row
                for row in records
                if row["traditional_model"] == ROBUSTNESS_MODEL
                and row["scenario_name"] == scenario_name
                and row["variant"] == "baseline"
            ),
            None,
        )

        if baseline_row is None and llm_row is not None:
            synthetic_baseline = {
                "traditional_model": ROBUSTNESS_MODEL,
                "model_label": get_traditional_model_label(ROBUSTNESS_MODEL),
                "llm_name": "baseline",
                "scenario_name": scenario_name,
                "variant": "baseline",
                "report_path": llm_row.get("report_path", ""),
                "performance_report_path": llm_row.get("performance_report_path", ""),
                "run_name": f"{get_traditional_model_label(ROBUSTNESS_MODEL)} baseline",
            }
            for metric_key, _, _ in CORE_METRICS:
                synthetic_baseline[metric_key] = llm_row.get(f"baseline_snapshot_{metric_key}")
            for metric_key, _, _ in NET_METRICS:
                synthetic_baseline[metric_key] = llm_row.get(f"baseline_snapshot_{metric_key}")
            baseline_row = synthetic_baseline

        if baseline_row is not None:
            scenario_rows.append(dict(baseline_row))
        if llm_row is not None:
            scenario_rows.append(
                {
                    key: value
                    for key, value in llm_row.items()
                    if not str(key).startswith("baseline_snapshot_")
                }
            )

    return scenario_rows


def build_lightgbm_deepseek_robustness_display_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    display_rows: List[Dict[str, Any]] = []
    for row in rows:
        display_rows.append(
            {
                "参数设定": SCENARIO_DISPLAY_LABELS.get(row["scenario_name"], row["scenario_name"]),
                "方案": "baseline" if row["variant"] == "baseline" else "DeepSeek",
                "年化收益率": format_metric(row.get("annual_return"), "percent"),
                "年化收益率（净）": format_metric(row.get("annual_return_net"), "percent"),
                "夏普比率": format_metric(row.get("sharpe_ratio"), "float"),
                "夏普比率（净）": format_metric(row.get("sharpe_ratio_net"), "float"),
                "最大回撤": format_metric(row.get("max_drawdown"), "percent"),
                "胜率": format_metric(row.get("win_rate"), "percent"),
            }
        )
    return display_rows


def generate_lightgbm_deepseek_robustness_chart(
    rows: List[Dict[str, Any]],
    charts_dir: str,
) -> Optional[str]:
    scenario_buckets: Dict[str, Dict[str, Dict[str, Any]]] = {
        scenario_name: {} for scenario_name in ROBUSTNESS_SCENARIOS
    }
    for row in rows:
        scenario_name = row.get("scenario_name")
        if scenario_name not in scenario_buckets:
            continue
        key = "baseline" if row.get("variant") == "baseline" else "deepseek_chat"
        scenario_buckets[scenario_name][key] = row

    if not any(scenario_buckets[scenario_name] for scenario_name in ROBUSTNESS_SCENARIOS):
        return None

    ensure_chart_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.suptitle("LightGBM 最小稳健性检验对比（DeepSeek）", fontsize=14)

    scenario_labels = [SCENARIO_DISPLAY_LABELS.get(name, name) for name in ROBUSTNESS_SCENARIOS]
    x = list(range(len(ROBUSTNESS_SCENARIOS)))
    width = 0.34
    metric_specs = (
        ("annual_return_net", "年化收益率（净）"),
        ("sharpe_ratio_net", "夏普比率（净）"),
        ("max_drawdown", "最大回撤"),
    )

    for axis, (metric_key, title) in zip(axes, metric_specs):
        baseline_values = []
        deepseek_values = []
        for scenario_name in ROBUSTNESS_SCENARIOS:
            baseline_values.append(
                coerce_number(scenario_buckets[scenario_name].get("baseline", {}).get(metric_key)) or 0.0
            )
            deepseek_values.append(
                coerce_number(scenario_buckets[scenario_name].get("deepseek_chat", {}).get(metric_key)) or 0.0
            )

        baseline_bars = axis.bar(
            [value - width / 2 for value in x],
            baseline_values,
            width,
            label="baseline",
            color=CHART_SERIES_COLORS["baseline"],
        )
        deepseek_bars = axis.bar(
            [value + width / 2 for value in x],
            deepseek_values,
            width,
            label="DeepSeek",
            color=CHART_SERIES_COLORS["deepseek_chat"],
        )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(scenario_labels)
        axis.set_ylabel(title)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        for bars in (baseline_bars, deepseek_bars):
            for bar in bars:
                value = bar.get_height()
                text = f"{value:.2%}" if metric_key in {"annual_return_net", "max_drawdown"} else f"{value:.2f}"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    text,
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8.5,
                )

    axes[0].legend()
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(charts_dir, exist_ok=True)
    chart_path = os.path.join(charts_dir, "lightgbm_deepseek_robustness.png")
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def build_summary_json_payload(
    overview_rows: List[Dict[str, Any]],
    chart_items: List[Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scenario_name": DEFAULT_SCENARIO,
        "overview_rows": overview_rows,
        "chart_items": chart_items,
    }


def build_lightgbm_deepseek_robustness_markdown(
    output_dir: str,
    display_rows: List[Dict[str, Any]],
    chart_path: Optional[str],
    missing_scenarios: List[str],
) -> str:
    lines = [
        "# LightGBM 最小稳健性检验报告（DeepSeek）",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "本报告固定第一阶段模型为 LightGBM，固定第二阶段 LLM 为 DeepSeek，",
        "比较 Top50/Top10、Top30/Top10、Top50/Top5 三种参数设定下 baseline 与 LLM 两阶段策略的表现，",
        "用于观察最小稳健性检验下结果方向是否一致。",
        "",
    ]
    if missing_scenarios:
        missing_text = "、".join(SCENARIO_DISPLAY_LABELS.get(name, name) for name in missing_scenarios)
        lines.extend([f"当前缺失的参数设定：{missing_text}。", ""])

    lines.append(build_markdown_table(display_rows, list(display_rows[0].keys()) if display_rows else []))
    lines.append("")
    if chart_path:
        relative_path = relpath_for_output(chart_path, output_dir)
        lines.extend(
            [
                "**LightGBM 最小稳健性检验对比图（DeepSeek）**",
                "",
                f"![LightGBM 最小稳健性检验对比图（DeepSeek）]({relative_path})",
                "",
            ]
        )
    else:
        lines.extend(["当前可用数据不足，未生成稳健性对比图。", ""])
    return "\n".join(lines)


def build_lightgbm_deepseek_robustness_html(
    output_dir: str,
    display_rows: List[Dict[str, Any]],
    chart_path: Optional[str],
    missing_scenarios: List[str],
) -> str:
    if chart_path:
        chart_html = (
            '<div class="chart-card">'
            "<h2>LightGBM 最小稳健性检验对比图（DeepSeek）</h2>"
            f'<img class="chart-image" src="{html_escape(relpath_for_output(chart_path, output_dir))}" '
            'alt="LightGBM 最小稳健性检验对比图（DeepSeek）">'
            "</div>"
        )
    else:
        chart_html = '<div class="empty-state">当前可用数据不足，未生成稳健性对比图。</div>'

    missing_html = ""
    if missing_scenarios:
        missing_text = "、".join(SCENARIO_DISPLAY_LABELS.get(name, name) for name in missing_scenarios)
        missing_html = f'<div class="empty-state">当前缺失的参数设定：{html_escape(missing_text)}。</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LightGBM 最小稳健性检验报告（DeepSeek）</title>
    <style>
        body {{
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            line-height: 1.6;
            color: #222;
            max-width: 1280px;
            margin: 0 auto;
            padding: 24px;
            background: #f5f7fb;
        }}
        .hero {{
            background: linear-gradient(135deg, #0f4c5c 0%, #2c7a7b 100%);
            color: white;
            padding: 28px;
            border-radius: 12px;
            margin-bottom: 24px;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 30px;
            font-weight: 600;
        }}
        .hero p {{
            margin: 10px 0 0;
            opacity: 0.94;
        }}
        .content {{
            background: white;
            border-radius: 10px;
            padding: 24px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
        }}
        .table-wrapper {{
            overflow-x: auto;
            margin-top: 16px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .summary-table th {{
            background: #0f4c5c;
            color: white;
            text-align: left;
            padding: 12px;
        }}
        .summary-table td {{
            padding: 12px;
            border-bottom: 1px solid #e6edf5;
        }}
        .summary-table tr:nth-child(even) {{
            background: #f8fafc;
        }}
        .chart-card {{
            margin-top: 24px;
            background: #f8fafc;
            border: 1px solid #e6edf5;
            border-radius: 10px;
            padding: 18px;
        }}
        .chart-card h2 {{
            margin-top: 0;
            color: #0f4c5c;
            font-size: 20px;
        }}
        .chart-image {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            background: white;
        }}
        .empty-state {{
            padding: 14px 16px;
            background: #f8fafc;
            border-left: 4px solid #9db4c0;
            color: #4f6470;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>LightGBM 最小稳健性检验报告（DeepSeek）</h1>
        <p>生成时间：{html_escape(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
    </div>
    <div class="content">
        <p>本报告固定第一阶段模型为 LightGBM，固定第二阶段 LLM 为 DeepSeek，比较 Top50/Top10、Top30/Top10、Top50/Top5 三种参数设定下 baseline 与 LLM 两阶段策略的表现，用于观察最小稳健性检验下结果方向是否一致。</p>
        {missing_html}
        {build_html_table(display_rows, list(display_rows[0].keys()) if display_rows else [], output_dir)}
        {chart_html}
    </div>
</body>
</html>"""


def write_lightgbm_deepseek_robustness_outputs(
    output_dir: str,
    records: List[Dict[str, Any]],
) -> Dict[str, str]:
    robustness_rows = build_lightgbm_deepseek_robustness_rows(records)
    display_rows = build_lightgbm_deepseek_robustness_display_rows(robustness_rows)
    charts_dir = os.path.join(output_dir, "charts")
    chart_path = generate_lightgbm_deepseek_robustness_chart(robustness_rows, charts_dir)
    present_scenarios = {row["scenario_name"] for row in robustness_rows}
    missing_scenarios = [name for name in ROBUSTNESS_SCENARIOS if name not in present_scenarios]

    markdown_path = os.path.join(output_dir, "lightgbm_deepseek_robustness.md")
    html_path = os.path.join(output_dir, "lightgbm_deepseek_robustness.html")

    with open(markdown_path, "w", encoding="utf-8") as handle:
        handle.write(
            build_lightgbm_deepseek_robustness_markdown(
                output_dir,
                display_rows,
                chart_path,
                missing_scenarios,
            )
        )
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(
            build_lightgbm_deepseek_robustness_html(
                output_dir,
                display_rows,
                chart_path,
                missing_scenarios,
            )
        )

    outputs = {
        "robustness_markdown": markdown_path,
        "robustness_html": html_path,
    }
    if chart_path:
        outputs["robustness_chart"] = chart_path
    return outputs


def build_markdown_content(
    output_dir: str,
    display_rows: List[Dict[str, Any]],
    chart_items: List[Dict[str, str]],
) -> str:
    lines = [
        "# ETF实验结果总览",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"汇总场景：{DEFAULT_SCENARIO}",
        "",
        "以下总览仅统计 default 场景，并按每个传统模型展示 baseline、DeepSeek、Gemini 三组结果，同时补充 ETF池等权市场基准的净值指标。",
        "",
        build_markdown_table(display_rows, list(display_rows[0].keys()) if display_rows else []),
        "",
    ]
    for chart_item in chart_items:
        relative_path = relpath_for_output(chart_item["chart_path"], output_dir)
        lines.extend(
            [
                f"**{chart_item['title']}**",
                "",
                f"![{chart_item['title']}]({relative_path})",
                "",
            ]
        )
    if not chart_items:
        lines.append("当前没有可展示的模型对比图。")
        lines.append("")
    return "\n".join(lines)


def build_html_content(
    output_dir: str,
    display_rows: List[Dict[str, Any]],
    chart_items: List[Dict[str, str]],
) -> str:
    if chart_items:
        chart_html = "".join(
            (
                '<div class="chart-card">'
                f"<h3>{html_escape(chart_item['title'])}</h3>"
                f'<img class="chart-image" src="{html_escape(relpath_for_output(chart_item["chart_path"], output_dir))}" '
                f'alt="{html_escape(chart_item["title"])}">'
                "</div>"
            )
            for chart_item in chart_items
        )
    else:
        chart_html = '<div class="empty-state">当前没有可展示的模型对比图。</div>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF实验结果总览</title>
    <style>
        body {{
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            line-height: 1.6;
            color: #222;
            max-width: 1380px;
            margin: 0 auto;
            padding: 24px;
            background: #f5f7fb;
        }}
        .hero {{
            background: linear-gradient(135deg, #154360 0%, #2471a3 100%);
            color: white;
            padding: 28px;
            border-radius: 12px;
            margin-bottom: 24px;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 30px;
            font-weight: 600;
        }}
        .hero p {{
            margin: 10px 0 0;
            opacity: 0.92;
        }}
        .content {{
            background: white;
            border-radius: 10px;
            padding: 24px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
        }}
        .content h2 {{
            margin-top: 0;
            color: #154360;
            border-bottom: 2px solid #e6edf5;
            padding-bottom: 10px;
        }}
        .content p {{
            color: #425466;
        }}
        .table-wrapper {{
            overflow-x: auto;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 14px;
        }}
        .summary-table th {{
            background: #154360;
            color: white;
            text-align: left;
            padding: 12px;
        }}
        .summary-table td {{
            padding: 12px;
            border-bottom: 1px solid #e6edf5;
            vertical-align: top;
        }}
        .summary-table tr:nth-child(even) {{
            background: #f8fafc;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-top: 24px;
        }}
        .chart-card {{
            background: #f8fafc;
            border: 1px solid #e6edf5;
            border-radius: 10px;
            padding: 18px;
        }}
        .chart-card h3 {{
            margin-top: 0;
            color: #154360;
            font-size: 18px;
        }}
        .chart-image {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            background: white;
        }}
        .empty-state {{
            padding: 14px 16px;
            background: #f8fafc;
            border-left: 4px solid #9db4c0;
            color: #4f6470;
            margin-top: 20px;
        }}
        a {{
            color: #1d5f8c;
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>ETF实验结果总览</h1>
        <p>生成时间：{html_escape(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
        <p>汇总场景：{html_escape(DEFAULT_SCENARIO)}</p>
    </div>

    <div class="content">
        <h2>总览表</h2>
        <p>以下总览仅统计 default 场景，并按每个传统模型展示 baseline、DeepSeek、Gemini 三组结果，同时补充 ETF池等权市场基准的净值指标。</p>
        {build_html_table(display_rows, list(display_rows[0].keys()) if display_rows else [], output_dir)}
        <div class="charts-grid">
            {chart_html}
        </div>
    </div>
</body>
</html>"""


SUMMARY_MODEL_SECTION_TITLES = {
    "random_forest": "RandomForest",
    "linear": "LinearRegression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}

SUMMARY_MODEL_CHART_TITLES = {
    "random_forest": "RandomForest 不同策略收益对比",
    "linear": "LinearRegression 不同策略收益对比",
    "lightgbm": "LightGBM 不同策略收益对比",
    "xgboost": "XGBoost 不同策略收益对比",
}

SUMMARY_STRATEGY_ORDER = ("benchmark", "baseline", "deepseek_chat", "gemini_2_5_flash_lite")

SUMMARY_STRATEGY_DISPLAY_LABELS = {
    "benchmark": "市场基准",
    "baseline": "传统模型基线",
    "deepseek_chat": "传统模型 + DeepSeek",
    "gemini_2_5_flash_lite": "传统模型 + Gemini",
}

SUMMARY_TABLE_COLUMNS = [
    "策略",
    "累计收益率（净）",
    "年化收益率（净）",
    "夏普比率（净）",
    "最大回撤（净）",
]


def _build_summary_strategy_row(strategy_key: str, model_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = model_rows.get(strategy_key)
    if strategy_key == "benchmark":
        benchmark_source = next(
            (
                candidate
                for candidate in model_rows.values()
                if candidate
                and (
                    coerce_number(candidate.get("benchmark_equal_weight_cumulative_return_net")) is not None
                    or coerce_number(candidate.get("benchmark_equal_weight_annual_return_net")) is not None
                    or coerce_number(candidate.get("benchmark_equal_weight_sharpe_ratio_net")) is not None
                    or coerce_number(candidate.get("benchmark_equal_weight_max_drawdown_net")) is not None
                )
            ),
            None,
        )
        return {
            "策略": SUMMARY_STRATEGY_DISPLAY_LABELS[strategy_key],
            "累计收益率（净）": format_metric(
                None if benchmark_source is None else benchmark_source.get("benchmark_equal_weight_cumulative_return_net"),
                "percent",
            ),
            "年化收益率（净）": format_metric(
                None if benchmark_source is None else benchmark_source.get("benchmark_equal_weight_annual_return_net"),
                "percent",
            ),
            "夏普比率（净）": format_metric(
                None if benchmark_source is None else benchmark_source.get("benchmark_equal_weight_sharpe_ratio_net"),
                "float",
            ),
            "最大回撤（净）": format_metric(
                None if benchmark_source is None else benchmark_source.get("benchmark_equal_weight_max_drawdown_net"),
                "percent",
            ),
        }

    return {
        "策略": SUMMARY_STRATEGY_DISPLAY_LABELS[strategy_key],
        "累计收益率（净）": format_metric(None if row is None else row.get("cumulative_return_net"), "percent"),
        "年化收益率（净）": format_metric(None if row is None else row.get("annual_return_net"), "percent"),
        "夏普比率（净）": format_metric(None if row is None else row.get("sharpe_ratio_net"), "float"),
        "最大回撤（净）": format_metric(None if row is None else row.get("max_drawdown_net"), "percent"),
    }


def build_display_rows(overview_rows: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    _ = output_dir
    sections: List[Dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        model_candidates = [
            row
            for row in overview_rows
            if row.get("traditional_model") == model_name and row.get("scenario_name") == DEFAULT_SCENARIO
        ]
        if not model_candidates:
            continue

        model_rows = {
            row["llm_name"]: row
            for row in model_candidates
            if row.get("llm_name") in RUN_ORDER
        }
        section_rows = [
            _build_summary_strategy_row(strategy_key, model_rows)
            for strategy_key in SUMMARY_STRATEGY_ORDER
        ]
        sections.append(
            {
                "model_name": model_name,
                "title": SUMMARY_MODEL_SECTION_TITLES.get(model_name, get_traditional_model_label(model_name)),
                "rows": section_rows,
            }
        )
    return sections


def generate_model_comparison_chart(
    model_name: str,
    rows: List[Dict[str, Any]],
    charts_dir: str,
) -> Optional[str]:
    model_rows = {
        row["llm_name"]: row
        for row in rows
        if row.get("llm_name") in RUN_ORDER
    }
    benchmark_source = next(
        (
            candidate
            for candidate in model_rows.values()
            if (
                coerce_number(candidate.get("benchmark_equal_weight_annual_return_net")) is not None
                or coerce_number(candidate.get("benchmark_equal_weight_cumulative_return_net")) is not None
            )
        ),
        None,
    )
    if not model_rows and benchmark_source is None:
        return None

    labels = [SUMMARY_STRATEGY_DISPLAY_LABELS[key] for key in SUMMARY_STRATEGY_ORDER]
    values = []
    colors = []
    for strategy_key in SUMMARY_STRATEGY_ORDER:
        if strategy_key == "benchmark":
            value = None if benchmark_source is None else benchmark_source.get("benchmark_equal_weight_annual_return_net")
        else:
            value = None if model_rows.get(strategy_key) is None else model_rows[strategy_key].get("annual_return_net")
        values.append(coerce_number(value) or 0.0)
        colors.append(CHART_SERIES_COLORS.get(strategy_key, "#4c78a8"))

    ensure_chart_style()
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    bars = axis.bar(labels, values, color=colors, width=0.62)
    axis.set_title(SUMMARY_MODEL_CHART_TITLES.get(model_name, model_name), fontsize=14)
    axis.set_ylabel("年化收益率（净）")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.set_axisbelow(True)

    upper = max(values) if values else 0.0
    lower = min(values) if values else 0.0
    padding = max((upper - lower) * 0.15, 0.02)
    axis.set_ylim(lower - padding, upper + padding)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2%}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    os.makedirs(charts_dir, exist_ok=True)
    chart_path = os.path.join(charts_dir, MODEL_CHART_FILES[model_name])
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def generate_charts(overview_rows: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, str]]:
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    chart_items: List[Dict[str, str]] = []

    for model_name in MODEL_ORDER:
        model_rows = [
            row
            for row in overview_rows
            if row.get("traditional_model") == model_name
            and row.get("scenario_name") == DEFAULT_SCENARIO
            and row.get("llm_name") in RUN_ORDER
        ]
        if not model_rows:
            continue
        chart_path = generate_model_comparison_chart(model_name, model_rows, charts_dir)
        if not chart_path:
            continue
        chart_items.append(
            {
                "model_name": model_name,
                "title": SUMMARY_MODEL_CHART_TITLES.get(model_name, model_name),
                "chart_path": chart_path,
            }
        )
    return chart_items


def build_markdown_content(
    output_dir: str,
    display_rows: List[Dict[str, Any]],
    chart_items: List[Dict[str, str]],
) -> str:
    chart_lookup = {item["model_name"]: item for item in chart_items}
    lines = [
        "# ETF实验汇总",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"汇总场景：{DEFAULT_SCENARIO}",
        "",
        "以下结果仅展示 default 场景，并按模型分别比较市场基准、传统模型基线、传统模型 + DeepSeek、传统模型 + Gemini 四种方案的净值指标。",
        "",
    ]

    if not display_rows:
        lines.extend(["当前没有可展示的汇总结果。", ""])
        return "\n".join(lines)

    for section in display_rows:
        lines.extend(
            [
                f"## {section['title']}",
                "",
                build_markdown_table(section["rows"], SUMMARY_TABLE_COLUMNS),
                "",
            ]
        )
        chart_item = chart_lookup.get(section["model_name"])
        if chart_item:
            relative_path = relpath_for_output(chart_item["chart_path"], output_dir)
            lines.extend(
                [
                    f"![{chart_item['title']}]({relative_path})",
                    "",
                ]
            )

    return "\n".join(lines)


def build_html_content(
    output_dir: str,
    display_rows: List[Dict[str, Any]],
    chart_items: List[Dict[str, str]],
) -> str:
    chart_lookup = {item["model_name"]: item for item in chart_items}
    sections_html: List[str] = []

    for section in display_rows:
        chart_item = chart_lookup.get(section["model_name"])
        if chart_item:
            chart_block = (
                '<div class="chart-card">'
                f'<img class="chart-image" src="{html_escape(relpath_for_output(chart_item["chart_path"], output_dir))}" '
                f'alt="{html_escape(chart_item["title"])}">'
                "</div>"
            )
        else:
            chart_block = '<div class="empty-state">当前没有可展示的图表。</div>'

        sections_html.append(
            (
                '<section class="model-section">'
                f"<h2>{html_escape(section['title'])}</h2>"
                f"{build_html_table(section['rows'], SUMMARY_TABLE_COLUMNS, output_dir)}"
                f"{chart_block}"
                "</section>"
            )
        )

    if not sections_html:
        sections_html.append('<div class="empty-state">当前没有可展示的汇总结果。</div>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF实验汇总</title>
    <style>
        body {{
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            line-height: 1.6;
            color: #222;
            max-width: 1280px;
            margin: 0 auto;
            padding: 24px;
            background: #f5f7fb;
        }}
        .hero {{
            background: linear-gradient(135deg, #154360 0%, #2471a3 100%);
            color: white;
            padding: 28px;
            border-radius: 12px;
            margin-bottom: 24px;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 30px;
            font-weight: 600;
        }}
        .hero p {{
            margin: 10px 0 0;
            opacity: 0.92;
        }}
        .content {{
            background: white;
            border-radius: 10px;
            padding: 24px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
        }}
        .content p {{
            color: #425466;
            margin-top: 0;
        }}
        .model-section {{
            padding-top: 8px;
            margin-top: 28px;
            border-top: 1px solid #e6edf5;
        }}
        .model-section:first-of-type {{
            margin-top: 0;
            padding-top: 0;
            border-top: none;
        }}
        .model-section h2 {{
            margin: 0 0 14px;
            color: #154360;
            font-size: 24px;
        }}
        .table-wrapper {{
            overflow-x: auto;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
        }}
        .summary-table th {{
            background: #154360;
            color: white;
            text-align: left;
            padding: 12px;
        }}
        .summary-table td {{
            padding: 12px;
            border-bottom: 1px solid #e6edf5;
            vertical-align: top;
        }}
        .summary-table tr:nth-child(even) {{
            background: #f8fafc;
        }}
        .chart-card {{
            margin-top: 18px;
            background: #f8fafc;
            border: 1px solid #e6edf5;
            border-radius: 10px;
            padding: 16px;
        }}
        .chart-image {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            background: white;
        }}
        .empty-state {{
            padding: 14px 16px;
            background: #f8fafc;
            border-left: 4px solid #9db4c0;
            color: #4f6470;
            margin-top: 18px;
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>ETF实验汇总</h1>
        <p>生成时间：{html_escape(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
        <p>汇总场景：{html_escape(DEFAULT_SCENARIO)}</p>
    </div>

    <div class="content">
        <p>以下结果仅展示 default 场景，并按模型分别比较市场基准、传统模型基线、传统模型 + DeepSeek、传统模型 + Gemini 四种方案的净值指标。</p>
        {''.join(sections_html)}
    </div>
</body>
</html>"""


MARKET_BENCHMARK_KEY = "hs300_etf"
MARKET_BENCHMARK_NAME = "沪深300ETF"
MARKET_BENCHMARK_CODE = "510300"

SUMMARY_MODEL_CHART_TITLES = {
    "random_forest": "RandomForest 不同策略收益对比",
    "linear": "LinearRegression 不同策略收益对比",
    "lightgbm": "LightGBM 不同策略收益对比",
    "xgboost": "XGBoost 不同策略收益对比",
}

SUMMARY_STRATEGY_DISPLAY_LABELS = {
    "benchmark": "市场基准（510300 沪深300ETF）",
    "baseline": "传统模型基线",
    "deepseek_chat": "传统模型 + DeepSeek",
    "gemini_2_5_flash_lite": "传统模型 + Gemini",
}

SUMMARY_TABLE_COLUMNS = [
    "策略",
    "累计收益率（净）",
    "年化收益率（净）",
    "夏普比率（净）",
    "最大回撤（净）",
]


def extract_equal_weight_benchmark_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    parameters = report.get("parameters", {}) if isinstance(report, dict) else {}
    benchmark_output_paths = parameters.get("benchmark_output_paths", {}) if isinstance(parameters, dict) else {}
    benchmark_metrics_paths = parameters.get("benchmark_metrics_paths", {}) if isinstance(parameters, dict) else {}
    benchmark_section = report.get("benchmarks", {}) if isinstance(report, dict) else {}
    market_section = benchmark_section.get(MARKET_BENCHMARK_KEY, {}) if isinstance(benchmark_section, dict) else {}
    equal_weight_section = benchmark_section.get("equal_weight", {}) if isinstance(benchmark_section, dict) else {}

    market_gross = report.get("performance_market_benchmark", {}) if isinstance(report, dict) else {}
    market_net = report.get("performance_market_benchmark_net", {}) if isinstance(report, dict) else {}
    if isinstance(market_section, dict):
        market_gross = market_section.get("performance", market_gross)
        market_net = market_section.get("performance_net", market_net)

    market_output_path = str(benchmark_output_paths.get(MARKET_BENCHMARK_KEY, "")).strip()
    market_cumulative_return = coerce_number(market_gross.get("cumulative_return"))
    market_cumulative_return_net = coerce_number(market_net.get("cumulative_return"))
    if market_cumulative_return is None:
        market_cumulative_return = compute_cumulative_return_from_csv(
            market_output_path,
            "benchmark_market_return",
        )
    if market_cumulative_return_net is None:
        market_cumulative_return_net = compute_cumulative_return_from_csv(
            market_output_path,
            "benchmark_market_net_return",
        )

    gross = report.get("performance_benchmark_equal_weight", {}) if isinstance(report, dict) else {}
    net = report.get("performance_benchmark_equal_weight_net", {}) if isinstance(report, dict) else {}
    if isinstance(equal_weight_section, dict):
        gross = equal_weight_section.get("performance", gross)
        net = equal_weight_section.get("performance_net", net)

    benchmark_output_path = str(benchmark_output_paths.get("equal_weight", "")).strip()
    cumulative_return = coerce_number(gross.get("cumulative_return"))
    cumulative_return_net = coerce_number(net.get("cumulative_return"))
    if cumulative_return is None:
        cumulative_return = compute_cumulative_return_from_csv(
            benchmark_output_path,
            "benchmark_equal_weight_return",
        )
    if cumulative_return_net is None:
        cumulative_return_net = compute_cumulative_return_from_csv(
            benchmark_output_path,
            "benchmark_equal_weight_net_return",
        )

    return {
        "market_benchmark_name": str(parameters.get("benchmark_name") or MARKET_BENCHMARK_NAME),
        "market_benchmark_code": str(parameters.get("benchmark_code") or MARKET_BENCHMARK_CODE),
        "market_benchmark_path": market_output_path,
        "market_benchmark_metrics_path": str(benchmark_metrics_paths.get(MARKET_BENCHMARK_KEY, "")).strip(),
        "market_benchmark_cumulative_return": market_cumulative_return,
        "market_benchmark_annual_return": coerce_number(market_gross.get("annual_return")),
        "market_benchmark_sharpe_ratio": coerce_number(market_gross.get("sharpe_ratio")),
        "market_benchmark_max_drawdown": coerce_number(market_gross.get("max_drawdown")),
        "market_benchmark_cumulative_return_net": market_cumulative_return_net,
        "market_benchmark_annual_return_net": coerce_number(market_net.get("annual_return")),
        "market_benchmark_sharpe_ratio_net": coerce_number(market_net.get("sharpe_ratio")),
        "market_benchmark_max_drawdown_net": coerce_number(market_net.get("max_drawdown")),
        "benchmark_equal_weight_path": benchmark_output_path,
        "benchmark_equal_weight_metrics_path": str(benchmark_metrics_paths.get("equal_weight", "")).strip(),
        "benchmark_equal_weight_cumulative_return": cumulative_return,
        "benchmark_equal_weight_annual_return": coerce_number(gross.get("annual_return")),
        "benchmark_equal_weight_sharpe_ratio": coerce_number(gross.get("sharpe_ratio")),
        "benchmark_equal_weight_max_drawdown": coerce_number(gross.get("max_drawdown")),
        "benchmark_equal_weight_cumulative_return_net": cumulative_return_net,
        "benchmark_equal_weight_annual_return_net": coerce_number(net.get("annual_return")),
        "benchmark_equal_weight_sharpe_ratio_net": coerce_number(net.get("sharpe_ratio")),
        "benchmark_equal_weight_max_drawdown_net": coerce_number(net.get("max_drawdown")),
    }


def _build_summary_strategy_row(strategy_key: str, model_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row = model_rows.get(strategy_key)
    if strategy_key == "benchmark":
        benchmark_source = next(
            (
                candidate
                for candidate in model_rows.values()
                if candidate
                and (
                    coerce_number(candidate.get("market_benchmark_cumulative_return_net")) is not None
                    or coerce_number(candidate.get("market_benchmark_annual_return_net")) is not None
                    or coerce_number(candidate.get("market_benchmark_sharpe_ratio_net")) is not None
                    or coerce_number(candidate.get("market_benchmark_max_drawdown_net")) is not None
                )
            ),
            None,
        )
        return {
            "策略": SUMMARY_STRATEGY_DISPLAY_LABELS[strategy_key],
            "累计收益率（净）": format_metric(
                None if benchmark_source is None else benchmark_source.get("market_benchmark_cumulative_return_net"),
                "percent",
            ),
            "年化收益率（净）": format_metric(
                None if benchmark_source is None else benchmark_source.get("market_benchmark_annual_return_net"),
                "percent",
            ),
            "夏普比率（净）": format_metric(
                None if benchmark_source is None else benchmark_source.get("market_benchmark_sharpe_ratio_net"),
                "float",
            ),
            "最大回撤（净）": format_metric(
                None if benchmark_source is None else benchmark_source.get("market_benchmark_max_drawdown_net"),
                "percent",
            ),
        }

    return {
        "策略": SUMMARY_STRATEGY_DISPLAY_LABELS[strategy_key],
        "累计收益率（净）": format_metric(None if row is None else row.get("cumulative_return_net"), "percent"),
        "年化收益率（净）": format_metric(None if row is None else row.get("annual_return_net"), "percent"),
        "夏普比率（净）": format_metric(None if row is None else row.get("sharpe_ratio_net"), "float"),
        "最大回撤（净）": format_metric(None if row is None else row.get("max_drawdown_net"), "percent"),
    }


def build_display_rows(overview_rows: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    _ = output_dir
    sections: List[Dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        model_candidates = [
            row
            for row in overview_rows
            if row.get("traditional_model") == model_name and row.get("scenario_name") == DEFAULT_SCENARIO
        ]
        if not model_candidates:
            continue

        model_rows = {
            row["llm_name"]: row
            for row in model_candidates
            if row.get("llm_name") in RUN_ORDER
        }
        section_rows = [_build_summary_strategy_row(strategy_key, model_rows) for strategy_key in SUMMARY_STRATEGY_ORDER]
        sections.append(
            {
                "model_name": model_name,
                "title": SUMMARY_MODEL_SECTION_TITLES.get(model_name, get_traditional_model_label(model_name)),
                "rows": section_rows,
            }
        )
    return sections


def generate_model_comparison_chart(
    model_name: str,
    rows: List[Dict[str, Any]],
    charts_dir: str,
) -> Optional[str]:
    model_rows = {
        row["llm_name"]: row
        for row in rows
        if row.get("llm_name") in RUN_ORDER
    }
    benchmark_source = next(
        (
            candidate
            for candidate in model_rows.values()
            if (
                coerce_number(candidate.get("market_benchmark_annual_return_net")) is not None
                or coerce_number(candidate.get("market_benchmark_cumulative_return_net")) is not None
            )
        ),
        None,
    )
    if not model_rows and benchmark_source is None:
        return None

    labels = [SUMMARY_STRATEGY_DISPLAY_LABELS[key] for key in SUMMARY_STRATEGY_ORDER]
    values = []
    colors = []
    for strategy_key in SUMMARY_STRATEGY_ORDER:
        if strategy_key == "benchmark":
            value = None if benchmark_source is None else benchmark_source.get("market_benchmark_annual_return_net")
        else:
            value = None if model_rows.get(strategy_key) is None else model_rows[strategy_key].get("annual_return_net")
        values.append(coerce_number(value) or 0.0)
        colors.append(CHART_SERIES_COLORS.get(strategy_key, "#4c78a8"))

    ensure_chart_style()
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    bars = axis.bar(labels, values, color=colors, width=0.62)
    axis.set_title(SUMMARY_MODEL_CHART_TITLES.get(model_name, model_name), fontsize=14)
    axis.set_ylabel("年化收益率（净）")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.set_axisbelow(True)

    upper = max(values) if values else 0.0
    lower = min(values) if values else 0.0
    padding = max((upper - lower) * 0.15, 0.02)
    axis.set_ylim(lower - padding, upper + padding)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2%}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    os.makedirs(charts_dir, exist_ok=True)
    chart_path = os.path.join(charts_dir, MODEL_CHART_FILES[model_name])
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def build_markdown_content(
    output_dir: str,
    display_rows: List[Dict[str, Any]],
    chart_items: List[Dict[str, str]],
) -> str:
    chart_lookup = {item["model_name"]: item for item in chart_items}
    lines = [
        "# ETF实验汇总",
        "",
        f"生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"汇总场景：{DEFAULT_SCENARIO}",
        "",
        "市场基准为 510300 沪深300ETF。以下结果仅展示 default 场景，并按模型分别比较市场基准、传统模型基线、传统模型 + DeepSeek、传统模型 + Gemini 四种方案的净值指标。",
        "",
    ]

    if not display_rows:
        lines.extend(["当前没有可展示的汇总结果。", ""])
        return "\n".join(lines)

    for section in display_rows:
        lines.extend(
            [
                f"## {section['title']}",
                "",
                build_markdown_table(section["rows"], SUMMARY_TABLE_COLUMNS),
                "",
            ]
        )
        chart_item = chart_lookup.get(section["model_name"])
        if chart_item:
            relative_path = relpath_for_output(chart_item["chart_path"], output_dir)
            lines.extend([f"![{chart_item['title']}]({relative_path})", ""])

    return "\n".join(lines)


def build_html_content(
    output_dir: str,
    display_rows: List[Dict[str, Any]],
    chart_items: List[Dict[str, str]],
) -> str:
    chart_lookup = {item["model_name"]: item for item in chart_items}
    sections_html: List[str] = []

    for section in display_rows:
        chart_item = chart_lookup.get(section["model_name"])
        if chart_item:
            chart_block = (
                '<div class="chart-card">'
                f'<img class="chart-image" src="{html_escape(relpath_for_output(chart_item["chart_path"], output_dir))}" '
                f'alt="{html_escape(chart_item["title"])}">'
                "</div>"
            )
        else:
            chart_block = '<div class="empty-state">当前没有可展示的图表。</div>'

        sections_html.append(
            (
                '<section class="model-section">'
                f"<h2>{html_escape(section['title'])}</h2>"
                f"{build_html_table(section['rows'], SUMMARY_TABLE_COLUMNS, output_dir)}"
                f"{chart_block}"
                "</section>"
            )
        )

    if not sections_html:
        sections_html.append('<div class="empty-state">当前没有可展示的汇总结果。</div>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF实验汇总</title>
    <style>
        body {{
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            line-height: 1.6;
            color: #222;
            max-width: 1280px;
            margin: 0 auto;
            padding: 24px;
            background: #f5f7fb;
        }}
        .hero {{
            background: linear-gradient(135deg, #154360 0%, #2471a3 100%);
            color: white;
            padding: 28px;
            border-radius: 12px;
            margin-bottom: 24px;
        }}
        .hero h1 {{
            margin: 0;
            font-size: 30px;
            font-weight: 600;
        }}
        .hero p {{
            margin: 10px 0 0;
            opacity: 0.92;
        }}
        .content {{
            background: white;
            border-radius: 10px;
            padding: 24px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.06);
        }}
        .content p {{
            color: #425466;
            margin-top: 0;
        }}
        .model-section {{
            padding-top: 8px;
            margin-top: 28px;
            border-top: 1px solid #e6edf5;
        }}
        .model-section:first-of-type {{
            margin-top: 0;
            padding-top: 0;
            border-top: none;
        }}
        .model-section h2 {{
            margin: 0 0 14px;
            color: #154360;
            font-size: 24px;
        }}
        .table-wrapper {{
            overflow-x: auto;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
        }}
        .summary-table th {{
            background: #154360;
            color: white;
            text-align: left;
            padding: 12px;
        }}
        .summary-table td {{
            padding: 12px;
            border-bottom: 1px solid #e6edf5;
            vertical-align: top;
        }}
        .summary-table tr:nth-child(even) {{
            background: #f8fafc;
        }}
        .chart-card {{
            margin-top: 18px;
            background: #f8fafc;
            border: 1px solid #e6edf5;
            border-radius: 10px;
            padding: 16px;
        }}
        .chart-image {{
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
            background: white;
        }}
        .empty-state {{
            padding: 14px 16px;
            background: #f8fafc;
            border-left: 4px solid #9db4c0;
            color: #4f6470;
            margin-top: 18px;
        }}
    </style>
</head>
<body>
    <div class="hero">
        <h1>ETF实验汇总</h1>
        <p>生成时间：{html_escape(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
        <p>汇总场景：{html_escape(DEFAULT_SCENARIO)}</p>
    </div>

    <div class="content">
        <p>市场基准为 510300 沪深300ETF。以下结果仅展示 default 场景，并按模型分别比较市场基准、传统模型基线、传统模型 + DeepSeek、传统模型 + Gemini 四种方案的净值指标。</p>
        {''.join(sections_html)}
    </div>
</body>
</html>"""


def write_summary_outputs(output_dir: str, overview_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    remove_legacy_summary_files(output_dir)

    chart_items = generate_charts(overview_rows, output_dir)
    display_rows = build_display_rows(overview_rows, output_dir)

    overview_csv = os.path.join(output_dir, "experiment_overview.csv")
    summary_json = os.path.join(output_dir, "experiment_summary.json")
    summary_markdown = os.path.join(output_dir, "experiment_summary.md")
    summary_html = os.path.join(output_dir, "experiment_summary.html")

    write_csv(overview_csv, overview_rows)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(
            build_summary_json_payload(overview_rows, chart_items),
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with open(summary_markdown, "w", encoding="utf-8") as handle:
        handle.write(build_markdown_content(output_dir, display_rows, chart_items))
    with open(summary_html, "w", encoding="utf-8") as handle:
        handle.write(build_html_content(output_dir, display_rows, chart_items))

    return {
        "overview_csv": overview_csv,
        "summary_json": summary_json,
        "markdown": summary_markdown,
        "html": summary_html,
        "charts_dir": os.path.join(output_dir, "charts"),
    }


def generate_experiment_summary(models: Iterable[str], output_dir: str) -> Dict[str, str]:
    records = collect_run_records(models)
    overview_rows = build_overview_rows(records)
    output_paths = write_summary_outputs(output_dir, overview_rows)
    output_paths.update(write_lightgbm_deepseek_robustness_outputs(output_dir, records))
    return output_paths


def main() -> None:
    args = parse_args()
    output_paths = generate_experiment_summary(args.models, args.output_dir)
    print("实验汇总已生成：")
    for label, path in output_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()

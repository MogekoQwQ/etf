"""Plot net cumulative return curves from existing backtest results."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from traditional_model_config import (
    PLANNED_TRADITIONAL_MODELS,
    get_traditional_model_label,
    get_traditional_two_stage_output_dir,
)


matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_SCENARIO = "default"
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "net_return_curves")

DATE_COLUMN_CANDIDATES = ("rebalance_date", "date", "trade_date", "调仓日期")
GENERIC_NET_VALUE_COLUMNS = (
    "net_value",
    "strategy_net_value",
    "portfolio_value_net",
    "cumulative_nav_net",
)
GENERIC_NET_CUM_RETURN_COLUMNS = (
    "net_cum_return",
    "cumulative_return_net",
    "cum_return_net",
    "累计收益率（净）",
)
GENERIC_PERIOD_NET_RETURN_COLUMNS = (
    "period_return_net",
    "net_return",
    "weekly_return_net",
    "调仓收益率（净）",
)

STRATEGY_STYLES = {
    "baseline": {"label": "baseline", "color": "#4c78a8"},
    "deepseek": {"label": "DeepSeek", "color": "#f58518"},
    "gemini": {"label": "Gemini", "color": "#54a24b"},
}
MODEL_FILE_NAMES = {
    "random_forest": "random_forest_net_cum_curve.png",
    "linear": "linear_net_cum_curve.png",
    "lightgbm": "lightgbm_net_cum_curve.png",
    "xgboost": "xgboost_net_cum_curve.png",
}
LLM_DIR_MAPPING = {
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.5-flash-lite",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制净累计收益曲线图。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(PLANNED_TRADITIONAL_MODELS),
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="需要绘图的传统模型列表。",
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help="需要处理的scenario，默认仅处理 default。",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录。",
    )
    return parser.parse_args()


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("-", "_")


def resolve_exact_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized_map = {normalize_text(column): column for column in columns}
    for candidate in candidates:
        matched = normalized_map.get(normalize_text(candidate))
        if matched:
            return matched
    return None


def find_backtest_sources(model_name: str, scenario: str) -> Dict[str, str]:
    sources: Dict[str, str] = {}
    for strategy, llm_name in LLM_DIR_MAPPING.items():
        run_dir = get_traditional_two_stage_output_dir(
            PROJECT_ROOT,
            model_name,
            llm_name,
            50,
            10,
        ) if scenario == "default" else os.path.join(
            PROJECT_ROOT,
            "runs",
            "backtests",
            model_name,
            normalize_text(llm_name),
            scenario,
        )
        result_path = os.path.join(run_dir, "backtest_results.csv")
        if os.path.exists(result_path):
            sources[strategy] = result_path

    if "deepseek" in sources:
        sources["baseline"] = sources["deepseek"]
    elif "gemini" in sources:
        sources["baseline"] = sources["gemini"]

    required = {"baseline", "deepseek", "gemini"}
    missing = [name for name in required if name not in sources]
    if missing:
        raise FileNotFoundError(
            f"Missing backtest sources for `{model_name}` under scenario `{scenario}`: {missing}"
        )
    return sources


def infer_date_column(df: pd.DataFrame) -> str:
    date_column = resolve_exact_column(df.columns, DATE_COLUMN_CANDIDATES)
    if not date_column:
        raise ValueError(
            f"Failed to infer date column. Supported names: {', '.join(DATE_COLUMN_CANDIDATES)}"
        )
    return date_column


def get_strategy_specific_candidates(strategy: str) -> Dict[str, List[str]]:
    if strategy == "baseline":
        prefix_candidates = {
            "net_value": ["strategy_a_net_value", "baseline_net_value"],
            "net_cum_return": ["strategy_a_net_cum_return", "baseline_net_cum_return"],
            "period_net_return": ["strategy_a_net_return", "baseline_net_return"],
        }
    else:
        prefix_candidates = {
            "net_value": ["strategy_b_net_value", f"{strategy}_net_value"],
            "net_cum_return": ["strategy_b_net_cum_return", f"{strategy}_net_cum_return"],
            "period_net_return": ["strategy_b_net_return", f"{strategy}_net_return"],
        }
    return prefix_candidates


def find_first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_strategy_curve(source_path: str, strategy: str) -> pd.DataFrame:
    df = pd.read_csv(source_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Backtest result is empty: {source_path}")

    date_column = infer_date_column(df)
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column]).copy()
    if df.empty:
        raise ValueError(f"Backtest result has no valid dates: {source_path}")

    candidates = get_strategy_specific_candidates(strategy)
    net_value_column = find_first_existing_column(df.columns, [*candidates["net_value"], *GENERIC_NET_VALUE_COLUMNS])
    net_cum_return_column = find_first_existing_column(
        df.columns,
        [*candidates["net_cum_return"], *GENERIC_NET_CUM_RETURN_COLUMNS],
    )
    period_net_return_column = find_first_existing_column(
        df.columns,
        [*candidates["period_net_return"], *GENERIC_PERIOD_NET_RETURN_COLUMNS],
    )

    curve_df = pd.DataFrame({"date": df[date_column].dt.normalize()})
    if net_value_column:
        curve_df["net_value"] = pd.to_numeric(df[net_value_column], errors="coerce")
        curve_df["net_cum_return"] = curve_df["net_value"] - 1.0
    elif net_cum_return_column:
        curve_df["net_cum_return"] = pd.to_numeric(df[net_cum_return_column], errors="coerce")
        curve_df["net_value"] = curve_df["net_cum_return"] + 1.0
    elif period_net_return_column:
        period_net_return = pd.to_numeric(df[period_net_return_column], errors="coerce")
        curve_df["period_net_return"] = period_net_return
        curve_df["net_value"] = (1.0 + period_net_return).cumprod()
        curve_df["net_cum_return"] = curve_df["net_value"] - 1.0
    else:
        raise ValueError(
            "Failed to infer net curve data columns from "
            f"{source_path}. Checked net_value / net_cum_return / period_net_return priorities."
        )

    return normalize_curve_dataframe(curve_df)


def normalize_curve_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized = normalized.dropna(subset=["date", "net_value", "net_cum_return"]).copy()
    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if normalized.empty:
        raise ValueError("Curve dataframe is empty after normalization.")
    return normalized[["date", "net_value", "net_cum_return"]]


def align_curves_by_date(curves: Dict[str, pd.DataFrame]) -> tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    date_sets = {strategy: set(df["date"]) for strategy, df in curves.items()}
    common_dates = set.intersection(*date_sets.values()) if date_sets else set()
    if not common_dates:
        raise ValueError("No common rebalance dates across baseline / DeepSeek / Gemini curves.")

    aligned_curves: Dict[str, pd.DataFrame] = {}
    common_dates_sorted = sorted(common_dates)
    for strategy, df in curves.items():
        aligned = df[df["date"].isin(common_dates_sorted)].copy().sort_values("date").reset_index(drop=True)
        aligned_curves[strategy] = aligned

    alignment_info = {
        "original_counts": {strategy: int(len(df)) for strategy, df in curves.items()},
        "aligned_count": int(len(common_dates_sorted)),
        "aligned_start": pd.Timestamp(common_dates_sorted[0]).strftime("%Y-%m-%d"),
        "aligned_end": pd.Timestamp(common_dates_sorted[-1]).strftime("%Y-%m-%d"),
    }
    return aligned_curves, alignment_info


def plot_model_curves(
    model_name: str,
    aligned_curves: Dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    plt.figure(figsize=(10, 5.6))
    for strategy in ("baseline", "deepseek", "gemini"):
        curve_df = aligned_curves[strategy]
        plt.plot(
            curve_df["date"],
            curve_df["net_cum_return"],
            label=STRATEGY_STYLES[strategy]["label"],
            color=STRATEGY_STYLES[strategy]["color"],
            linewidth=2.0,
        )

    plt.title(f"{get_traditional_model_label(model_name)}净累计收益曲线对比")
    plt.xlabel("调仓日期")
    plt.ylabel("净累计收益率")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.25)

    axis = plt.gca()
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_curve_rows(model_name: str, aligned_curves: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for strategy, curve_df in aligned_curves.items():
        for _, row in curve_df.iterrows():
            rows.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "net_value": float(row["net_value"]),
                    "net_cum_return": float(row["net_cum_return"]),
                }
            )
    return rows


def write_curve_summary(report_path: str, model_summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "# 净累计收益曲线图简报",
        "",
        "本次基于现有回测结果补生成净累计收益曲线图。横轴为调仓日期，纵轴为净累计收益率。",
        "每个传统模型单独输出一张图，图中包含 baseline、DeepSeek、Gemini 三条曲线。",
        "",
        "## 曲线图生成情况",
        "",
    ]

    for summary in model_summaries:
        original_counts = summary["alignment"]["original_counts"]
        lines.append(
            f"- {summary['model_label']}：输出 `{summary['image_name']}`；"
            f"baseline / DeepSeek / Gemini 原始日期数分别为 "
            f"{original_counts['baseline']} / {original_counts['deepseek']} / {original_counts['gemini']}，"
            f"最终按交集对齐为 {summary['alignment']['aligned_count']} 个调仓日期，"
            f"区间为 {summary['alignment']['aligned_start']} 至 {summary['alignment']['aligned_end']}。"
        )

    lines.extend(
        [
            "",
            "若某些方案原始日期存在差异，脚本按三条曲线的日期交集进行对齐，不进行插值或伪造缺失收益。",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    all_curve_rows: List[Dict[str, Any]] = []
    model_summaries: List[Dict[str, Any]] = []

    for model_name in args.models:
        print(f"Plotting net cumulative return curves for {model_name}...")
        source_paths = find_backtest_sources(model_name, args.scenario)
        curves = {
            strategy: load_strategy_curve(source_paths[strategy], strategy)
            for strategy in ("baseline", "deepseek", "gemini")
        }
        aligned_curves, alignment_info = align_curves_by_date(curves)

        image_name = MODEL_FILE_NAMES[model_name]
        output_path = os.path.join(output_dir, image_name)
        plot_model_curves(model_name, aligned_curves, output_path)

        all_curve_rows.extend(build_curve_rows(model_name, aligned_curves))
        model_summaries.append(
            {
                "model": model_name,
                "model_label": get_traditional_model_label(model_name),
                "image_name": image_name,
                "alignment": alignment_info,
                "source_paths": source_paths,
            }
        )

    curves_df = pd.DataFrame(all_curve_rows)
    curves_output_path = os.path.join(output_dir, "net_return_curves_by_date.csv")
    curves_df.to_csv(curves_output_path, index=False, encoding="utf-8-sig")

    report_path = os.path.join(output_dir, "net_return_curves_report.md")
    write_curve_summary(report_path, model_summaries)

    print("Net cumulative return curve plots completed.")
    print(f"- curves_by_date: {curves_output_path}")
    for summary in model_summaries:
        print(f"- {summary['model']}: {os.path.join(output_dir, summary['image_name'])}")
    print(f"- report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

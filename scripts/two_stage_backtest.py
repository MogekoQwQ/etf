"""Compare stage-one traditional ranking against two-stage ranking."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from traditional_model_config import (
    DEFAULT_TRADITIONAL_MODEL,
    PLANNED_TRADITIONAL_MODELS,
    RANDOM_FOREST_PARAMS,
    TARGET_COLUMNS,
    ensure_implemented_traditional_model,
    get_traditional_explanation_dir,
    get_traditional_llm_log_dir,
    get_traditional_prediction_file,
    get_traditional_two_stage_dir,
)


warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "all_etf_factors.csv")

TRADITIONAL_MODEL = DEFAULT_TRADITIONAL_MODEL
PREDICTIONS_PATH = get_traditional_prediction_file(PROJECT_ROOT, TRADITIONAL_MODEL)
OUTPUT_DIR = get_traditional_two_stage_dir(PROJECT_ROOT, TRADITIONAL_MODEL)
LLM_LOG_DIR = get_traditional_llm_log_dir(PROJECT_ROOT, TRADITIONAL_MODEL)
EXPLANATION_DIR = get_traditional_explanation_dir(PROJECT_ROOT, TRADITIONAL_MODEL)

TARGET = "Y_future_5d_return"
PRED_COL = f"y_pred_{TARGET}"
TOP_N_FIRST = 50
TOP_N_FINAL = 10
REBALANCING_FREQ = "W"
HOLDING_DAYS = 5

USE_LLM = True
MOCK_LLM = False
ENABLE_EXPLANATIONS = False
EXPLANATION_DATE: Optional[str] = None
EXPLANATION_SAMPLE_SIZE = 3
SELECTED_EXPLANATION_DATES: Set[str] = set()
LLM_TIMEOUT: Optional[int] = None


def configure_paths(traditional_model: str) -> None:
    global TRADITIONAL_MODEL, PREDICTIONS_PATH, OUTPUT_DIR, LLM_LOG_DIR, EXPLANATION_DIR

    TRADITIONAL_MODEL = traditional_model
    PREDICTIONS_PATH = get_traditional_prediction_file(PROJECT_ROOT, traditional_model)
    OUTPUT_DIR = get_traditional_two_stage_dir(PROJECT_ROOT, traditional_model)
    LLM_LOG_DIR = get_traditional_llm_log_dir(PROJECT_ROOT, traditional_model)
    EXPLANATION_DIR = get_traditional_explanation_dir(PROJECT_ROOT, traditional_model)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF 两阶段策略回测对比")
    parser.add_argument(
        "--traditional-model",
        type=str,
        default=DEFAULT_TRADITIONAL_MODEL,
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="第一阶段传统模型名称。当前仅 random_forest 已实现。",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET,
        choices=["Y_next_day_return", "Y_future_5d_return", "Y_future_10d_return"],
        help="预测目标变量",
    )
    parser.add_argument(
        "--rebalancing-freq",
        type=str,
        default=REBALANCING_FREQ,
        choices=["D", "W", "M"],
        help="调仓频率",
    )
    parser.add_argument("--top-n-first", type=int, default=TOP_N_FIRST, help="第一阶段候选数量")
    parser.add_argument("--top-n-final", type=int, default=TOP_N_FINAL, help="最终组合数量")
    parser.add_argument("--no-llm", action="store_true", help="不使用大语言模型进行二次排序")
    parser.add_argument("--mock-llm", action="store_true", help="使用本地 mock 排序逻辑")
    parser.add_argument("--enable-explanations", action="store_true", help="启用展示型解释功能")
    parser.add_argument("--explanation-date", type=str, default=None, help="仅对指定调仓日生成解释")
    parser.add_argument(
        "--explanation-sample-size",
        type=int,
        default=3,
        help="启用解释但未指定 explanation-date 时的稳定抽样日期数量",
    )
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM API timeout in seconds")
    return parser.parse_args()


def load_data() -> pd.DataFrame:
    if os.path.exists(PREDICTIONS_PATH):
        print(f"读取已保存的传统模型预测文件: {PREDICTIONS_PATH}")
        df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["日期"])
        if PRED_COL not in df.columns:
            print(f"预测文件缺少列 {PRED_COL}，将回退到脚本内重新训练当前已实现基线。")
            df = pd.read_csv(DATA_PATH, parse_dates=["日期"])
            df = run_traditional_prediction(df)
    else:
        print(f"预测文件不存在: {PREDICTIONS_PATH}")
        print("将读取原始因子数据，并回退到脚本内重新生成当前目标的预测列。")
        df = pd.read_csv(DATA_PATH, parse_dates=["日期"])
        df = run_traditional_prediction(df)

    df["code"] = df["code"].astype(str)
    return df


def run_traditional_prediction(df: pd.DataFrame) -> pd.DataFrame:
    implemented_model = ensure_implemented_traditional_model(TRADITIONAL_MODEL)
    if implemented_model != "random_forest":
        raise NotImplementedError(f"尚未实现的传统模型回退预测逻辑: {implemented_model}")

    exclude_cols = ["code", "name", "日期", *TARGET_COLUMNS]
    feature_cols = [column for column in df.columns if column not in exclude_cols]
    split_date = df["日期"].quantile(0.8)
    train_df = df[df["日期"] <= split_date].copy()
    test_df = df[df["日期"] > split_date].copy()

    estimator = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    estimator.fit(train_df[feature_cols], train_df[TARGET])

    df.loc[df["日期"] > split_date, PRED_COL] = estimator.predict(test_df[feature_cols])
    df.loc[df["日期"] <= split_date, PRED_COL] = np.nan
    return df


def get_rebalancing_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    dates = pd.Series(pd.to_datetime(df["日期"].dropna().unique())).sort_values().reset_index(drop=True)
    if REBALANCING_FREQ == "W":
        rebalance_dates = dates.groupby(dates.dt.to_period("W-FRI")).max()
    elif REBALANCING_FREQ == "M":
        rebalance_dates = dates.groupby(dates.dt.to_period("M")).max()
    else:
        rebalance_dates = dates
    return pd.DatetimeIndex(pd.Series(rebalance_dates).dropna().unique())


def format_rebalance_date(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def select_explanation_dates(rebalance_dates: pd.DatetimeIndex) -> Set[str]:
    if not ENABLE_EXPLANATIONS:
        return set()

    available_dates = [format_rebalance_date(date) for date in rebalance_dates]
    if not available_dates:
        return set()

    if EXPLANATION_DATE:
        if EXPLANATION_DATE in available_dates:
            return {EXPLANATION_DATE}
        print(f"指定 explanation-date {EXPLANATION_DATE} 不在调仓日列表中，本次不生成解释。")
        return set()

    sample_size = min(EXPLANATION_SAMPLE_SIZE, len(available_dates))
    if sample_size <= 0:
        return set()

    rng = np.random.default_rng(42)
    selected_idx = np.sort(rng.choice(len(available_dates), size=sample_size, replace=False))
    return {available_dates[idx] for idx in selected_idx}


def execute_strategy_a(df: pd.DataFrame, date: pd.Timestamp) -> List[str]:
    day_data = df[df["日期"] == date].copy()
    day_data = day_data[day_data[PRED_COL].notna()]
    if len(day_data) < TOP_N_FINAL:
        return []
    return day_data.sort_values(PRED_COL, ascending=False).head(TOP_N_FINAL)["code"].tolist()


def execute_strategy_b(df: pd.DataFrame, date: pd.Timestamp, generate_explanation: bool = False) -> Dict[str, Any]:
    day_data = df[df["日期"] == date].copy()
    day_data = day_data[day_data[PRED_COL].notna()]

    if len(day_data) < TOP_N_FIRST:
        return {
            "selected_codes": [],
            "ranked_etfs": None,
            "used_llm_ranking": False,
            "fallback_used": False,
            "explanation_generated": False,
        }

    top_etfs = day_data.sort_values(PRED_COL, ascending=False).head(TOP_N_FIRST).copy()
    ranked_etfs = top_etfs.copy()
    ranked_etfs["llm_score"] = ranked_etfs[PRED_COL]
    used_llm_ranking = False
    fallback_used = False
    explanation_generated = False

    if USE_LLM:
        try:
            from llm_ranking import generate_explanations_for_date, rank_etfs_by_llm

            llm_ranked = rank_etfs_by_llm(
                top_etfs,
                format_rebalance_date(date),
                log_dir=LLM_LOG_DIR,
                enable_explanations=False,
                mock=MOCK_LLM,
                score_reference_col=PRED_COL,
                api_timeout=LLM_TIMEOUT,
            )
            if llm_ranked is None:
                fallback_used = True
                print("  LLM 排序失败，回退到第一阶段传统模型排序结果。")
            else:
                ranked_etfs = llm_ranked
                used_llm_ranking = True

                if generate_explanation:
                    try:
                        explanation_result = generate_explanations_for_date(
                            ranked_etfs.head(TOP_N_FINAL).copy(),
                            format_rebalance_date(date),
                            log_dir=LLM_LOG_DIR,
                            mock=MOCK_LLM,
                            api_timeout=LLM_TIMEOUT,
                        )
                        explanation_generated = explanation_result is not None
                        if not explanation_generated:
                            print(f"  解释生成失败，已跳过 {format_rebalance_date(date)}")
                    except Exception as exc:
                        print(f"  解释生成异常: {exc}，已跳过 {format_rebalance_date(date)}")
        except Exception as exc:
            fallback_used = True
            print(f"  LLM 排序异常: {exc}，回退到第一阶段传统模型排序结果。")

    selected_codes = ranked_etfs.head(TOP_N_FINAL)["code"].tolist()
    return {
        "selected_codes": selected_codes,
        "ranked_etfs": ranked_etfs,
        "used_llm_ranking": used_llm_ranking,
        "fallback_used": fallback_used,
        "explanation_generated": explanation_generated,
    }


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    rebalance_dates = get_rebalancing_dates(df)
    print(f"共 {len(rebalance_dates)} 个调仓日")

    global SELECTED_EXPLANATION_DATES
    SELECTED_EXPLANATION_DATES = select_explanation_dates(rebalance_dates)
    if ENABLE_EXPLANATIONS:
        if SELECTED_EXPLANATION_DATES:
            print(f"本次仅对以下调仓日生成解释: {', '.join(sorted(SELECTED_EXPLANATION_DATES))}")
        else:
            print("本次未命中任何解释日期，不生成解释。")

    results = []
    for index, date in enumerate(rebalance_dates):
        if index % 10 == 0:
            print(f"处理调仓日 {index + 1}/{len(rebalance_dates)}: {date.date()}")

        codes_a = execute_strategy_a(df, date)
        should_explain = format_rebalance_date(date) in SELECTED_EXPLANATION_DATES
        strategy_b_result = execute_strategy_b(df, date, generate_explanation=should_explain)
        codes_b = strategy_b_result["selected_codes"]
        if not codes_a or not codes_b:
            continue

        day_data = df[df["日期"] == date].copy()
        returns_a = [day_data[day_data["code"] == code][TARGET].values[0] for code in codes_a if len(day_data[day_data["code"] == code]) > 0]
        returns_b = [day_data[day_data["code"] == code][TARGET].values[0] for code in codes_b if len(day_data[day_data["code"] == code]) > 0]
        if not returns_a or not returns_b:
            continue

        results.append(
            {
                "rebalance_date": date,
                "strategy_a_codes": codes_a,
                "strategy_b_codes": codes_b,
                "strategy_a_return": float(np.mean(returns_a)),
                "strategy_b_return": float(np.mean(returns_b)),
                "strategy_a_count": len(codes_a),
                "strategy_b_count": len(codes_b),
                "llm_ranking_used": strategy_b_result["used_llm_ranking"],
                "llm_fallback_used": strategy_b_result["fallback_used"],
                "explanation_requested": should_explain,
                "explanation_generated": strategy_b_result["explanation_generated"],
            }
        )

    return pd.DataFrame(results)


def calculate_performance(returns: pd.Series, holding_days: Optional[int] = None) -> Dict[str, float]:
    if len(returns) == 0:
        return {}

    if holding_days is None:
        holding_days = HOLDING_DAYS

    periods_per_year = 252 / holding_days
    mean_return = returns.mean()
    annual_return = (1 + mean_return) ** periods_per_year - 1
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else np.nan
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1
    max_drawdown = drawdown.min()
    win_rate = (returns > 0).mean()

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_periods": len(returns),
        "mean_return": mean_return,
        "std_return": returns.std(),
    }


def plot_results(results_df: pd.DataFrame, performance_a: Dict[str, float], performance_b: Dict[str, float]) -> None:
    cumulative_a = (1 + results_df["strategy_a_return"]).cumprod()
    cumulative_b = (1 + results_df["strategy_b_return"]).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(results_df["rebalance_date"], cumulative_a, label="策略A（仅第一阶段传统模型）", linewidth=2)
    plt.plot(results_df["rebalance_date"], cumulative_b, label="策略B（两阶段）", linewidth=2)
    plt.xlabel("调仓日期")
    plt.ylabel("累计收益")
    plt.title("两阶段策略 vs 单阶段传统模型策略 累计收益对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_returns.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(results_df["strategy_a_return"], bins=30, alpha=0.7, label="策略A")
    plt.xlabel("持有期收益")
    plt.ylabel("频次")
    plt.title("策略A收益分布")
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.hist(results_df["strategy_b_return"], bins=30, alpha=0.7, label="策略B", color="orange")
    plt.xlabel("持有期收益")
    plt.ylabel("频次")
    plt.title("策略B收益分布")
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "returns_distribution.png"), dpi=300)
    plt.close()

    metrics_df = pd.DataFrame(
        {
            "策略A（仅第一阶段传统模型）": [
                f"{performance_a.get('annual_return', 0):.2%}",
                f"{performance_a.get('annual_volatility', 0):.2%}",
                f"{performance_a.get('sharpe_ratio', 0):.2f}",
                f"{performance_a.get('max_drawdown', 0):.2%}",
                f"{performance_a.get('win_rate', 0):.2%}",
                f"{performance_a.get('total_periods', 0)}",
            ],
            "策略B（两阶段）": [
                f"{performance_b.get('annual_return', 0):.2%}",
                f"{performance_b.get('annual_volatility', 0):.2%}",
                f"{performance_b.get('sharpe_ratio', 0):.2f}",
                f"{performance_b.get('max_drawdown', 0):.2%}",
                f"{performance_b.get('win_rate', 0):.2%}",
                f"{performance_b.get('total_periods', 0)}",
            ],
        },
        index=["年化收益率", "年化波动率", "夏普比率", "最大回撤", "胜率", "调仓周期数"],
    )
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "performance_metrics.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].bar(["策略A", "策略B"], [performance_a.get("annual_return", 0), performance_b.get("annual_return", 0)])
    axes[0, 0].set_title("年化收益率")
    axes[0, 1].bar(["策略A", "策略B"], [performance_a.get("sharpe_ratio", 0), performance_b.get("sharpe_ratio", 0)])
    axes[0, 1].set_title("夏普比率")
    axes[1, 0].bar(["策略A", "策略B"], [performance_a.get("max_drawdown", 0), performance_b.get("max_drawdown", 0)])
    axes[1, 0].set_title("最大回撤")
    axes[1, 1].bar(["策略A", "策略B"], [performance_a.get("win_rate", 0), performance_b.get("win_rate", 0)])
    axes[1, 1].set_title("胜率")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "performance_comparison.png"), dpi=300)
    plt.close()


def generate_explanation_reports() -> None:
    if not os.path.isdir(EXPLANATION_DIR):
        print("未发现解释目录，跳过解释报告生成。")
        return

    try:
        from explanation_reporter import ExplanationReporter
    except ImportError as exc:
        print(f"警告: 无法导入 explanation_reporter: {exc}")
        return

    reporter = ExplanationReporter(EXPLANATION_DIR)
    available_dates = reporter.list_available_dates()
    if not available_dates:
        print("未发现实际解释结果，跳过解释报告生成。")
        return

    print("\n生成解释报告...")
    for date_str in available_dates:
        try:
            report_path = reporter.generate_report_for_date(date_str)
            print(f"  [OK] {date_str}: {report_path}")
        except Exception as exc:
            print(f"  [FAIL] {date_str}: {exc}")

    try:
        summary_path = reporter.generate_summary_report()
        print(f"解释汇总报告: {summary_path}")
    except Exception as exc:
        print(f"解释汇总报告生成失败: {exc}")


def main() -> None:
    args = parse_args()
    if args.llm_timeout is not None and args.llm_timeout <= 0:
        raise ValueError("--llm-timeout must be a positive integer")
    if args.explanation_sample_size <= 0:
        raise ValueError("--explanation-sample-size must be a positive integer")

    traditional_model = ensure_implemented_traditional_model(args.traditional_model)
    configure_paths(traditional_model)

    global TARGET, PRED_COL, TOP_N_FIRST, TOP_N_FINAL
    global REBALANCING_FREQ, USE_LLM, MOCK_LLM, ENABLE_EXPLANATIONS
    global EXPLANATION_DATE, EXPLANATION_SAMPLE_SIZE, LLM_TIMEOUT

    TARGET = args.target
    PRED_COL = f"y_pred_{TARGET}"
    TOP_N_FIRST = args.top_n_first
    TOP_N_FINAL = args.top_n_final
    REBALANCING_FREQ = args.rebalancing_freq
    USE_LLM = not args.no_llm
    MOCK_LLM = args.mock_llm and USE_LLM
    ENABLE_EXPLANATIONS = args.enable_explanations and USE_LLM
    EXPLANATION_DATE = args.explanation_date
    EXPLANATION_SAMPLE_SIZE = args.explanation_sample_size
    LLM_TIMEOUT = args.llm_timeout

    if args.enable_explanations and not USE_LLM:
        print("提示: 已启用 explanations，但当前使用 --no-llm，解释功能不会执行。")
    if args.mock_llm and not USE_LLM:
        print("提示: 指定了 --mock-llm，但同时禁用了 LLM 排序，mock 排序不会生效。")

    print("=" * 60)
    print("ETF 两阶段策略回测对比")
    print("=" * 60)
    print(f"第一阶段传统模型: {traditional_model}")
    print(f"预测目标: {TARGET}")
    print(f"调仓频率: {REBALANCING_FREQ}")
    print(f"第一阶段候选数量: {TOP_N_FIRST}")
    print(f"最终组合数量: {TOP_N_FINAL}")
    print(f"使用 LLM 排序: {USE_LLM}")
    print(f"Mock LLM: {MOCK_LLM}")
    print(f"启用解释: {ENABLE_EXPLANATIONS}")
    print(f"结果输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    df = load_data()
    print(f"数据加载完成，时间范围: {df['日期'].min().date()} 到 {df['日期'].max().date()}")
    print(f"ETF 数量: {df['code'].nunique()}")
    print(f"样本总数: {len(df)}")

    results_df = run_backtest(df)
    if results_df.empty:
        print("回测未产生有效结果，请检查参数和数据。")
        return

    performance_a = calculate_performance(results_df["strategy_a_return"], holding_days=HOLDING_DAYS)
    performance_b = calculate_performance(results_df["strategy_b_return"], holding_days=HOLDING_DAYS)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("绩效指标对比")
    print("=" * 60)
    print(f"{'指标':<20} {'策略A':<20} {'策略B':<20}")
    print("-" * 60)
    print(f"{'年化收益率':<20} {performance_a.get('annual_return', 0):.2%} {performance_b.get('annual_return', 0):.2%}")
    print(f"{'年化波动率':<20} {performance_a.get('annual_volatility', 0):.2%} {performance_b.get('annual_volatility', 0):.2%}")
    print(f"{'夏普比率':<20} {performance_a.get('sharpe_ratio', 0):.2f} {performance_b.get('sharpe_ratio', 0):.2f}")
    print(f"{'最大回撤':<20} {performance_a.get('max_drawdown', 0):.2%} {performance_b.get('max_drawdown', 0):.2%}")
    print(f"{'胜率':<20} {performance_a.get('win_rate', 0):.2%} {performance_b.get('win_rate', 0):.2%}")
    print(f"{'调仓周期数':<20} {performance_a.get('total_periods', 0):<20} {performance_b.get('total_periods', 0):<20}")

    plot_results(results_df, performance_a, performance_b)

    generated_explanation_dates = sorted(
        results_df.loc[results_df["explanation_generated"], "rebalance_date"].dt.strftime("%Y-%m-%d").tolist()
    )
    report = {
        "parameters": {
            "traditional_model": traditional_model,
            "target": TARGET,
            "top_n_first": TOP_N_FIRST,
            "top_n_final": TOP_N_FINAL,
            "rebalancing_freq": REBALANCING_FREQ,
            "holding_days": HOLDING_DAYS,
            "use_llm": USE_LLM,
            "mock_llm": MOCK_LLM,
            "enable_explanations": ENABLE_EXPLANATIONS,
            "explanation_date": EXPLANATION_DATE,
            "explanation_sample_size": EXPLANATION_SAMPLE_SIZE,
            "selected_explanation_dates": sorted(SELECTED_EXPLANATION_DATES),
            "generated_explanation_dates": generated_explanation_dates,
            "generated_explanations_count": len(generated_explanation_dates),
            "llm_timeout": LLM_TIMEOUT,
            "predictions_path": PREDICTIONS_PATH,
            "output_dir": OUTPUT_DIR,
        },
        "performance_strategy_a": performance_a,
        "performance_strategy_b": performance_b,
    }

    report_path = os.path.join(OUTPUT_DIR, "performance_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\n详细绩效报告已保存至: {report_path}")

    if ENABLE_EXPLANATIONS:
        generate_explanation_reports()


if __name__ == "__main__":
    main()

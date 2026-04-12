"""Compare stage-one traditional ranking against two-stage ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from market_data_utils import read_csv_with_fallback, standardize_date_column_name
from traditional_model_config import (
    DEFAULT_TRADITIONAL_MODEL,
    DEFAULT_TOP_N_FINAL,
    DEFAULT_TOP_N_FIRST,
    PLANNED_TRADITIONAL_MODELS,
    TARGET_COLUMNS,
    build_traditional_estimator,
    ensure_implemented_traditional_model,
    get_all_factors_file,
    get_backtest_scenario_name,
    get_llm_config_local_file,
    get_traditional_benchmark_dir,
    get_traditional_explanation_dir,
    get_traditional_llm_log_dir,
    get_traditional_prediction_file,
    get_shared_benchmark_dir,
    get_traditional_two_stage_dir,
    get_traditional_two_stage_manifest_file,
    get_traditional_two_stage_output_dir,
)


warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = get_all_factors_file(PROJECT_ROOT)
DATE_COLUMN = "日期"

TRADITIONAL_MODEL = DEFAULT_TRADITIONAL_MODEL
PREDICTIONS_PATH = get_traditional_prediction_file(PROJECT_ROOT, TRADITIONAL_MODEL)
OUTPUT_DIR = get_traditional_two_stage_output_dir(
    PROJECT_ROOT,
    TRADITIONAL_MODEL,
    "deepseek-chat",
    DEFAULT_TOP_N_FIRST,
    DEFAULT_TOP_N_FINAL,
)
LLM_LOG_DIR = get_traditional_llm_log_dir(PROJECT_ROOT, TRADITIONAL_MODEL, "deepseek-chat")
EXPLANATION_DIR = get_traditional_explanation_dir(PROJECT_ROOT, TRADITIONAL_MODEL, "deepseek-chat")

TARGET = "Y_future_5d_return"
PRED_COL = f"y_pred_{TARGET}"
TOP_N_FIRST = 50
TOP_N_FINAL = 10
REBALANCING_FREQ = "W"
HOLDING_DAYS = 5
BENCHMARK_FACTOR_CANDIDATES = ("momentum_20", "return_mean_20")
MARKET_BENCHMARK_KEY = "hs300_etf"
MARKET_BENCHMARK_CODE = "510300"
MARKET_BENCHMARK_NAME = "沪深300ETF"
TRANSACTION_COST_RATE = 0.001
SLIPPAGE_RATE = 0.0005
DATE_COLUMN_CANDIDATES = ("日期", "date", "交易日期", "时间")

USE_LLM = True
MOCK_LLM = False
ENABLE_EXPLANATIONS = False
EXPLANATION_DATE: Optional[str] = None
EXPLANATION_SAMPLE_SIZE = 3
SELECTED_EXPLANATION_DATES: Set[str] = set()
LLM_TIMEOUT: Optional[int] = None
SIMPLE_RULE_BENCHMARK_FACTOR: Optional[str] = None
SECOND_STAGE_LLM = "deepseek-chat"
LLM_CONFIG_PATH = get_llm_config_local_file(PROJECT_ROOT)
BENCHMARK_PERFORMANCE_CACHE: Dict[str, Dict[str, float]] = {}
BENCHMARK_OUTPUT_CACHE: Dict[str, Any] = {}


def to_project_relative_path(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path_obj)


def convert_paths_to_relative(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: convert_paths_to_relative(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [convert_paths_to_relative(value) for value in payload]
    if isinstance(payload, Path):
        return to_project_relative_path(payload)
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return payload
        try:
            candidate = Path(text)
            if candidate.is_absolute():
                return to_project_relative_path(candidate)
        except (OSError, ValueError):
            return payload
    return payload


def get_strategy_display_names() -> tuple[str, str]:
    baseline_label = "传统模型基线策略"
    if MOCK_LLM:
        return baseline_label, "Mock 两阶段验证策略"
    if USE_LLM:
        return baseline_label, "LLM 两阶段策略"
    return baseline_label, "两阶段流程策略（未启用 LLM）"


def get_chart_two_stage_label() -> str:
    if MOCK_LLM:
        return "Mock 两阶段验证策略"
    if USE_LLM:
        return "LLM 两阶段策略"
    return "两阶段流程策略"


def configure_paths(traditional_model: str) -> None:
    global TRADITIONAL_MODEL, PREDICTIONS_PATH, OUTPUT_DIR, LLM_LOG_DIR, EXPLANATION_DIR

    TRADITIONAL_MODEL = traditional_model
    PREDICTIONS_PATH = get_traditional_prediction_file(PROJECT_ROOT, traditional_model)
    OUTPUT_DIR = get_traditional_two_stage_output_dir(
        PROJECT_ROOT,
        traditional_model,
        SECOND_STAGE_LLM,
        TOP_N_FIRST,
        TOP_N_FINAL,
    )
    LLM_LOG_DIR = get_traditional_llm_log_dir(PROJECT_ROOT, traditional_model, SECOND_STAGE_LLM)
    EXPLANATION_DIR = get_traditional_explanation_dir(PROJECT_ROOT, traditional_model, SECOND_STAGE_LLM)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 ETF 两阶段回测对比。")
    parser.add_argument(
        "--traditional-model",
        type=str,
        default=DEFAULT_TRADITIONAL_MODEL,
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="第一阶段传统模型名称：random_forest、linear、lightgbm、xgboost。",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=TARGET,
        choices=["Y_next_day_return", "Y_future_5d_return", "Y_future_10d_return"],
        help="预测目标列。",
    )
    parser.add_argument(
        "--rebalancing-freq",
        type=str,
        default=REBALANCING_FREQ,
        choices=["D", "W", "M"],
        help="调仓频率。",
    )
    parser.add_argument("--top-n-first", type=int, default=TOP_N_FIRST, help="第一阶段候选数量。")
    parser.add_argument("--top-n-final", type=int, default=TOP_N_FINAL, help="最终组合数量。")
    parser.add_argument("--no-llm", action="store_true", help="禁用 LLM 两阶段排序。")
    parser.add_argument("--mock-llm", action="store_true", help="使用本地 Mock 两阶段排序。")
    parser.add_argument("--enable-explanations", action="store_true", help="生成解释结果。")
    parser.add_argument("--explanation-date", type=str, default=None, help="仅对指定日期生成解释。")
    parser.add_argument(
        "--explanation-sample-size",
        type=int,
        default=3,
        help="未提供 --explanation-date 时，抽样生成解释的日期数量。",
    )
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM API 超时时间（秒）。")
    parser.add_argument(
        "--second-stage-llm",
        type=str,
        default=SECOND_STAGE_LLM,
        choices=["deepseek-chat", "gemini-2.5-flash-lite"],
        help="Second-stage LLM model.",
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=LLM_CONFIG_PATH,
        help="Path to the local LLM config yaml file.",
    )
    return parser.parse_args()


def load_data() -> pd.DataFrame:
    if PREDICTIONS_PATH.exists():
        print(f"Loading saved traditional model predictions: {PREDICTIONS_PATH}")
        df = read_csv_with_fallback(PREDICTIONS_PATH)
        df = standardize_date_column_name(df)
        df["日期"] = pd.to_datetime(df["日期"])
        if PRED_COL not in df.columns:
            print(f"Prediction column {PRED_COL} is missing. Regenerating predictions inside backtest script.")
            df = read_csv_with_fallback(DATA_PATH)
            df = standardize_date_column_name(df)
            df["日期"] = pd.to_datetime(df["日期"])
            df = run_traditional_prediction(df)
    else:
        print(f"Prediction file not found: {PREDICTIONS_PATH}")
        print("Loading raw factor data and generating fallback predictions with the selected model.")
        df = read_csv_with_fallback(DATA_PATH)
        df = standardize_date_column_name(df)
        df["日期"] = pd.to_datetime(df["日期"])
        df = run_traditional_prediction(df)

    df["code"] = df["code"].astype(str)
    return df


def run_traditional_prediction(df: pd.DataFrame) -> pd.DataFrame:
    implemented_model = ensure_implemented_traditional_model(TRADITIONAL_MODEL)

    exclude_cols = ["code", "name", "日期", *TARGET_COLUMNS]
    feature_cols = [column for column in df.columns if column not in exclude_cols]
    split_date = df["日期"].quantile(0.8)
    train_df = df[df["日期"] <= split_date].copy()
    test_df = df[df["日期"] > split_date].copy()

    estimator = build_traditional_estimator(implemented_model)
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
        print(f"Specified explanation date {EXPLANATION_DATE} is not in the rebalancing calendar. Skipping.")
        return set()

    sample_size = min(EXPLANATION_SAMPLE_SIZE, len(available_dates))
    if sample_size <= 0:
        return set()

    rng = np.random.default_rng(42)
    selected_idx = np.sort(rng.choice(len(available_dates), size=sample_size, replace=False))
    return {available_dates[idx] for idx in selected_idx}


def get_tradeable_day_data(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    day_data = df[df[DATE_COLUMN] == date].copy()
    day_data = day_data[day_data[TARGET].notna()]
    day_data = day_data[day_data[PRED_COL].notna()]
    day_data["code"] = day_data["code"].astype(str)
    return day_data


def resolve_simple_rule_benchmark_factor(df: pd.DataFrame) -> Optional[str]:
    for factor in BENCHMARK_FACTOR_CANDIDATES:
        if factor in df.columns and pd.api.types.is_numeric_dtype(df[factor]):
            return factor
    return None


def execute_equal_weight_benchmark(day_data: pd.DataFrame) -> List[str]:
    if day_data.empty:
        return []
    return sorted(day_data["code"].astype(str).unique().tolist())


def execute_market_benchmark(day_data: pd.DataFrame) -> List[str]:
    benchmark_data = day_data[day_data["code"].astype(str) == MARKET_BENCHMARK_CODE]
    if benchmark_data.empty:
        return []
    return [MARKET_BENCHMARK_CODE]


def execute_simple_rule_benchmark(day_data: pd.DataFrame) -> List[str]:
    if SIMPLE_RULE_BENCHMARK_FACTOR is None:
        return []
    ranked_day_data = day_data.dropna(subset=[SIMPLE_RULE_BENCHMARK_FACTOR])
    if len(ranked_day_data) < TOP_N_FINAL:
        return []
    return (
        ranked_day_data.sort_values(SIMPLE_RULE_BENCHMARK_FACTOR, ascending=False)
        .head(TOP_N_FINAL)["code"]
        .astype(str)
        .tolist()
    )


def calculate_portfolio_return(day_data: pd.DataFrame, codes: List[str]) -> float:
    if not codes:
        return float("nan")
    portfolio_data = day_data[day_data["code"].isin(codes)]
    if portfolio_data.empty:
        return float("nan")
    return float(portfolio_data[TARGET].mean())


def calculate_turnover(previous_codes: List[str], current_codes: List[str]) -> float:
    if not current_codes:
        return float("nan")
    if not previous_codes:
        return 1.0

    previous_weights = {code: 1.0 / len(previous_codes) for code in previous_codes} if previous_codes else {}
    current_weights = {code: 1.0 / len(current_codes) for code in current_codes}
    all_codes = set(previous_weights) | set(current_weights)
    if not all_codes:
        return float("nan")
    return float(
        0.5 * sum(abs(current_weights.get(code, 0.0) - previous_weights.get(code, 0.0)) for code in all_codes)
    )


def calculate_net_return_components(gross_return: float, turnover: float) -> Dict[str, float]:
    if pd.isna(gross_return) or pd.isna(turnover):
        return {
            "transaction_cost": float("nan"),
            "slippage": float("nan"),
            "net_return": float("nan"),
        }

    transaction_cost = turnover * TRANSACTION_COST_RATE
    slippage = turnover * SLIPPAGE_RATE
    net_return = gross_return - transaction_cost - slippage
    return {
        "transaction_cost": float(transaction_cost),
        "slippage": float(slippage),
        "net_return": float(net_return),
    }


def build_portfolio_result(
    prefix: str,
    day_data: pd.DataFrame,
    codes: List[str],
    previous_codes: List[str],
) -> Dict[str, Any]:
    gross_return = calculate_portfolio_return(day_data, codes)
    turnover = calculate_turnover(previous_codes, codes)
    net_components = calculate_net_return_components(gross_return, turnover)
    return {
        f"{prefix}_codes": codes,
        f"{prefix}_count": len(codes),
        f"{prefix}_return": gross_return,
        f"{prefix}_turnover": turnover,
        f"{prefix}_transaction_cost": net_components["transaction_cost"],
        f"{prefix}_slippage": net_components["slippage"],
        f"{prefix}_net_return": net_components["net_return"],
    }


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
                top_n_final=TOP_N_FINAL,
                score_reference_col=PRED_COL,
                api_timeout=LLM_TIMEOUT,
            )
            if llm_ranked is None:
                fallback_used = True
                print("  LLM reranking failed, falling back to stage-one ranking.")
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
                            print(f"  Explanation generation failed for {format_rebalance_date(date)}")
                    except Exception as exc:
                        print(f"  Explanation generation error for {format_rebalance_date(date)}: {exc}")
        except Exception as exc:
            fallback_used = True
            print(f"  LLM reranking error: {exc}. Falling back to stage-one ranking.")

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
    print(f"Total rebalance dates: {len(rebalance_dates)}")

    global SELECTED_EXPLANATION_DATES
    SELECTED_EXPLANATION_DATES = select_explanation_dates(rebalance_dates)
    if ENABLE_EXPLANATIONS:
        if SELECTED_EXPLANATION_DATES:
            print(f"Explanation dates: {', '.join(sorted(SELECTED_EXPLANATION_DATES))}")
        else:
            print("No explanation dates selected for this run.")

    results = []
    for index, date in enumerate(rebalance_dates):
        if index % 10 == 0:
            print(f"Processing rebalance date {index + 1}/{len(rebalance_dates)}: {date.date()}")

        codes_a = execute_strategy_a(df, date)
        should_explain = format_rebalance_date(date) in SELECTED_EXPLANATION_DATES
        strategy_b_result = execute_strategy_b(df, date, generate_explanation=should_explain)
        codes_b = strategy_b_result["selected_codes"]
        if not codes_a or not codes_b:
            continue

        day_data = df[df["日期"] == date].copy()
        returns_a = [
            day_data[day_data["code"] == code][TARGET].values[0]
            for code in codes_a
            if len(day_data[day_data["code"] == code]) > 0
        ]
        returns_b = [
            day_data[day_data["code"] == code][TARGET].values[0]
            for code in codes_b
            if len(day_data[day_data["code"] == code]) > 0
        ]
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
                "llm_cached_result_used": strategy_b_result.get("cached_result_used", False),
                "llm_cache_hit": strategy_b_result.get("cache_hit", False),
                "llm_actual_api_call_made": strategy_b_result.get("actual_llm_call_made", False),
                "llm_fallback_used": strategy_b_result["fallback_used"],
                "candidate_pool_fingerprint": strategy_b_result.get("candidate_pool_fingerprint"),
                "scenario_name": get_current_scenario_name(),
                "top_n_first": TOP_N_FIRST,
                "top_n_final": TOP_N_FINAL,
                "explanation_requested": should_explain,
                "explanation_generated": strategy_b_result["explanation_generated"],
            }
        )

    return pd.DataFrame(results)


def get_tradeable_day_data(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    standardized_df = standardize_date_column_name(df)
    date_column = next((candidate for candidate in DATE_COLUMN_CANDIDATES if candidate in standardized_df.columns), None)
    if date_column is None:
        raise KeyError("Date column is missing from the backtest dataset.")
    day_data = standardized_df[standardized_df[date_column] == date].copy()
    day_data = day_data[day_data[TARGET].notna()]
    day_data = day_data[day_data[PRED_COL].notna()]
    day_data["code"] = day_data["code"].astype(str)
    return day_data


def execute_strategy_a(df: pd.DataFrame, date: pd.Timestamp) -> List[str]:
    day_data = get_tradeable_day_data(df, date)
    if len(day_data) < TOP_N_FINAL:
        return []
    return day_data.sort_values(PRED_COL, ascending=False).head(TOP_N_FINAL)["code"].tolist()


def execute_strategy_b(df: pd.DataFrame, date: pd.Timestamp, generate_explanation: bool = False) -> Dict[str, Any]:
    day_data = get_tradeable_day_data(df, date)

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
                print("  LLM reranking failed, falling back to stage-one ranking.")
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
                            print(f"  Explanation generation failed for {format_rebalance_date(date)}")
                    except Exception as exc:
                        print(f"  Explanation generation error for {format_rebalance_date(date)}: {exc}")
        except Exception as exc:
            fallback_used = True
            print(f"  LLM reranking error: {exc}. Falling back to stage-one ranking.")

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
    print(f"Total rebalance dates: {len(rebalance_dates)}")

    global SELECTED_EXPLANATION_DATES, SIMPLE_RULE_BENCHMARK_FACTOR
    SELECTED_EXPLANATION_DATES = select_explanation_dates(rebalance_dates)
    SIMPLE_RULE_BENCHMARK_FACTOR = resolve_simple_rule_benchmark_factor(df)
    if SIMPLE_RULE_BENCHMARK_FACTOR:
        print(f"Simple rule benchmark factor: {SIMPLE_RULE_BENCHMARK_FACTOR}")
    else:
        print("Simple rule benchmark factor is unavailable in the current dataset.")

    if ENABLE_EXPLANATIONS:
        if SELECTED_EXPLANATION_DATES:
            print(f"Explanation dates: {', '.join(sorted(SELECTED_EXPLANATION_DATES))}")
        else:
            print("No explanation dates selected for this run.")

    results = []
    previous_holdings = {
        "strategy_a": [],
        "strategy_b": [],
        "benchmark_market": [],
        "benchmark_equal_weight": [],
        "benchmark_simple_rule": [],
    }

    for index, date in enumerate(rebalance_dates):
        if index % 10 == 0:
            print(f"Processing rebalance date {index + 1}/{len(rebalance_dates)}: {date.date()}")

        day_data = get_tradeable_day_data(df, date)
        if day_data.empty:
            continue

        codes_a = execute_strategy_a(df, date)
        should_explain = format_rebalance_date(date) in SELECTED_EXPLANATION_DATES
        strategy_b_result = execute_strategy_b(df, date, generate_explanation=should_explain)
        codes_b = strategy_b_result["selected_codes"]
        benchmark_market_codes = execute_market_benchmark(day_data)
        benchmark_equal_weight_codes = execute_equal_weight_benchmark(day_data)
        benchmark_simple_rule_codes = execute_simple_rule_benchmark(day_data)

        if (
            not codes_a
            or not codes_b
            or not benchmark_market_codes
            or not benchmark_equal_weight_codes
            or not benchmark_simple_rule_codes
        ):
            continue

        result_row = {
            "rebalance_date": date,
            "second_stage_llm": SECOND_STAGE_LLM,
            "llm_ranking_used": strategy_b_result["used_llm_ranking"],
            "llm_fallback_used": strategy_b_result["fallback_used"],
            "explanation_requested": should_explain,
            "explanation_generated": strategy_b_result["explanation_generated"],
            "simple_rule_benchmark_factor": SIMPLE_RULE_BENCHMARK_FACTOR,
        }
        result_row.update(
            build_portfolio_result(
                "strategy_a",
                day_data,
                codes_a,
                previous_holdings["strategy_a"],
            )
        )
        result_row.update(
            build_portfolio_result(
                "strategy_b",
                day_data,
                codes_b,
                previous_holdings["strategy_b"],
            )
        )
        result_row.update(
            build_portfolio_result(
                "benchmark_market",
                day_data,
                benchmark_market_codes,
                previous_holdings["benchmark_market"],
            )
        )
        result_row.update(
            build_portfolio_result(
                "benchmark_equal_weight",
                day_data,
                benchmark_equal_weight_codes,
                previous_holdings["benchmark_equal_weight"],
            )
        )
        result_row.update(
            build_portfolio_result(
                "benchmark_simple_rule",
                day_data,
                benchmark_simple_rule_codes,
                previous_holdings["benchmark_simple_rule"],
            )
        )
        results.append(result_row)

        previous_holdings["strategy_a"] = codes_a
        previous_holdings["strategy_b"] = codes_b
        previous_holdings["benchmark_market"] = benchmark_market_codes
        previous_holdings["benchmark_equal_weight"] = benchmark_equal_weight_codes
        previous_holdings["benchmark_simple_rule"] = benchmark_simple_rule_codes

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
    cumulative_return = cumulative.iloc[-1] - 1 if not cumulative.empty else np.nan

    return {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_periods": len(returns),
        "mean_return": mean_return,
        "std_return": returns.std(),
    }


def save_performance_metrics(performance_map: Dict[str, Dict[str, float]]) -> None:
    metric_names = [
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "total_periods",
        "mean_return",
        "std_return",
    ]
    metrics_df = pd.DataFrame(
        {
            strategy_name: [strategy_metrics.get(metric_name, np.nan) for metric_name in metric_names]
            for strategy_name, strategy_metrics in performance_map.items()
        },
        index=metric_names,
    )
    metrics_df.to_csv(OUTPUT_DIR / "performance_metrics.csv", encoding="utf-8-sig")


def plot_results(results_df: pd.DataFrame, performance_a: Dict[str, float], performance_b: Dict[str, float]) -> None:
    baseline_label, two_stage_label = get_strategy_display_names()
    two_stage_chart_label = get_chart_two_stage_label()
    cumulative_a = (1 + results_df["strategy_a_return"]).cumprod()
    cumulative_b = (1 + results_df["strategy_b_return"]).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(results_df["rebalance_date"], cumulative_a, label=baseline_label, linewidth=2)
    plt.plot(results_df["rebalance_date"], cumulative_b, label=two_stage_label, linewidth=2)
    plt.xlabel("调仓日期")
    plt.ylabel("累计收益")
    plt.title(f"{baseline_label}与{two_stage_chart_label}累计收益对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cumulative_returns.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(results_df["strategy_a_return"], bins=30, alpha=0.7, label=baseline_label)
    plt.xlabel("持有期收益")
    plt.ylabel("频数")
    plt.title(f"{baseline_label}收益分布")
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="盈亏平衡线")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(results_df["strategy_b_return"], bins=30, alpha=0.7, label=two_stage_label, color="orange")
    plt.xlabel("持有期收益")
    plt.ylabel("频数")
    plt.title(f"{two_stage_label}收益分布")
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="盈亏平衡线")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "returns_distribution.png", dpi=300)
    plt.close()

    metrics_df = pd.DataFrame(
        {
            "strategy_a": [
                f"{performance_a.get('annual_return', 0):.2%}",
                f"{performance_a.get('annual_volatility', 0):.2%}",
                f"{performance_a.get('sharpe_ratio', 0):.2f}",
                f"{performance_a.get('max_drawdown', 0):.2%}",
                f"{performance_a.get('win_rate', 0):.2%}",
                f"{performance_a.get('total_periods', 0)}",
            ],
            "strategy_b": [
                f"{performance_b.get('annual_return', 0):.2%}",
                f"{performance_b.get('annual_volatility', 0):.2%}",
                f"{performance_b.get('sharpe_ratio', 0):.2f}",
                f"{performance_b.get('max_drawdown', 0):.2%}",
                f"{performance_b.get('win_rate', 0):.2%}",
                f"{performance_b.get('total_periods', 0)}",
            ],
        },
        index=[
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "total_periods",
        ],
    )
    metrics_df.to_csv(OUTPUT_DIR / "performance_metrics.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].bar([baseline_label, two_stage_label], [performance_a.get("annual_return", 0), performance_b.get("annual_return", 0)])
    axes[0, 0].set_title("年化收益率")
    axes[0, 0].set_ylabel("数值")
    axes[0, 1].bar([baseline_label, two_stage_label], [performance_a.get("sharpe_ratio", 0), performance_b.get("sharpe_ratio", 0)])
    axes[0, 1].set_title("夏普比率")
    axes[0, 1].set_ylabel("数值")
    axes[1, 0].bar([baseline_label, two_stage_label], [performance_a.get("max_drawdown", 0), performance_b.get("max_drawdown", 0)])
    axes[1, 0].set_title("最大回撤")
    axes[1, 0].set_ylabel("数值")
    axes[1, 1].bar([baseline_label, two_stage_label], [performance_a.get("win_rate", 0), performance_b.get("win_rate", 0)])
    axes[1, 1].set_title("胜率")
    axes[1, 1].set_ylabel("数值")
    fig.suptitle(f"{baseline_label}与{two_stage_chart_label}绩效指标对比", fontsize=14)
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUTPUT_DIR / "performance_comparison.png", dpi=300)
    plt.close()


def generate_explanation_reports() -> None:
    if not EXPLANATION_DIR.is_dir():
        print("未找到解释目录，跳过解释报告生成。")
        return

    try:
        from explanation_reporter import ExplanationReporter
    except ImportError as exc:
        print(f"警告：导入 explanation_reporter 失败：{exc}")
        return

    reporter = ExplanationReporter(EXPLANATION_DIR)
    available_dates = reporter.list_available_dates()
    if not available_dates:
        print("未找到解释结果，跳过解释报告生成。")
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
        print(f"解释汇总报告：{summary_path}")
    except Exception as exc:
        print(f"解释汇总报告生成失败：{exc}")


def get_llm_log_subdir_name(llm_model: str) -> str:
    return str(llm_model).strip().replace("-", "_").replace(".", "_")


def get_two_stage_output_dir(traditional_model: str, llm_model: str) -> Path:
    return get_traditional_two_stage_output_dir(
        PROJECT_ROOT,
        traditional_model,
        llm_model,
        TOP_N_FIRST,
        TOP_N_FINAL,
    )


def configure_paths(traditional_model: str) -> None:
    global TRADITIONAL_MODEL, PREDICTIONS_PATH, OUTPUT_DIR, LLM_LOG_DIR, EXPLANATION_DIR

    TRADITIONAL_MODEL = traditional_model
    PREDICTIONS_PATH = get_traditional_prediction_file(PROJECT_ROOT, traditional_model)
    OUTPUT_DIR = get_two_stage_output_dir(traditional_model, SECOND_STAGE_LLM)
    LLM_LOG_DIR = get_traditional_llm_log_dir(PROJECT_ROOT, traditional_model, SECOND_STAGE_LLM)
    EXPLANATION_DIR = get_traditional_explanation_dir(PROJECT_ROOT, traditional_model, SECOND_STAGE_LLM)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_rebalance_cache_path(date: pd.Timestamp, candidate_pool_fingerprint: Optional[str] = None) -> Path:
    fingerprint_suffix = ""
    if candidate_pool_fingerprint:
        fingerprint_suffix = f"_{str(candidate_pool_fingerprint)[:12]}"
    return LLM_LOG_DIR / f"rebalance_{pd.Timestamp(date).strftime('%Y%m%d')}{fingerprint_suffix}.csv"


def get_current_scenario_name() -> str:
    return get_backtest_scenario_name(TOP_N_FIRST, TOP_N_FINAL)


def build_candidate_pool_fingerprint(candidate_df: pd.DataFrame, date: pd.Timestamp) -> str:
    normalized_candidates = candidate_df.copy()
    normalized_candidates["code"] = normalized_candidates["code"].astype(str)
    fingerprint_payload = {
        "rebalance_date": format_rebalance_date(date),
        "traditional_model": TRADITIONAL_MODEL,
        "second_stage_llm": SECOND_STAGE_LLM,
        "target": TARGET,
        "scenario_name": get_current_scenario_name(),
        "top_n_first": int(TOP_N_FIRST),
        "top_n_final": int(TOP_N_FINAL),
        "candidate_codes": normalized_candidates["code"].tolist(),
    }
    serialized_payload = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def load_cached_rebalance_selection(
    candidate_df: pd.DataFrame,
    date: pd.Timestamp,
    candidate_pool_fingerprint: str,
) -> Optional[pd.DataFrame]:
    cache_path = get_rebalance_cache_path(date, candidate_pool_fingerprint=candidate_pool_fingerprint)
    if not cache_path.exists():
        return None

    cached_df = read_csv_with_fallback(cache_path)
    required_columns = {"code", "candidate_pool_fingerprint"}
    missing_columns = [column for column in required_columns if column not in cached_df.columns]
    if missing_columns:
        print(
            f"  Cached rebalance file missing required columns {missing_columns}, ignoring: {cache_path}"
        )
        return None

    cached_fingerprint = str(cached_df["candidate_pool_fingerprint"].iloc[0]).strip()
    if cached_fingerprint != str(candidate_pool_fingerprint).strip():
        print(f"  Cached rebalance fingerprint mismatch for {format_rebalance_date(date)}, refreshing LLM call.")
        return None

    cached_df["code"] = cached_df["code"].astype(str)
    ranked_etfs = candidate_df.copy()
    ranked_etfs["code"] = ranked_etfs["code"].astype(str)
    ranked_etfs = ranked_etfs.merge(
        cached_df[
            ["code"]
            + [
                col
                for col in (
                    "llm_score",
                    "second_stage_llm",
                    "candidate_pool_fingerprint",
                    "scenario_name",
                    "top_n_first",
                    "top_n_final",
                    "cache_hit",
                )
                if col in cached_df.columns
            ]
        ],
        on="code",
        how="inner",
    )
    if ranked_etfs.empty:
        print(f"  Cached rebalance file does not match current candidate pool, ignoring: {cache_path}")
        return None

    if "llm_score" not in ranked_etfs.columns:
        ranked_etfs["llm_score"] = ranked_etfs[PRED_COL]
    if "second_stage_llm" not in ranked_etfs.columns:
        ranked_etfs["second_stage_llm"] = SECOND_STAGE_LLM
    if "candidate_pool_fingerprint" not in ranked_etfs.columns:
        ranked_etfs["candidate_pool_fingerprint"] = candidate_pool_fingerprint
    if "scenario_name" not in ranked_etfs.columns:
        ranked_etfs["scenario_name"] = get_current_scenario_name()
    if "top_n_first" not in ranked_etfs.columns:
        ranked_etfs["top_n_first"] = TOP_N_FIRST
    if "top_n_final" not in ranked_etfs.columns:
        ranked_etfs["top_n_final"] = TOP_N_FINAL
    ranked_etfs["cache_hit"] = True

    cache_order = {code: idx for idx, code in enumerate(cached_df["code"].tolist())}
    ranked_etfs["cache_rank"] = ranked_etfs["code"].map(cache_order)
    ranked_etfs = ranked_etfs.sort_values(["cache_rank", "llm_score"], ascending=[True, False]).reset_index(drop=True)
    ranked_etfs = ranked_etfs.drop(columns=["cache_rank"])
    return ranked_etfs.head(TOP_N_FINAL).copy()


def save_rebalance_cache(
    ranked_etfs: pd.DataFrame,
    date: pd.Timestamp,
    candidate_pool_fingerprint: str,
) -> None:
    cache_path = get_rebalance_cache_path(date, candidate_pool_fingerprint=candidate_pool_fingerprint)
    final_etfs = ranked_etfs.head(TOP_N_FINAL).copy()
    cache_df = pd.DataFrame(
        {
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "code": final_etfs["code"].astype(str),
            "name": final_etfs["name"] if "name" in final_etfs.columns else "",
            "llm_score": final_etfs["llm_score"] if "llm_score" in final_etfs.columns else np.nan,
            "traditional_score": final_etfs[PRED_COL] if PRED_COL in final_etfs.columns else np.nan,
            "second_stage_llm": SECOND_STAGE_LLM,
            "traditional_model": TRADITIONAL_MODEL,
            "target": TARGET,
            "scenario_name": get_current_scenario_name(),
            "top_n_first": TOP_N_FIRST,
            "top_n_final": TOP_N_FINAL,
            "candidate_pool_fingerprint": candidate_pool_fingerprint,
            "cache_hit": False,
        }
    )
    cache_df.to_csv(cache_path, index=False, encoding="utf-8-sig")


def save_turnover_summary(results_df: pd.DataFrame) -> Path:
    turnover_columns = [
        "rebalance_date",
        "strategy_a_turnover",
        "strategy_a_transaction_cost",
        "strategy_a_slippage",
        "strategy_a_net_return",
        "strategy_b_turnover",
        "strategy_b_transaction_cost",
        "strategy_b_slippage",
        "strategy_b_net_return",
        "benchmark_market_turnover",
        "benchmark_market_transaction_cost",
        "benchmark_market_slippage",
        "benchmark_market_net_return",
        "benchmark_equal_weight_turnover",
        "benchmark_equal_weight_transaction_cost",
        "benchmark_equal_weight_slippage",
        "benchmark_equal_weight_net_return",
        "benchmark_simple_rule_turnover",
        "benchmark_simple_rule_transaction_cost",
        "benchmark_simple_rule_slippage",
        "benchmark_simple_rule_net_return",
    ]
    available_columns = [column for column in turnover_columns if column in results_df.columns]
    turnover_summary_path = OUTPUT_DIR / "turnover_summary.csv"
    results_df[available_columns].to_csv(turnover_summary_path, index=False, encoding="utf-8-sig")
    return turnover_summary_path


def save_benchmark_outputs(results_df: pd.DataFrame) -> dict[str, Path]:
    benchmark_specs = {
        "equal_weight": {
            "columns": [
                "rebalance_date",
                "benchmark_equal_weight_codes",
                "benchmark_equal_weight_count",
                "benchmark_equal_weight_return",
                "benchmark_equal_weight_turnover",
                "benchmark_equal_weight_transaction_cost",
                "benchmark_equal_weight_slippage",
                "benchmark_equal_weight_net_return",
            ],
            "gross_performance": "performance_benchmark_equal_weight",
            "net_performance": "performance_benchmark_equal_weight_net",
            "display_name": "ETF池等权组合",
        },
        "single_factor_momentum_20": {
            "columns": [
                "rebalance_date",
                "benchmark_simple_rule_codes",
                "benchmark_simple_rule_count",
                "benchmark_simple_rule_return",
                "benchmark_simple_rule_turnover",
                "benchmark_simple_rule_transaction_cost",
                "benchmark_simple_rule_slippage",
                "benchmark_simple_rule_net_return",
            ],
            "gross_performance": "performance_benchmark_simple_rule",
            "net_performance": "performance_benchmark_simple_rule_net",
            "display_name": "简单规则基准",
        },
    }
    benchmark_paths: dict[str, Path] = {}
    legacy_shared_benchmark_paths: dict[str, Path] = {}
    benchmark_metrics_paths: dict[str, Path] = {}
    benchmark_reports: dict[str, Any] = {}
    scenario_name = get_current_scenario_name()

    for benchmark_name, spec in benchmark_specs.items():
        columns = spec["columns"]
        available_columns = [column for column in columns if column in results_df.columns]

        benchmark_dir = get_traditional_benchmark_dir(
            PROJECT_ROOT,
            TRADITIONAL_MODEL,
            benchmark_name,
            TOP_N_FIRST,
            TOP_N_FINAL,
        )
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_result_path = benchmark_dir / "backtest_results.csv"
        results_df[available_columns].to_csv(benchmark_result_path, index=False, encoding="utf-8-sig")
        benchmark_paths[benchmark_name] = benchmark_result_path

        legacy_shared_dir = get_shared_benchmark_dir(
            PROJECT_ROOT,
            benchmark_name,
            TOP_N_FIRST,
            TOP_N_FINAL,
        )
        legacy_shared_dir.mkdir(parents=True, exist_ok=True)
        legacy_shared_result_path = legacy_shared_dir / "backtest_results.csv"
        results_df[available_columns].to_csv(legacy_shared_result_path, index=False, encoding="utf-8-sig")
        legacy_shared_benchmark_paths[benchmark_name] = legacy_shared_result_path

        metrics_payload = {
            "parameters": {
                "traditional_model": TRADITIONAL_MODEL,
                "benchmark_name": benchmark_name,
                "benchmark_label": spec["display_name"],
                "scenario": scenario_name,
                "scenario_name": scenario_name,
                "top_n_first": TOP_N_FIRST,
                "top_n_final": TOP_N_FINAL,
                "rebalancing_freq": REBALANCING_FREQ,
                "holding_days": HOLDING_DAYS,
                "backtest_results_path": to_project_relative_path(benchmark_result_path),
            },
            "performance": BENCHMARK_PERFORMANCE_CACHE.get(spec["gross_performance"], {}),
            "performance_net": BENCHMARK_PERFORMANCE_CACHE.get(spec["net_performance"], {}),
        }
        benchmark_metrics_path = benchmark_dir / "benchmark_metrics.json"
        with open(benchmark_metrics_path, "w", encoding="utf-8") as handle:
            json.dump(convert_paths_to_relative(metrics_payload), handle, indent=2, ensure_ascii=False)
        benchmark_metrics_paths[benchmark_name] = benchmark_metrics_path
        benchmark_reports[benchmark_name] = metrics_payload

    BENCHMARK_OUTPUT_CACHE.clear()
    BENCHMARK_OUTPUT_CACHE.update(
        {
            "model_output_paths": benchmark_paths,
            "legacy_shared_output_paths": legacy_shared_benchmark_paths,
            "metrics_paths": benchmark_metrics_paths,
            "reports": benchmark_reports,
        }
    )
    return benchmark_paths


def execute_strategy_b(df: pd.DataFrame, date: pd.Timestamp, generate_explanation: bool = False) -> Dict[str, Any]:
    day_data = get_tradeable_day_data(df, date)

    if len(day_data) < TOP_N_FIRST:
        return {
            "selected_codes": [],
            "ranked_etfs": None,
            "used_llm_ranking": False,
            "cached_result_used": False,
            "fallback_used": False,
            "explanation_generated": False,
            "cache_hit": False,
            "actual_llm_call_made": False,
            "candidate_pool_fingerprint": None,
        }

    top_etfs = day_data.sort_values(PRED_COL, ascending=False).head(TOP_N_FIRST).copy()
    candidate_pool_fingerprint = build_candidate_pool_fingerprint(top_etfs, date)
    ranked_etfs = top_etfs.copy()
    ranked_etfs["llm_score"] = ranked_etfs[PRED_COL]
    ranked_etfs["candidate_pool_fingerprint"] = candidate_pool_fingerprint
    ranked_etfs["scenario_name"] = get_current_scenario_name()
    ranked_etfs["top_n_first"] = TOP_N_FIRST
    ranked_etfs["top_n_final"] = TOP_N_FINAL
    used_llm_ranking = False
    cached_result_used = False
    fallback_used = False
    explanation_generated = False
    cache_hit = False
    actual_llm_call_made = False

    if USE_LLM:
        cached_ranked = load_cached_rebalance_selection(
            candidate_df=top_etfs,
            date=date,
            candidate_pool_fingerprint=candidate_pool_fingerprint,
        )
        if cached_ranked is not None:
            ranked_etfs = cached_ranked
            used_llm_ranking = True
            cached_result_used = True
            cache_hit = True
            print(f"  Cache hit for {format_rebalance_date(date)}, skipping LLM API call.")
        else:
            try:
                from llm_ranking import generate_explanations_for_date, rank_etfs_by_llm

                actual_llm_call_made = True
                llm_ranked = rank_etfs_by_llm(
                    top_etfs,
                    format_rebalance_date(date),
                    log_dir=LLM_LOG_DIR,
                    enable_explanations=False,
                    mock=MOCK_LLM,
                    top_n_final=TOP_N_FINAL,
                    score_reference_col=PRED_COL,
                    api_timeout=LLM_TIMEOUT,
                    llm_model=SECOND_STAGE_LLM,
                    config_path=LLM_CONFIG_PATH,
                )
                if llm_ranked is None:
                    fallback_used = True
                    print("  LLM reranking failed, falling back to stage-one ranking.")
                else:
                    ranked_etfs = llm_ranked
                    ranked_etfs["candidate_pool_fingerprint"] = candidate_pool_fingerprint
                    ranked_etfs["scenario_name"] = get_current_scenario_name()
                    ranked_etfs["top_n_first"] = TOP_N_FIRST
                    ranked_etfs["top_n_final"] = TOP_N_FINAL
                    ranked_etfs["cache_hit"] = False
                    used_llm_ranking = True
                    save_rebalance_cache(
                        ranked_etfs=ranked_etfs,
                        date=date,
                        candidate_pool_fingerprint=candidate_pool_fingerprint,
                    )

                    if generate_explanation:
                        try:
                            explanation_result = generate_explanations_for_date(
                                ranked_etfs.head(TOP_N_FINAL).copy(),
                                format_rebalance_date(date),
                                log_dir=LLM_LOG_DIR,
                                mock=MOCK_LLM,
                                api_timeout=LLM_TIMEOUT,
                                llm_model=SECOND_STAGE_LLM,
                                config_path=LLM_CONFIG_PATH,
                            )
                            explanation_generated = explanation_result is not None
                            if not explanation_generated:
                                print(f"  Explanation generation failed for {format_rebalance_date(date)}")
                        except Exception as exc:
                            print(f"  Explanation generation error for {format_rebalance_date(date)}: {exc}")
            except Exception as exc:
                fallback_used = True
                print(f"  LLM reranking error: {exc}. Falling back to stage-one ranking.")

    selected_codes = ranked_etfs.head(TOP_N_FINAL)["code"].tolist()
    return {
        "selected_codes": selected_codes,
        "ranked_etfs": ranked_etfs,
        "used_llm_ranking": used_llm_ranking,
        "cached_result_used": cached_result_used,
        "fallback_used": fallback_used,
        "explanation_generated": explanation_generated,
        "cache_hit": cache_hit,
        "actual_llm_call_made": actual_llm_call_made,
        "candidate_pool_fingerprint": candidate_pool_fingerprint,
    }


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    rebalance_dates = get_rebalancing_dates(df)
    print(f"Total rebalance dates: {len(rebalance_dates)}")

    global SELECTED_EXPLANATION_DATES, SIMPLE_RULE_BENCHMARK_FACTOR
    SELECTED_EXPLANATION_DATES = select_explanation_dates(rebalance_dates)
    SIMPLE_RULE_BENCHMARK_FACTOR = resolve_simple_rule_benchmark_factor(df)
    if SIMPLE_RULE_BENCHMARK_FACTOR:
        print(f"Simple rule benchmark factor: {SIMPLE_RULE_BENCHMARK_FACTOR}")
    else:
        print("Simple rule benchmark factor is unavailable in the current dataset.")

    if ENABLE_EXPLANATIONS:
        if SELECTED_EXPLANATION_DATES:
            print(f"Explanation dates: {', '.join(sorted(SELECTED_EXPLANATION_DATES))}")
        else:
            print("No explanation dates selected for this run.")

    results = []
    previous_holdings = {
        "strategy_a": [],
        "strategy_b": [],
        "benchmark_market": [],
        "benchmark_equal_weight": [],
        "benchmark_simple_rule": [],
    }

    for index, date in enumerate(rebalance_dates):
        if index % 10 == 0:
            print(f"Processing rebalance date {index + 1}/{len(rebalance_dates)}: {date.date()}")

        day_data = get_tradeable_day_data(df, date)
        if day_data.empty:
            continue

        codes_a = execute_strategy_a(df, date)
        should_explain = format_rebalance_date(date) in SELECTED_EXPLANATION_DATES
        strategy_b_result = execute_strategy_b(df, date, generate_explanation=should_explain)
        codes_b = strategy_b_result["selected_codes"]
        benchmark_market_codes = execute_market_benchmark(day_data)
        benchmark_equal_weight_codes = execute_equal_weight_benchmark(day_data)
        benchmark_simple_rule_codes = execute_simple_rule_benchmark(day_data)

        if (
            not codes_a
            or not codes_b
            or not benchmark_market_codes
            or not benchmark_equal_weight_codes
            or not benchmark_simple_rule_codes
        ):
            continue

        result_row = {
            "rebalance_date": date,
            "second_stage_llm": SECOND_STAGE_LLM,
            "llm_ranking_used": strategy_b_result["used_llm_ranking"],
            "llm_fallback_used": strategy_b_result["fallback_used"],
            "explanation_requested": should_explain,
            "explanation_generated": strategy_b_result["explanation_generated"],
            "simple_rule_benchmark_factor": SIMPLE_RULE_BENCHMARK_FACTOR,
        }
        result_row.update(
            build_portfolio_result(
                "strategy_a",
                day_data,
                codes_a,
                previous_holdings["strategy_a"],
            )
        )
        result_row.update(
            build_portfolio_result(
                "strategy_b",
                day_data,
                codes_b,
                previous_holdings["strategy_b"],
            )
        )
        result_row.update(
            build_portfolio_result(
                "benchmark_market",
                day_data,
                benchmark_market_codes,
                previous_holdings["benchmark_market"],
            )
        )
        result_row.update(
            build_portfolio_result(
                "benchmark_equal_weight",
                day_data,
                benchmark_equal_weight_codes,
                previous_holdings["benchmark_equal_weight"],
            )
        )
        result_row.update(
            build_portfolio_result(
                "benchmark_simple_rule",
                day_data,
                benchmark_simple_rule_codes,
                previous_holdings["benchmark_simple_rule"],
            )
        )
        results.append(result_row)

        previous_holdings["strategy_a"] = codes_a
        previous_holdings["strategy_b"] = codes_b
        previous_holdings["benchmark_market"] = benchmark_market_codes
        previous_holdings["benchmark_equal_weight"] = benchmark_equal_weight_codes
        previous_holdings["benchmark_simple_rule"] = benchmark_simple_rule_codes

    return pd.DataFrame(results)


def save_turnover_summary(results_df: pd.DataFrame) -> Path:
    turnover_columns = [
        "rebalance_date",
        "strategy_a_turnover",
        "strategy_a_transaction_cost",
        "strategy_a_slippage",
        "strategy_a_net_return",
        "strategy_b_turnover",
        "strategy_b_transaction_cost",
        "strategy_b_slippage",
        "strategy_b_net_return",
        "benchmark_market_turnover",
        "benchmark_market_transaction_cost",
        "benchmark_market_slippage",
        "benchmark_market_net_return",
        "benchmark_equal_weight_turnover",
        "benchmark_equal_weight_transaction_cost",
        "benchmark_equal_weight_slippage",
        "benchmark_equal_weight_net_return",
        "benchmark_simple_rule_turnover",
        "benchmark_simple_rule_transaction_cost",
        "benchmark_simple_rule_slippage",
        "benchmark_simple_rule_net_return",
    ]
    available_columns = [column for column in turnover_columns if column in results_df.columns]
    turnover_summary_path = OUTPUT_DIR / "turnover_summary.csv"
    results_df[available_columns].to_csv(turnover_summary_path, index=False, encoding="utf-8-sig")
    return turnover_summary_path


def save_benchmark_outputs(results_df: pd.DataFrame) -> dict[str, Path]:
    benchmark_specs = {
        MARKET_BENCHMARK_KEY: {
            "columns": [
                "rebalance_date",
                "benchmark_market_codes",
                "benchmark_market_count",
                "benchmark_market_return",
                "benchmark_market_turnover",
                "benchmark_market_transaction_cost",
                "benchmark_market_slippage",
                "benchmark_market_net_return",
            ],
            "gross_performance": "performance_market_benchmark",
            "net_performance": "performance_market_benchmark_net",
            "display_name": MARKET_BENCHMARK_NAME,
            "benchmark_name": MARKET_BENCHMARK_NAME,
            "benchmark_code": MARKET_BENCHMARK_CODE,
        },
        "equal_weight": {
            "columns": [
                "rebalance_date",
                "benchmark_equal_weight_codes",
                "benchmark_equal_weight_count",
                "benchmark_equal_weight_return",
                "benchmark_equal_weight_turnover",
                "benchmark_equal_weight_transaction_cost",
                "benchmark_equal_weight_slippage",
                "benchmark_equal_weight_net_return",
            ],
            "gross_performance": "performance_benchmark_equal_weight",
            "net_performance": "performance_benchmark_equal_weight_net",
            "display_name": "样本池等权ETF组合",
            "benchmark_name": "样本池等权ETF组合",
            "benchmark_code": "",
        },
        "single_factor_momentum_20": {
            "columns": [
                "rebalance_date",
                "benchmark_simple_rule_codes",
                "benchmark_simple_rule_count",
                "benchmark_simple_rule_return",
                "benchmark_simple_rule_turnover",
                "benchmark_simple_rule_transaction_cost",
                "benchmark_simple_rule_slippage",
                "benchmark_simple_rule_net_return",
            ],
            "gross_performance": "performance_benchmark_simple_rule",
            "net_performance": "performance_benchmark_simple_rule_net",
            "display_name": "单因子规则基准",
            "benchmark_name": "单因子规则基准",
            "benchmark_code": "",
        },
    }
    benchmark_paths: dict[str, Path] = {}
    legacy_shared_benchmark_paths: dict[str, Path] = {}
    benchmark_metrics_paths: dict[str, Path] = {}
    benchmark_reports: dict[str, Any] = {}
    scenario_name = get_current_scenario_name()

    for benchmark_name, spec in benchmark_specs.items():
        columns = spec["columns"]
        available_columns = [column for column in columns if column in results_df.columns]

        benchmark_dir = get_traditional_benchmark_dir(
            PROJECT_ROOT,
            TRADITIONAL_MODEL,
            benchmark_name,
            TOP_N_FIRST,
            TOP_N_FINAL,
        )
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_result_path = benchmark_dir / "backtest_results.csv"
        results_df[available_columns].to_csv(benchmark_result_path, index=False, encoding="utf-8-sig")
        benchmark_paths[benchmark_name] = benchmark_result_path

        legacy_shared_dir = get_shared_benchmark_dir(
            PROJECT_ROOT,
            benchmark_name,
            TOP_N_FIRST,
            TOP_N_FINAL,
        )
        legacy_shared_dir.mkdir(parents=True, exist_ok=True)
        legacy_shared_result_path = legacy_shared_dir / "backtest_results.csv"
        results_df[available_columns].to_csv(legacy_shared_result_path, index=False, encoding="utf-8-sig")
        legacy_shared_benchmark_paths[benchmark_name] = legacy_shared_result_path

        metrics_payload = {
            "parameters": {
                "traditional_model": TRADITIONAL_MODEL,
                "benchmark_name": benchmark_name,
                "benchmark_label": spec["display_name"],
                "benchmark_name_display": spec.get("benchmark_name", spec["display_name"]),
                "benchmark_code": spec.get("benchmark_code", ""),
                "scenario": scenario_name,
                "scenario_name": scenario_name,
                "top_n_first": TOP_N_FIRST,
                "top_n_final": TOP_N_FINAL,
                "rebalancing_freq": REBALANCING_FREQ,
                "holding_days": HOLDING_DAYS,
                "backtest_results_path": to_project_relative_path(benchmark_result_path),
            },
            "performance": BENCHMARK_PERFORMANCE_CACHE.get(spec["gross_performance"], {}),
            "performance_net": BENCHMARK_PERFORMANCE_CACHE.get(spec["net_performance"], {}),
        }
        benchmark_metrics_path = benchmark_dir / "benchmark_metrics.json"
        with open(benchmark_metrics_path, "w", encoding="utf-8") as handle:
            json.dump(convert_paths_to_relative(metrics_payload), handle, indent=2, ensure_ascii=False)
        benchmark_metrics_paths[benchmark_name] = benchmark_metrics_path
        benchmark_reports[benchmark_name] = metrics_payload

    BENCHMARK_OUTPUT_CACHE.clear()
    BENCHMARK_OUTPUT_CACHE.update(
        {
            "model_output_paths": benchmark_paths,
            "legacy_shared_output_paths": legacy_shared_benchmark_paths,
            "metrics_paths": benchmark_metrics_paths,
            "reports": benchmark_reports,
        }
    )
    return benchmark_paths


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
    global SECOND_STAGE_LLM, LLM_CONFIG_PATH

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
    SECOND_STAGE_LLM = args.second_stage_llm
    LLM_CONFIG_PATH = args.llm_config
    LLM_TIMEOUT = args.llm_timeout
    configure_paths(traditional_model)

    if args.enable_explanations and not USE_LLM:
        print("Explanations were requested with --no-llm. Explanation generation will be skipped.")
    if args.mock_llm and not USE_LLM:
        print("--mock-llm was passed together with --no-llm. Mock reranking will be ignored.")

    baseline_label, two_stage_label = get_strategy_display_names()
    print(f"Second-stage LLM: {SECOND_STAGE_LLM}")
    print(f"LLM log dir: {LLM_LOG_DIR}")
    print("=" * 60)
    print("ETF 两阶段回测")
    print("=" * 60)
    print(f"第一阶段传统模型: {traditional_model}")
    print(f"预测目标: {TARGET}")
    print(f"调仓频率: {REBALANCING_FREQ}")
    print(f"第一阶段候选数: {TOP_N_FIRST}")
    print(f"最终持仓数: {TOP_N_FINAL}")
    print(f"基线策略名称: {baseline_label}")
    print(f"两阶段策略名称: {two_stage_label}")
    print(f"是否启用 LLM 排序: {USE_LLM}")
    print(f"是否使用 Mock 验证模式: {MOCK_LLM}")
    print(f"是否生成解释报告: {ENABLE_EXPLANATIONS}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    df = load_data()
    print(f"数据范围: {df['日期'].min().date()} 至 {df['日期'].max().date()}")
    print(f"ETF 数量: {df['code'].nunique()}")
    print(f"样本总数: {len(df)}")

    results_df = run_backtest(df)
    if results_df.empty:
        print("Backtest produced no valid results. Check parameters and input data.")
        return

    performance_a = calculate_performance(results_df["strategy_a_return"], holding_days=HOLDING_DAYS)
    performance_b = calculate_performance(results_df["strategy_b_return"], holding_days=HOLDING_DAYS)
    performance_a_net = calculate_performance(results_df["strategy_a_net_return"], holding_days=HOLDING_DAYS)
    performance_b_net = calculate_performance(results_df["strategy_b_net_return"], holding_days=HOLDING_DAYS)
    performance_market_benchmark = calculate_performance(
        results_df["benchmark_market_return"],
        holding_days=HOLDING_DAYS,
    )
    performance_market_benchmark_net = calculate_performance(
        results_df["benchmark_market_net_return"],
        holding_days=HOLDING_DAYS,
    )
    performance_benchmark_equal_weight = calculate_performance(
        results_df["benchmark_equal_weight_return"],
        holding_days=HOLDING_DAYS,
    )
    performance_benchmark_equal_weight_net = calculate_performance(
        results_df["benchmark_equal_weight_net_return"],
        holding_days=HOLDING_DAYS,
    )
    performance_benchmark_simple_rule = calculate_performance(
        results_df["benchmark_simple_rule_return"],
        holding_days=HOLDING_DAYS,
    )
    performance_benchmark_simple_rule_net = calculate_performance(
        results_df["benchmark_simple_rule_net_return"],
        holding_days=HOLDING_DAYS,
    )
    BENCHMARK_PERFORMANCE_CACHE.clear()
    BENCHMARK_PERFORMANCE_CACHE.update(
        {
            "performance_market_benchmark": performance_market_benchmark,
            "performance_market_benchmark_net": performance_market_benchmark_net,
            "performance_benchmark_equal_weight": performance_benchmark_equal_weight,
            "performance_benchmark_equal_weight_net": performance_benchmark_equal_weight_net,
            "performance_benchmark_simple_rule": performance_benchmark_simple_rule,
            "performance_benchmark_simple_rule_net": performance_benchmark_simple_rule_net,
        }
    )

    backtest_results_path = OUTPUT_DIR / "backtest_results.csv"
    results_df.to_csv(backtest_results_path, index=False, encoding="utf-8-sig")
    turnover_summary_path = save_turnover_summary(results_df)
    benchmark_paths = save_benchmark_outputs(results_df)

    print("\n" + "=" * 60)
    print("绩效指标对比")
    print("=" * 60)
    print(f"{'指标':<20} {baseline_label:<24} {two_stage_label:<24}")
    print("-" * 60)
    print(f"{'年化收益率':<20} {performance_a.get('annual_return', 0):.2%} {performance_b.get('annual_return', 0):.2%}")
    print(f"{'年化波动率':<20} {performance_a.get('annual_volatility', 0):.2%} {performance_b.get('annual_volatility', 0):.2%}")
    print(f"{'夏普比率':<20} {performance_a.get('sharpe_ratio', 0):.2f} {performance_b.get('sharpe_ratio', 0):.2f}")
    print(f"{'最大回撤':<20} {performance_a.get('max_drawdown', 0):.2%} {performance_b.get('max_drawdown', 0):.2%}")
    print(f"{'胜率':<20} {performance_a.get('win_rate', 0):.2%} {performance_b.get('win_rate', 0):.2%}")
    print(f"{'调仓周期数':<20} {performance_a.get('total_periods', 0):<20} {performance_b.get('total_periods', 0):<20}")

    plot_results(results_df, performance_a, performance_b)
    save_performance_metrics(
        {
            "strategy_a": performance_a,
            "strategy_b": performance_b,
            "strategy_a_net": performance_a_net,
            "strategy_b_net": performance_b_net,
            "market_benchmark": performance_market_benchmark,
            "market_benchmark_net": performance_market_benchmark_net,
            "benchmark_equal_weight": performance_benchmark_equal_weight,
            "benchmark_equal_weight_net": performance_benchmark_equal_weight_net,
            "benchmark_simple_rule": performance_benchmark_simple_rule,
            "benchmark_simple_rule_net": performance_benchmark_simple_rule_net,
        }
    )

    generated_explanation_dates = sorted(
        results_df.loc[results_df["explanation_generated"], "rebalance_date"].dt.strftime("%Y-%m-%d").tolist()
    )
    scenario_name = get_backtest_scenario_name(TOP_N_FIRST, TOP_N_FINAL)
    report = {
        "parameters": {
            "traditional_model": traditional_model,
            "target": TARGET,
            "second_stage_llm": SECOND_STAGE_LLM,
            "scenario": scenario_name,
            "scenario_name": scenario_name,
            "top_n_first": TOP_N_FIRST,
            "top_n_final": TOP_N_FINAL,
            "rebalancing_freq": REBALANCING_FREQ,
            "holding_days": HOLDING_DAYS,
            "llm_config_path": to_project_relative_path(LLM_CONFIG_PATH),
            "llm_log_dir": to_project_relative_path(LLM_LOG_DIR),
            "benchmark_output_paths": convert_paths_to_relative(benchmark_paths),
            "benchmark_output_paths_legacy_shared": convert_paths_to_relative(
                BENCHMARK_OUTPUT_CACHE.get("legacy_shared_output_paths", {})
            ),
            "benchmark_metrics_paths": convert_paths_to_relative(BENCHMARK_OUTPUT_CACHE.get("metrics_paths", {})),
            "benchmark_name": MARKET_BENCHMARK_NAME,
            "benchmark_code": MARKET_BENCHMARK_CODE,
            "simple_rule_benchmark_factor": SIMPLE_RULE_BENCHMARK_FACTOR,
            "benchmark_factor_candidates": list(BENCHMARK_FACTOR_CANDIDATES),
            "transaction_cost_rate": TRANSACTION_COST_RATE,
            "slippage_rate": SLIPPAGE_RATE,
            "use_llm": USE_LLM,
            "mock_llm": MOCK_LLM,
            "enable_explanations": ENABLE_EXPLANATIONS,
            "explanation_date": EXPLANATION_DATE,
            "explanation_sample_size": EXPLANATION_SAMPLE_SIZE,
            "selected_explanation_dates": sorted(SELECTED_EXPLANATION_DATES),
            "generated_explanation_dates": generated_explanation_dates,
            "generated_explanations_count": len(generated_explanation_dates),
            "llm_timeout": LLM_TIMEOUT,
            "predictions_path": to_project_relative_path(PREDICTIONS_PATH),
            "output_dir": to_project_relative_path(OUTPUT_DIR),
            "turnover_summary_path": to_project_relative_path(turnover_summary_path),
            "backtest_results_path": to_project_relative_path(backtest_results_path),
        },
        "performance_strategy_a": performance_a,
        "performance_strategy_b": performance_b,
        "performance_strategy_a_net": performance_a_net,
        "performance_strategy_b_net": performance_b_net,
        "performance_market_benchmark": performance_market_benchmark,
        "performance_market_benchmark_net": performance_market_benchmark_net,
        "performance_benchmark_equal_weight": performance_benchmark_equal_weight,
        "performance_benchmark_equal_weight_net": performance_benchmark_equal_weight_net,
        "performance_benchmark_simple_rule": performance_benchmark_simple_rule,
        "performance_benchmark_simple_rule_net": performance_benchmark_simple_rule_net,
        "benchmarks": BENCHMARK_OUTPUT_CACHE.get("reports", {}),
    }

    report_path = OUTPUT_DIR / "performance_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(convert_paths_to_relative(report), handle, indent=2, ensure_ascii=False)
    print(f"\n绩效报告已保存至: {report_path}")

    run_manifest = {
        "traditional_model": traditional_model,
        "target": TARGET,
        "second_stage_llm": SECOND_STAGE_LLM,
        "scenario": scenario_name,
        "scenario_name": scenario_name,
        "data_path": to_project_relative_path(DATA_PATH),
        "predictions_path": to_project_relative_path(PREDICTIONS_PATH),
        "llm_log_dir": to_project_relative_path(LLM_LOG_DIR),
        "output_dir": to_project_relative_path(OUTPUT_DIR),
        "top_n_first": TOP_N_FIRST,
        "top_n_final": TOP_N_FINAL,
        "rebalancing_freq": REBALANCING_FREQ,
        "holding_days": HOLDING_DAYS,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    run_manifest_path = get_traditional_two_stage_manifest_file(
        PROJECT_ROOT,
        traditional_model,
        SECOND_STAGE_LLM,
        TOP_N_FIRST,
        TOP_N_FINAL,
    )
    with open(run_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(convert_paths_to_relative(run_manifest), handle, indent=2, ensure_ascii=False)
    print(f"Run manifest saved to: {to_project_relative_path(run_manifest_path)}")

    if ENABLE_EXPLANATIONS:
        generate_explanation_reports()


if __name__ == "__main__":
    main()

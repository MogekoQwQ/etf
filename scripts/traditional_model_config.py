"""Shared conventions, model factories, and project-relative path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_TRADITIONAL_MODEL = "random_forest"
DEFAULT_TARGET = "Y_future_5d_return"
DEFAULT_TOP_N_FIRST = 50
DEFAULT_TOP_N_FINAL = 10
PLANNED_TRADITIONAL_MODELS = (
    "random_forest",
    "linear",
    "lightgbm",
    "xgboost",
)
IMPLEMENTED_TRADITIONAL_MODELS = PLANNED_TRADITIONAL_MODELS
EXPERIMENT_RESULT_VARIANTS = (
    "baseline",
    "llm",
    "mock_llm",
)

TRADITIONAL_MODEL_LABELS = {
    "random_forest": "RandomForest",
    "linear": "LinearRegression",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}

TRADITIONAL_MODEL_ALIASES = {
    "rf": "random_forest",
    "randomforest": "random_forest",
    "random_forest": "random_forest",
    "linear": "linear",
    "lr": "linear",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "xgboost": "xgboost",
    "xgb": "xgboost",
}

TARGET_COLUMNS = (
    "Y_next_day_return",
    "Y_future_5d_return",
    "Y_future_10d_return",
    "Y_future_5d_vol_change",
    "Y_future_10d_vol_change",
)

RANDOM_FOREST_PARAMS = {
    "n_estimators": 800,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "random_state": 42,
}

LINEAR_PARAMS: dict[str, Any] = {}

LIGHTGBM_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "verbosity": -1,
}

XGBOOST_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "objective": "reg:squarederror",
    "verbosity": 0,
}


def _as_path(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve()


def normalize_traditional_model(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower().replace("-", "_")
    normalized = TRADITIONAL_MODEL_ALIASES.get(normalized, normalized)
    if normalized not in PLANNED_TRADITIONAL_MODELS:
        raise ValueError(
            f"Unsupported traditional model: {model_name}. "
            f"Available options: {', '.join(PLANNED_TRADITIONAL_MODELS)}"
        )
    return normalized


def ensure_implemented_traditional_model(model_name: str) -> str:
    normalized = normalize_traditional_model(model_name)
    if normalized not in IMPLEMENTED_TRADITIONAL_MODELS:
        raise NotImplementedError(
            f"Traditional model is registered but not implemented yet: {normalized}"
        )
    return normalized


def get_traditional_model_label(model_name: str) -> str:
    normalized = normalize_traditional_model(model_name)
    return TRADITIONAL_MODEL_LABELS.get(normalized, normalized)


def _missing_dependency_error(model_name: str, package_name: str, exc: ImportError) -> ImportError:
    normalized = normalize_traditional_model(model_name)
    return ImportError(
        f"The `{normalized}` traditional model requires the `{package_name}` package. "
        "Install project dependencies with `pip install -r requirements.txt` "
        f"or install `{package_name}` manually. Original error: {exc}"
    )


def build_traditional_estimator(model_name: str):
    normalized = ensure_implemented_traditional_model(model_name)

    if normalized == "random_forest":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise _missing_dependency_error(normalized, "scikit-learn", exc) from exc
        return RandomForestRegressor(**RANDOM_FOREST_PARAMS)

    if normalized == "linear":
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError as exc:
            raise _missing_dependency_error(normalized, "scikit-learn", exc) from exc
        return LinearRegression(**LINEAR_PARAMS)

    if normalized == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise _missing_dependency_error(normalized, "lightgbm", exc) from exc
        return LGBMRegressor(**LIGHTGBM_PARAMS)

    if normalized == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise _missing_dependency_error(normalized, "xgboost", exc) from exc
        return XGBRegressor(**XGBOOST_PARAMS)

    raise NotImplementedError(f"Unsupported traditional model builder: {normalized}")


def ensure_parent_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_data_root(project_root: str | Path) -> Path:
    return _as_path(project_root) / "data"


def get_data_raw_dir(project_root: str | Path) -> Path:
    return get_data_root(project_root) / "raw"


def get_data_processed_dir(project_root: str | Path) -> Path:
    return get_data_root(project_root) / "processed"


def get_etf_list_file(project_root: str | Path) -> Path:
    return get_data_raw_dir(project_root) / "etf_list.csv"


def get_etf_daily_dir(project_root: str | Path) -> Path:
    return get_data_raw_dir(project_root) / "etf_daily"


def get_download_progress_file(project_root: str | Path) -> Path:
    return get_data_raw_dir(project_root) / "download_progress.txt"


def get_factor_data_dir(project_root: str | Path) -> Path:
    return get_data_processed_dir(project_root) / "factor_data"


def get_all_factors_file(project_root: str | Path) -> Path:
    return get_data_processed_dir(project_root) / "all_etf_factors.csv"


def get_dataset_manifest_file(project_root: str | Path) -> Path:
    return get_data_processed_dir(project_root) / "dataset_manifest.json"


def get_traditional_prediction_dir(project_root: str | Path, model_name: str) -> Path:
    return get_data_root(project_root) / "predictions" / normalize_traditional_model(model_name)


def get_traditional_prediction_file(project_root: str | Path, model_name: str) -> Path:
    return get_traditional_prediction_dir(project_root, model_name) / "test_set_with_predictions.csv"


def get_traditional_split_info_file(project_root: str | Path, model_name: str) -> Path:
    return get_traditional_prediction_dir(project_root, model_name) / "split_info.json"


def get_traditional_prediction_manifest_file(project_root: str | Path, model_name: str) -> Path:
    return get_traditional_prediction_dir(project_root, model_name) / "prediction_manifest.json"


def get_runs_dir(project_root: str | Path) -> Path:
    return _as_path(project_root) / "runs"


def get_training_runs_root(project_root: str | Path) -> Path:
    return get_runs_dir(project_root) / "training"


def get_traditional_training_eval_dir(project_root: str | Path, model_name: str) -> Path:
    return get_training_runs_root(project_root) / normalize_traditional_model(model_name)


def get_traditional_training_manifest_file(project_root: str | Path, model_name: str) -> Path:
    return get_traditional_training_eval_dir(project_root, model_name) / "run_manifest.json"


def get_backtests_root(project_root: str | Path) -> Path:
    return get_runs_dir(project_root) / "backtests"


def get_traditional_two_stage_dir(project_root: str | Path, model_name: str) -> Path:
    return get_backtests_root(project_root) / normalize_traditional_model(model_name)


def normalize_second_stage_llm_name(llm_model: str) -> str:
    return str(llm_model or "").strip().replace("-", "_").replace(".", "_")


def get_backtest_scenario_name(
    top_n_first: int = DEFAULT_TOP_N_FIRST,
    top_n_final: int = DEFAULT_TOP_N_FINAL,
) -> str:
    if int(top_n_first) == DEFAULT_TOP_N_FIRST and int(top_n_final) == DEFAULT_TOP_N_FINAL:
        return "default"
    return f"top{int(top_n_first)}_top{int(top_n_final)}"


def get_traditional_two_stage_output_dir(
    project_root: str | Path,
    model_name: str,
    llm_model: str,
    top_n_first: int = DEFAULT_TOP_N_FIRST,
    top_n_final: int = DEFAULT_TOP_N_FINAL,
) -> Path:
    return (
        get_traditional_two_stage_dir(project_root, model_name)
        / normalize_second_stage_llm_name(llm_model)
        / get_backtest_scenario_name(top_n_first, top_n_final)
    )


def get_traditional_two_stage_reports_dir(
    project_root: str | Path,
    model_name: str,
    llm_model: str = "deepseek-chat",
    top_n_first: int = DEFAULT_TOP_N_FIRST,
    top_n_final: int = DEFAULT_TOP_N_FINAL,
) -> Path:
    return get_traditional_two_stage_output_dir(
        project_root,
        model_name,
        llm_model,
        top_n_first,
        top_n_final,
    ) / "reports"


def get_traditional_two_stage_report_file(
    project_root: str | Path,
    model_name: str,
    llm_model: str = "deepseek-chat",
    top_n_first: int = DEFAULT_TOP_N_FIRST,
    top_n_final: int = DEFAULT_TOP_N_FINAL,
) -> Path:
    return get_traditional_two_stage_reports_dir(
        project_root,
        model_name,
        llm_model,
        top_n_first,
        top_n_final,
    ) / "report_summary.html"


def get_traditional_two_stage_manifest_file(
    project_root: str | Path,
    model_name: str,
    llm_model: str,
    top_n_first: int = DEFAULT_TOP_N_FIRST,
    top_n_final: int = DEFAULT_TOP_N_FINAL,
) -> Path:
    return get_traditional_two_stage_output_dir(
        project_root,
        model_name,
        llm_model,
        top_n_first,
        top_n_final,
    ) / "run_manifest.json"


def get_traditional_llm_log_dir(project_root: str | Path, model_name: str, llm_model: str) -> Path:
    return (
        get_traditional_two_stage_dir(project_root, model_name)
        / "logs"
        / normalize_second_stage_llm_name(llm_model)
    )


def get_traditional_explanation_dir(project_root: str | Path, model_name: str, llm_model: str) -> Path:
    return get_traditional_llm_log_dir(project_root, model_name, llm_model) / "explanations"


def get_traditional_benchmark_dir(
    project_root: str | Path,
    model_name: str,
    benchmark_name: str,
    top_n_first: int = DEFAULT_TOP_N_FIRST,
    top_n_final: int = DEFAULT_TOP_N_FINAL,
) -> Path:
    return (
        get_traditional_two_stage_dir(project_root, model_name)
        / "benchmarks"
        / get_backtest_scenario_name(top_n_first, top_n_final)
        / str(benchmark_name or "").strip()
    )


def get_shared_benchmark_dir(
    project_root: str | Path,
    benchmark_name: str,
    top_n_first: int = DEFAULT_TOP_N_FIRST,
    top_n_final: int = DEFAULT_TOP_N_FINAL,
) -> Path:
    return (
        get_backtests_root(project_root)
        / "benchmarks"
        / get_backtest_scenario_name(top_n_first, top_n_final)
        / str(benchmark_name or "").strip()
    )


def get_experiments_root(project_root: str | Path) -> Path:
    return _as_path(project_root) / "experiments"


def get_experiment_snapshots_dir(project_root: str | Path) -> Path:
    return get_experiments_root(project_root) / "snapshots"


def get_traditional_experiment_runs_dir(project_root: str | Path, model_name: str) -> Path:
    return get_experiment_snapshots_dir(project_root) / "exp_main" / normalize_traditional_model(model_name)


def get_experiment_snapshot_runner_name(variant: str, llm_model: str | None = None) -> str:
    normalized_variant = str(variant or "").strip().lower()
    if normalized_variant not in EXPERIMENT_RESULT_VARIANTS:
        raise ValueError(
            f"Unsupported experiment result variant: {variant}. "
            f"Available options: {', '.join(EXPERIMENT_RESULT_VARIANTS)}"
        )
    if normalized_variant == "baseline":
        return "baseline"
    if normalized_variant == "mock_llm":
        return "mock_llm"
    return normalize_second_stage_llm_name(llm_model or "deepseek-chat")


def get_traditional_experiment_run_dir(
    project_root: str | Path,
    model_name: str,
    variant: str,
    llm_model: str | None = None,
    top_n_first: int = DEFAULT_TOP_N_FIRST,
    top_n_final: int = DEFAULT_TOP_N_FINAL,
) -> Path:
    return (
        get_traditional_experiment_runs_dir(project_root, model_name)
        / str(variant or "").strip().lower()
        / get_experiment_snapshot_runner_name(variant, llm_model)
        / get_backtest_scenario_name(top_n_first, top_n_final)
    )


def get_experiment_summary_dir(project_root: str | Path) -> Path:
    return get_experiments_root(project_root) / "summary"


def get_config_dir(project_root: str | Path) -> Path:
    return _as_path(project_root) / "config"


def get_llm_config_example_file(project_root: str | Path) -> Path:
    return get_config_dir(project_root) / "llm_config.example.yaml"


def get_llm_config_local_file(project_root: str | Path) -> Path:
    return get_config_dir(project_root) / "llm_config.local.yaml"


def get_backtest_defaults_file(project_root: str | Path) -> Path:
    return get_config_dir(project_root) / "backtest_defaults.yaml"


def get_experiment_defaults_file(project_root: str | Path) -> Path:
    return get_config_dir(project_root) / "experiment_defaults.yaml"


def get_reports_dir(project_root: str | Path) -> Path:
    return _as_path(project_root) / "reports"


def get_reports_notes_dir(project_root: str | Path) -> Path:
    return get_reports_dir(project_root) / "notes"


def get_reports_thesis_assets_dir(project_root: str | Path) -> Path:
    return get_reports_dir(project_root) / "thesis_assets"

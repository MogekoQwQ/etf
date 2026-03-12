"""Shared conventions for stage-one traditional model outputs."""

from __future__ import annotations

import os


DEFAULT_TRADITIONAL_MODEL = "random_forest"
PLANNED_TRADITIONAL_MODELS = (
    "random_forest",
    "linear",
    "lightgbm",
    "xgboost",
)
IMPLEMENTED_TRADITIONAL_MODELS = (DEFAULT_TRADITIONAL_MODEL,)

TRADITIONAL_MODEL_LABELS = {
    "random_forest": "RandomForest",
    "linear": "Linear",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}

TRADITIONAL_MODEL_ALIASES = {
    "rf": "random_forest",
    "randomforest": "random_forest",
    "random_forest": "random_forest",
    "linear": "linear",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
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


def normalize_traditional_model(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower().replace("-", "_")
    normalized = TRADITIONAL_MODEL_ALIASES.get(normalized, normalized)
    if normalized not in PLANNED_TRADITIONAL_MODELS:
        raise ValueError(
            f"不支持的传统模型: {model_name}. "
            f"可选项: {', '.join(PLANNED_TRADITIONAL_MODELS)}"
        )
    return normalized


def ensure_implemented_traditional_model(model_name: str) -> str:
    normalized = normalize_traditional_model(model_name)
    if normalized not in IMPLEMENTED_TRADITIONAL_MODELS:
        raise NotImplementedError(
            f"当前项目仅实现 {DEFAULT_TRADITIONAL_MODEL} 基线。"
            f"已为后续扩展预留 {normalized} 的目录与参数接口，但尚未实现训练逻辑。"
        )
    return normalized


def get_traditional_model_label(model_name: str) -> str:
    normalized = normalize_traditional_model(model_name)
    return TRADITIONAL_MODEL_LABELS.get(normalized, normalized)


def get_traditional_prediction_dir(project_root: str, model_name: str) -> str:
    return os.path.join(project_root, "data", "predictions", normalize_traditional_model(model_name))


def get_traditional_prediction_file(project_root: str, model_name: str) -> str:
    return os.path.join(get_traditional_prediction_dir(project_root, model_name), "test_set_with_predictions.csv")


def get_traditional_split_info_file(project_root: str, model_name: str) -> str:
    return os.path.join(get_traditional_prediction_dir(project_root, model_name), "split_info.json")


def get_traditional_results_root(project_root: str, model_name: str) -> str:
    return os.path.join(project_root, "results", "traditional_models", normalize_traditional_model(model_name))


def get_traditional_training_eval_dir(project_root: str, model_name: str) -> str:
    return os.path.join(get_traditional_results_root(project_root, model_name), "training_eval")


def get_traditional_two_stage_dir(project_root: str, model_name: str) -> str:
    return os.path.join(get_traditional_results_root(project_root, model_name), "two_stage")


def get_traditional_llm_log_dir(project_root: str, model_name: str) -> str:
    return os.path.join(get_traditional_two_stage_dir(project_root, model_name), "llm_logs")


def get_traditional_explanation_dir(project_root: str, model_name: str) -> str:
    return os.path.join(get_traditional_llm_log_dir(project_root, model_name), "explanations")

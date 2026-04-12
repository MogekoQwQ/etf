"""Train and evaluate the stage-one traditional model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from traditional_model_config import (
    DEFAULT_TARGET,
    DEFAULT_TRADITIONAL_MODEL,
    PLANNED_TRADITIONAL_MODELS,
    TARGET_COLUMNS,
    build_traditional_estimator,
    ensure_implemented_traditional_model,
    get_all_factors_file,
    get_traditional_model_label,
    get_traditional_prediction_dir,
    get_traditional_prediction_manifest_file,
    get_traditional_split_info_file,
    get_traditional_training_eval_dir,
    get_traditional_training_manifest_file,
)


matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = get_all_factors_file(PROJECT_ROOT)
DATE_COLUMN = "日期"
RANKING_TOP_COUNTS = (10, 50)
CROSS_SECTIONAL_CLIP_QUANTILES = (0.01, 0.99)
CROSS_SECTIONAL_CANDIDATE_FEATURES = (
    "momentum_20",
    "volatility_20",
    "volume_mean_20",
    "return_mean_20",
    "amplitude_mean_20",
    "turnover_mean_20",
    "MA_5",
    "MA_10",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the stage-one traditional model and export evaluation and prediction artifacts."
    )
    parser.add_argument(
        "--traditional-model",
        default=DEFAULT_TRADITIONAL_MODEL,
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="Traditional model name: random_forest, linear, lightgbm, xgboost.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=list(TARGET_COLUMNS),
        help="Target column to train and evaluate.",
    )
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to the merged ETF factor dataset.")
    return parser.parse_args()


def build_estimator(model_name: str):
    return build_traditional_estimator(model_name)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude_cols = ["code", "name", DATE_COLUMN, *TARGET_COLUMNS]
    return [column for column in df.columns if column not in exclude_cols]


def get_cross_sectional_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        feature
        for feature in CROSS_SECTIONAL_CANDIDATE_FEATURES
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature])
    ]


def build_cross_sectional_zscore_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = get_cross_sectional_feature_columns(df)
    if not feature_columns:
        return df, []

    lower_q, upper_q = CROSS_SECTIONAL_CLIP_QUANTILES
    enhanced_df = df.copy()
    grouped_dates = enhanced_df.groupby(DATE_COLUMN, sort=False)

    for feature in feature_columns:
        clipped_feature = grouped_dates[feature].transform(
            lambda series: series.clip(
                lower=series.quantile(lower_q),
                upper=series.quantile(upper_q),
            )
        )
        clipped_grouped = clipped_feature.groupby(enhanced_df[DATE_COLUMN], sort=False)
        feature_mean = clipped_grouped.transform("mean")
        feature_std = clipped_grouped.transform("std")
        zscore_feature = (clipped_feature - feature_mean) / feature_std
        enhanced_df[f"{feature}_cs_z"] = zscore_feature.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return enhanced_df, [f"{feature}_cs_z" for feature in feature_columns]


def extract_feature_scores(model, feature_cols: Iterable[str]) -> pd.DataFrame:
    feature_list = list(feature_cols)
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
        score_name = "importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        scores = np.abs(coef.reshape(-1))
        score_name = "abs_coef"
    else:
        scores = np.zeros(len(feature_list), dtype=float)
        score_name = "score"

    return (
        pd.DataFrame({"feature": feature_list, score_name: scores})
        .sort_values(score_name, ascending=False)
        .reset_index(drop=True)
    )


def resolve_output_path(preferred_path: str) -> str:
    os.makedirs(os.path.dirname(preferred_path), exist_ok=True)
    try:
        file_descriptor = os.open(preferred_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY)
        os.close(file_descriptor)
        return preferred_path
    except PermissionError:
        root, extension = os.path.splitext(preferred_path)
        fallback_path = f"{root}.latest{extension}"
        print(f"Output file is locked, writing to fallback path instead: {fallback_path}")
        return fallback_path


def save_prediction_plot(y_true: pd.Series, y_pred: pd.Series, output_path: str, target_name: str) -> None:
    output_path = resolve_output_path(output_path)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Prediction vs Actual: {target_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_feature_plot(feature_scores: pd.DataFrame, score_name: str, output_path: str, target_name: str) -> None:
    output_path = resolve_output_path(output_path)
    top_features = feature_scores.head(10)
    score_label_map = {
        "importance": "Importance",
        "abs_coef": "Absolute Coefficient",
        "coefficient_abs": "Absolute Coefficient",
        "coefficient": "Coefficient",
    }
    score_label = score_label_map.get(score_name, score_name)
    plt.figure(figsize=(8, 6))
    plt.barh(top_features["feature"][::-1], top_features[score_name][::-1])
    plt.xlabel(score_label)
    plt.ylabel("Feature")
    plt.title(f"Top 10 Features: {target_name}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def safe_correlation(left: pd.Series, right: pd.Series, method: str = "pearson") -> float:
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 2:
        return float("nan")
    if valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def compute_ranking_metrics_by_date(
    test_df: pd.DataFrame,
    target_name: str,
    prediction_col: str,
) -> pd.DataFrame:
    ranking_source = test_df[[DATE_COLUMN, "code", target_name, prediction_col]].dropna(
        subset=[target_name, prediction_col]
    )
    results = []

    for date_value, day_df in ranking_source.groupby(DATE_COLUMN):
        ranked_day_df = day_df.sort_values(prediction_col, ascending=False)
        universe_mean_target = float(day_df[target_name].mean())

        top_10_df = ranked_day_df.head(RANKING_TOP_COUNTS[0])
        top_50_df = ranked_day_df.head(RANKING_TOP_COUNTS[1])
        top_10_mean_target = float(top_10_df[target_name].mean()) if not top_10_df.empty else float("nan")
        top_50_mean_target = float(top_50_df[target_name].mean()) if not top_50_df.empty else float("nan")

        results.append(
            {
                "target": target_name,
                "date": pd.Timestamp(date_value),
                "sample_count": int(len(day_df)),
                "top10_count": int(len(top_10_df)),
                "top50_count": int(len(top_50_df)),
                "ic": safe_correlation(day_df[prediction_col], day_df[target_name], method="pearson"),
                "rank_ic": safe_correlation(day_df[prediction_col], day_df[target_name], method="spearman"),
                "universe_mean_target": universe_mean_target,
                "top10_mean_target": top_10_mean_target,
                "top50_mean_target": top_50_mean_target,
                "top10_excess_target": top_10_mean_target - universe_mean_target,
                "top50_excess_target": top_50_mean_target - universe_mean_target,
            }
        )

    ranking_metrics_by_date = pd.DataFrame(results)
    if ranking_metrics_by_date.empty:
        return ranking_metrics_by_date
    return ranking_metrics_by_date.sort_values("date").reset_index(drop=True)


def summarize_ranking_metrics(
    test_df: pd.DataFrame,
    ranking_metrics_by_date: pd.DataFrame,
    target_name: str,
    prediction_col: str,
) -> pd.DataFrame:
    overall_source = test_df[[target_name, prediction_col]].dropna()

    summary = {
        "target": target_name,
        "prediction_col": prediction_col,
        "evaluated_dates": int(ranking_metrics_by_date["date"].nunique()) if not ranking_metrics_by_date.empty else 0,
        "evaluated_rows": int(len(overall_source)),
        "overall_ic": safe_correlation(overall_source[prediction_col], overall_source[target_name], method="pearson"),
        "overall_rank_ic": safe_correlation(
            overall_source[prediction_col], overall_source[target_name], method="spearman"
        ),
        "ic": float(ranking_metrics_by_date["ic"].mean()) if not ranking_metrics_by_date.empty else float("nan"),
        "rank_ic": (
            float(ranking_metrics_by_date["rank_ic"].mean()) if not ranking_metrics_by_date.empty else float("nan")
        ),
        "universe_mean_target": (
            float(ranking_metrics_by_date["universe_mean_target"].mean())
            if not ranking_metrics_by_date.empty
            else float("nan")
        ),
        "top10_mean_target": (
            float(ranking_metrics_by_date["top10_mean_target"].mean())
            if not ranking_metrics_by_date.empty
            else float("nan")
        ),
        "top50_mean_target": (
            float(ranking_metrics_by_date["top50_mean_target"].mean())
            if not ranking_metrics_by_date.empty
            else float("nan")
        ),
        "top10_excess_target": (
            float(ranking_metrics_by_date["top10_excess_target"].mean())
            if not ranking_metrics_by_date.empty
            else float("nan")
        ),
        "top50_excess_target": (
            float(ranking_metrics_by_date["top50_excess_target"].mean())
            if not ranking_metrics_by_date.empty
            else float("nan")
        ),
    }
    return pd.DataFrame([summary])


def evaluate_target(
    model_name: str,
    target_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str,
) -> Tuple[pd.Series, Dict[str, float], pd.DataFrame]:
    estimator = build_estimator(model_name)
    estimator.fit(train_df[feature_cols], train_df[target_name])
    predictions = pd.Series(estimator.predict(test_df[feature_cols]), index=test_df.index)

    metrics = {
        "r2": r2_score(test_df[target_name], predictions),
        "mae": mean_absolute_error(test_df[target_name], predictions),
        "rmse": float(np.sqrt(mean_squared_error(test_df[target_name], predictions))),
    }

    feature_scores = extract_feature_scores(estimator, feature_cols)
    score_name = feature_scores.columns[1]

    save_prediction_plot(
        y_true=test_df[target_name],
        y_pred=predictions,
        output_path=os.path.join(output_dir, f"pred_vs_actual_{target_name}.png"),
        target_name=target_name,
    )
    save_feature_plot(
        feature_scores=feature_scores,
        score_name=score_name,
        output_path=os.path.join(output_dir, f"feature_importance_{target_name}.png"),
        target_name=target_name,
    )
    return predictions, metrics, feature_scores


def main() -> int:
    args = parse_args()
    traditional_model = ensure_implemented_traditional_model(args.traditional_model)
    model_label = get_traditional_model_label(traditional_model)
    target_name = args.target

    output_dir = get_traditional_training_eval_dir(PROJECT_ROOT, traditional_model)
    prediction_dir = get_traditional_prediction_dir(PROJECT_ROOT, traditional_model)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    print(f"Stage-one traditional model training started: {model_label}")
    print(f"Input data: {args.data_path}")
    print(f"Target: {target_name}")
    print(f"Training evaluation output dir: {output_dir}")
    print(f"Prediction output dir: {prediction_dir}")

    df = pd.read_csv(args.data_path, parse_dates=[DATE_COLUMN], encoding="utf-8-sig")
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    df, cross_sectional_feature_cols = build_cross_sectional_zscore_features(df)

    feature_cols = get_feature_columns(df)
    split_date = df[DATE_COLUMN].quantile(0.8)
    train_df = df[df[DATE_COLUMN] <= split_date].copy()
    test_df = df[df[DATE_COLUMN] > split_date].copy()

    result_file = resolve_output_path(os.path.join(output_dir, "model_results.txt"))
    ranking_metrics_path = resolve_output_path(os.path.join(output_dir, "ranking_metrics.csv"))
    ranking_metrics_by_date_path = resolve_output_path(os.path.join(output_dir, "ranking_metrics_by_date.csv"))

    with open(result_file, "w", encoding="utf-8") as handle:
        handle.write(f"traditional_model: {traditional_model}\n")
        handle.write(f"model_label: {model_label}\n")
        handle.write(f"target: {target_name}\n")
        handle.write(f"split_date: {split_date.strftime('%Y-%m-%d')}\n")
        handle.write(f"train_samples: {len(train_df)}\n")
        handle.write(f"test_samples: {len(test_df)}\n")
        handle.write("当前实验已启用“按日期去极值 + 横截面标准化”的增强特征。\n")
        handle.write(
            "feature_enhancement: enabled (winsorize by date at 1%/99% + cross-sectional z-score)\n"
        )
        handle.write(f"cross_sectional_feature_count: {len(cross_sectional_feature_cols)}\n")
        handle.write(
            "cross_sectional_feature_columns: "
            f"{', '.join(cross_sectional_feature_cols) if cross_sectional_feature_cols else 'none'}\n"
        )

        print(f"\nProcessing target: {target_name}")
        predictions, metrics, feature_scores = evaluate_target(
            model_name=traditional_model,
            target_name=target_name,
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            output_dir=output_dir,
        )
        prediction_col = f"y_pred_{target_name}"
        test_df[prediction_col] = predictions

        handle.write(f"\nTarget: {target_name}\n")
        handle.write(f"R^2: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}\n")
        handle.write("Top 10 features:\n")
        handle.write(feature_scores.head(10).to_string(index=False))
        handle.write("\n")

        print(
            f"R^2: {metrics['r2']:.4f}, "
            f"MAE: {metrics['mae']:.4f}, "
            f"RMSE: {metrics['rmse']:.4f}"
        )

        ranking_metrics_by_date = compute_ranking_metrics_by_date(
            test_df=test_df,
            target_name=target_name,
            prediction_col=prediction_col,
        )
        ranking_metrics = summarize_ranking_metrics(
            test_df=test_df,
            ranking_metrics_by_date=ranking_metrics_by_date,
            target_name=target_name,
            prediction_col=prediction_col,
        )
        ranking_metrics.to_csv(ranking_metrics_path, index=False, encoding="utf-8-sig")
        ranking_metrics_by_date.to_csv(
            ranking_metrics_by_date_path,
            index=False,
            date_format="%Y-%m-%d",
            encoding="utf-8-sig",
        )

        ranking_summary = ranking_metrics.iloc[0].to_dict()
        handle.write(f"\nRanking metrics summary ({target_name}):\n")
        handle.write(
            "IC(mean_by_date): {ic:.4f}, "
            "RankIC(mean_by_date): {rank_ic:.4f}, "
            "IC(overall): {overall_ic:.4f}, "
            "RankIC(overall): {overall_rank_ic:.4f}\n".format(**ranking_summary)
        )
        handle.write(
            "Top10 mean target: {top10_mean_target:.4f}, "
            "Top50 mean target: {top50_mean_target:.4f}, "
            "Universe mean target: {universe_mean_target:.4f}\n".format(**ranking_summary)
        )
        handle.write(
            "Top10 excess target: {top10_excess_target:.4f}, "
            "Top50 excess target: {top50_excess_target:.4f}, "
            "Evaluated dates: {evaluated_dates}, "
            "Evaluated rows: {evaluated_rows}\n".format(**ranking_summary)
        )

        print(
            "Ranking metrics "
            f"IC(mean_by_date): {ranking_summary['ic']:.4f}, "
            f"RankIC(mean_by_date): {ranking_summary['rank_ic']:.4f}, "
            f"Top10 excess: {ranking_summary['top10_excess_target']:.4f}, "
            f"Top50 excess: {ranking_summary['top50_excess_target']:.4f}"
        )

    prediction_file = resolve_output_path(os.path.join(prediction_dir, "test_set_with_predictions.csv"))
    test_df.to_csv(prediction_file, index=False, encoding="utf-8-sig")

    split_info = {
        "traditional_model": traditional_model,
        "model_label": model_label,
        "target": target_name,
        "split_date": split_date.strftime("%Y-%m-%d"),
        "train_start": train_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "train_end": train_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "test_start": test_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "test_end": test_df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }
    split_info_path = resolve_output_path(get_traditional_split_info_file(PROJECT_ROOT, traditional_model))
    with open(split_info_path, "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2, ensure_ascii=False)

    training_manifest = {
        "model": traditional_model,
        "target": target_name,
        "data_path": args.data_path,
        "split_date": split_date.strftime("%Y-%m-%d"),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    training_manifest_path = resolve_output_path(get_traditional_training_manifest_file(PROJECT_ROOT, traditional_model))
    with open(training_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(training_manifest, handle, indent=2, ensure_ascii=False)

    prediction_manifest = {
        "model": traditional_model,
        "target": target_name,
        "data_path": args.data_path,
        "prediction_file": prediction_file,
        "split_info_file": get_traditional_split_info_file(PROJECT_ROOT, traditional_model),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    prediction_manifest_path = resolve_output_path(get_traditional_prediction_manifest_file(PROJECT_ROOT, traditional_model))
    with open(prediction_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(prediction_manifest, handle, indent=2, ensure_ascii=False)

    print(f"\nEvaluation results saved to: {output_dir}")
    print(f"Prediction file saved to: {prediction_file}")
    print(f"Split info saved to: {split_info_path}")
    print(f"Training manifest saved to: {training_manifest_path}")
    print(f"Prediction manifest saved to: {prediction_manifest_path}")
    print(f"Ranking metrics saved to: {ranking_metrics_path}")
    print(f"Ranking metrics by date saved to: {ranking_metrics_by_date_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train and evaluate the stage-one traditional model baseline."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from traditional_model_config import (
    DEFAULT_TRADITIONAL_MODEL,
    PLANNED_TRADITIONAL_MODELS,
    RANDOM_FOREST_PARAMS,
    TARGET_COLUMNS,
    ensure_implemented_traditional_model,
    get_traditional_model_label,
    get_traditional_prediction_dir,
    get_traditional_training_eval_dir,
)


matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "all_etf_factors.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练第一阶段传统模型并生成评估结果与测试集预测文件")
    parser.add_argument(
        "--traditional-model",
        default=DEFAULT_TRADITIONAL_MODEL,
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="传统模型名称。当前仅 random_forest 已实现，其余选项仅为后续扩展预留。",
    )
    parser.add_argument("--data-path", default=DATA_PATH, help="合并后的 ETF 因子总表路径")
    return parser.parse_args()


def build_estimator(model_name: str):
    implemented_model = ensure_implemented_traditional_model(model_name)
    if implemented_model == "random_forest":
        return RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    raise NotImplementedError(f"未实现的传统模型: {implemented_model}")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude_cols = ["code", "name", "日期", *TARGET_COLUMNS]
    return [column for column in df.columns if column not in exclude_cols]


def extract_feature_scores(model, feature_cols: Iterable[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
        score_name = "importance"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        scores = np.abs(coef.reshape(-1))
        score_name = "abs_coef"
    else:
        raise ValueError("当前模型未暴露可直接输出的特征重要性或系数。")

    return (
        pd.DataFrame({"feature": list(feature_cols), score_name: scores})
        .sort_values(score_name, ascending=False)
        .reset_index(drop=True)
    )


def save_prediction_plot(y_true: pd.Series, y_pred: pd.Series, output_path: str, target_name: str) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("实际值")
    plt.ylabel("预测值")
    plt.title(f"预测值 vs 实际值 ({target_name})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_feature_plot(feature_scores: pd.DataFrame, score_name: str, output_path: str, target_name: str) -> None:
    top_features = feature_scores.head(10)
    plt.figure(figsize=(8, 6))
    plt.barh(top_features["feature"][::-1], top_features[score_name][::-1])
    plt.xlabel(score_name)
    plt.title(f"Top 10 特征贡献 ({target_name})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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

    output_dir = get_traditional_training_eval_dir(PROJECT_ROOT, traditional_model)
    prediction_dir = get_traditional_prediction_dir(PROJECT_ROOT, traditional_model)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)

    print(f"阶段一传统模型训练开始: {model_label}")
    print(f"输入数据: {args.data_path}")
    print(f"训练评估输出目录: {output_dir}")
    print(f"预测输出目录: {prediction_dir}")

    df = pd.read_csv(args.data_path, parse_dates=["日期"], encoding="utf-8-sig")
    df = df.sort_values("日期").reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    split_date = df["日期"].quantile(0.8)
    train_df = df[df["日期"] <= split_date].copy()
    test_df = df[df["日期"] > split_date].copy()

    result_file = os.path.join(output_dir, "model_results.txt")
    with open(result_file, "w", encoding="utf-8") as handle:
        handle.write(f"traditional_model: {traditional_model}\n")
        handle.write(f"model_label: {model_label}\n")
        handle.write(f"split_date: {split_date.strftime('%Y-%m-%d')}\n")
        handle.write(f"train_samples: {len(train_df)}\n")
        handle.write(f"test_samples: {len(test_df)}\n")

        for target_name in TARGET_COLUMNS:
            print(f"\n处理目标: {target_name}")
            predictions, metrics, feature_scores = evaluate_target(
                model_name=traditional_model,
                target_name=target_name,
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                output_dir=output_dir,
            )
            test_df[f"y_pred_{target_name}"] = predictions

            handle.write(f"\n目标: {target_name}\n")
            handle.write(
                f"R^2: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}\n"
            )
            handle.write("Top 10 特征:\n")
            handle.write(feature_scores.head(10).to_string(index=False))
            handle.write("\n")

            print(
                f"R^2: {metrics['r2']:.4f}, "
                f"MAE: {metrics['mae']:.4f}, "
                f"RMSE: {metrics['rmse']:.4f}"
            )

    prediction_file = os.path.join(prediction_dir, "test_set_with_predictions.csv")
    test_df.to_csv(prediction_file, index=False)

    split_info = {
        "traditional_model": traditional_model,
        "model_label": model_label,
        "split_date": split_date.strftime("%Y-%m-%d"),
        "train_start": train_df["日期"].min().strftime("%Y-%m-%d"),
        "train_end": train_df["日期"].max().strftime("%Y-%m-%d"),
        "test_start": test_df["日期"].min().strftime("%Y-%m-%d"),
        "test_end": test_df["日期"].max().strftime("%Y-%m-%d"),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }
    split_info_path = os.path.join(prediction_dir, "split_info.json")
    with open(split_info_path, "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2, ensure_ascii=False)

    print(f"\n评估结果已保存至: {output_dir}")
    print(f"测试集预测数据已保存至: {prediction_file}")
    print(f"数据切分信息已保存至: {split_info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

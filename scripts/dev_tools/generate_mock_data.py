"""Generate lightweight mock data for local validation."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from traditional_model_config import DEFAULT_TRADITIONAL_MODEL, TARGET_COLUMNS, get_traditional_prediction_dir


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ETF_DATA_DIR = os.path.join(DATA_DIR, "etf_data")
FACTOR_DATA_DIR = os.path.join(DATA_DIR, "factor_data")
PREDICTION_DIR = get_traditional_prediction_dir(PROJECT_ROOT, DEFAULT_TRADITIONAL_MODEL)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ETF_DATA_DIR, exist_ok=True)
os.makedirs(FACTOR_DATA_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

etf_list = [
    {"code": "510760", "name": "上证综指ETF"},
    {"code": "159903", "name": "深成ETF"},
    {"code": "510050", "name": "上证50ETF"},
    {"code": "510180", "name": "上证180ETF"},
    {"code": "510300", "name": "沪深300ETF"},
]

dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="B").sort_values(ascending=False)

print("生成 ETF 原始数据...")
for etf in etf_list:
    code = etf["code"]
    name = etf["name"]

    np.random.seed(42 + int(code[:3]))
    base_price = 100.0 + np.random.randn() * 20
    returns = np.random.randn(len(dates)) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))
    volumes = np.random.lognormal(mean=10, sigma=1.5, size=len(dates))
    turnover_rates = np.random.uniform(0.1, 5.0, size=len(dates))

    df = pd.DataFrame(
        {
            "日期": dates,
            "开盘": prices * (1 + np.random.randn(len(dates)) * 0.005),
            "收盘": prices,
            "最高": prices * (1 + np.random.uniform(0, 0.03, size=len(dates))),
            "最低": prices * (1 - np.random.uniform(0, 0.03, size=len(dates))),
            "涨跌幅": returns,
            "振幅": np.random.uniform(0.01, 0.06, size=len(dates)),
            "换手率": turnover_rates,
            "成交量": volumes,
            "成交额": prices * volumes,
        }
    ).sort_values("日期")

    csv_path = os.path.join(ETF_DATA_DIR, f"{code}.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as handle:
        handle.write(f"# code={code}, name={name}\n")
    df.to_csv(csv_path, mode="a", index=False, encoding="utf-8-sig")
    print(f"  生成 {code}.csv: {len(df)} 行")

print("\n生成因子数据...")
all_factors = []
for etf in etf_list:
    code = etf["code"]
    name = etf["name"]
    csv_path = os.path.join(ETF_DATA_DIR, f"{code}.csv")
    df = pd.read_csv(csv_path, skiprows=1, parse_dates=["日期"]).sort_values("日期").reset_index(drop=True)

    window = 20
    df["momentum_20"] = df["收盘"].pct_change(window)
    df["volatility_20"] = df["收盘"].pct_change().rolling(window).std()
    df["volume_mean_20"] = df["成交量"].rolling(window).mean()
    df["return_mean_20"] = df["涨跌幅"].rolling(window).mean()
    df["amplitude_mean_20"] = df["振幅"].rolling(window).mean()
    df["turnover_mean_20"] = df["换手率"].rolling(window).mean()
    df["MA_5"] = df["收盘"].rolling(5).mean()
    df["MA_10"] = df["收盘"].rolling(10).mean()

    future_days_1 = 5
    future_days_2 = 10
    df["Y_next_day_return"] = df["收盘"].pct_change().shift(-1)
    df["Y_future_5d_return"] = df["收盘"].shift(-future_days_1) / df["收盘"] - 1
    df["Y_future_10d_return"] = df["收盘"].shift(-future_days_2) / df["收盘"] - 1
    df["Y_future_5d_vol_change"] = (
        df["收盘"].pct_change().rolling(future_days_1).std().shift(-future_days_1)
        / df["收盘"].pct_change().rolling(future_days_1).std()
        - 1
    )
    df["Y_future_10d_vol_change"] = (
        df["收盘"].pct_change().rolling(future_days_2).std().shift(-future_days_2)
        / df["收盘"].pct_change().rolling(future_days_2).std()
        - 1
    )

    df.insert(0, "code", code)
    df.insert(1, "name", name)

    factor_path = os.path.join(FACTOR_DATA_DIR, f"{code}_factor.csv")
    df.to_csv(factor_path, index=False)
    print(f"  生成 {code}_factor.csv")
    all_factors.append(df)

all_df = pd.concat(all_factors, ignore_index=True).dropna().reset_index(drop=True)
all_df = all_df.sort_values(by=["日期", "code"]).reset_index(drop=True)
output_file = os.path.join(DATA_DIR, "all_etf_factors.csv")
all_df.to_csv(output_file, index=False)
print(f"\n合并后的因子总表已生成: {output_file}")

print("\n生成传统模型预测数据...")
split_date = all_df["日期"].quantile(0.8)
test_df = all_df[all_df["日期"] > split_date].copy()

np.random.seed(42)
for target in TARGET_COLUMNS:
    actual = test_df[target].values
    noise = np.random.randn(len(actual)) * 0.1 * np.std(actual)
    test_df[f"y_pred_{target}"] = actual + noise

predictions_file = os.path.join(PREDICTION_DIR, "test_set_with_predictions.csv")
test_df.to_csv(predictions_file, index=False)
print(f"生成预测文件: {predictions_file}")

split_info = {
    "traditional_model": DEFAULT_TRADITIONAL_MODEL,
    "split_date": split_date.strftime("%Y-%m-%d"),
    "train_start": all_df[all_df["日期"] <= split_date]["日期"].min().strftime("%Y-%m-%d"),
    "train_end": all_df[all_df["日期"] <= split_date]["日期"].max().strftime("%Y-%m-%d"),
    "test_start": test_df["日期"].min().strftime("%Y-%m-%d"),
    "test_end": test_df["日期"].max().strftime("%Y-%m-%d"),
    "train_samples": len(all_df[all_df["日期"] <= split_date]),
    "test_samples": len(test_df),
}
split_info_path = os.path.join(PREDICTION_DIR, "split_info.json")
with open(split_info_path, "w", encoding="utf-8") as handle:
    json.dump(split_info, handle, indent=2, ensure_ascii=False)
print(f"生成切分信息: {split_info_path}")

import os

import pandas as pd

from market_data_utils import ensure_columns, read_csv_with_fallback, standardize_etf_market_columns
from traditional_model_config import get_etf_daily_dir, get_etf_list_file, get_factor_data_dir


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_FOLDER = get_etf_daily_dir(PROJECT_ROOT)
FACTOR_FOLDER = get_factor_data_dir(PROJECT_ROOT)

os.makedirs(FACTOR_FOLDER, exist_ok=True)

ETF_LIST = read_csv_with_fallback(get_etf_list_file(PROJECT_ROOT), dtype={"code": str})
WINDOW = 20
FUTURE_DAYS_1 = 5
FUTURE_DAYS_2 = 10


for _, row in ETF_LIST.iterrows():
    code = str(row["code"])
    csv_path = os.path.join(DATA_FOLDER, f"{code}.csv")

    if not os.path.exists(csv_path):
        print(f"{code} 原始行情文件不存在，跳过。")
        continue

    df = read_csv_with_fallback(csv_path, skiprows=1)
    df = standardize_etf_market_columns(df, allow_position_fallback=True)
    ensure_columns(df, ["日期", "收盘", "成交量", "涨跌幅", "振幅", "换手率"], csv_path)

    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)

    df["momentum_20"] = df["收盘"].pct_change(WINDOW)
    df["volatility_20"] = df["收盘"].pct_change().rolling(WINDOW).std()
    df["volume_mean_20"] = df["成交量"].rolling(WINDOW).mean()
    df["return_mean_20"] = df["涨跌幅"].rolling(WINDOW).mean()
    df["amplitude_mean_20"] = df["振幅"].rolling(WINDOW).mean()
    df["turnover_mean_20"] = df["换手率"].rolling(WINDOW).mean()
    df["MA_5"] = df["收盘"].rolling(5).mean()
    df["MA_10"] = df["收盘"].rolling(10).mean()

    df["Y_next_day_return"] = df["收盘"].pct_change().shift(-1)
    df["Y_future_5d_return"] = df["收盘"].shift(-FUTURE_DAYS_1) / df["收盘"] - 1
    df["Y_future_10d_return"] = df["收盘"].shift(-FUTURE_DAYS_2) / df["收盘"] - 1
    df["Y_future_5d_vol_change"] = (
        df["收盘"].pct_change().rolling(FUTURE_DAYS_1).std().shift(-FUTURE_DAYS_1)
        / df["收盘"].pct_change().rolling(FUTURE_DAYS_1).std()
        - 1
    )
    df["Y_future_10d_vol_change"] = (
        df["收盘"].pct_change().rolling(FUTURE_DAYS_2).std().shift(-FUTURE_DAYS_2)
        / df["收盘"].pct_change().rolling(FUTURE_DAYS_2).std()
        - 1
    )

    output_path = os.path.join(FACTOR_FOLDER, f"{code}_factor.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{code} 因子与标签已保存: {output_path}")

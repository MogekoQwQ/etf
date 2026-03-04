"""
生成模拟的ETF因子数据用于测试
"""
import pandas as pd
import numpy as np
import os

# 创建data目录
os.makedirs("../data", exist_ok=True)
os.makedirs("../data/etf_data", exist_ok=True)
os.makedirs("../data/factor_data", exist_ok=True)
os.makedirs("../data/predictions", exist_ok=True)

# ETF列表（选取前5只）
etf_list = [
    {"code": "510760", "name": "上证综指ETF"},
    {"code": "159903", "name": "深成ETF"},
    {"code": "510050", "name": "上证50ETF"},
    {"code": "510180", "name": "上证180ETF"},
    {"code": "510300", "name": "沪深300ETF"}
]

# 生成日期范围（最近100个交易日）
dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='B')  # 交易日
dates = dates.sort_values(ascending=False)

# 生成每个ETF的模拟日线数据
print("生成ETF原始数据...")
for etf in etf_list:
    code = etf["code"]
    name = etf["name"]

    # 生成随机价格数据
    np.random.seed(42 + int(code[:3]))  # 基于code的种子

    # 基础价格
    base_price = 100.0 + np.random.randn() * 20

    # 随机游走生成价格序列
    returns = np.random.randn(len(dates)) * 0.02  # 日收益率~N(0, 0.02)
    prices = base_price * np.exp(np.cumsum(returns))

    # 生成成交量等数据
    volumes = np.random.lognormal(mean=10, sigma=1.5, size=len(dates))
    turnover_rates = np.random.uniform(0.1, 5.0, size=len(dates))

    # 创建DataFrame
    df = pd.DataFrame({
        "日期": dates,
        "开盘": prices * (1 + np.random.randn(len(dates)) * 0.005),
        "收盘": prices,
        "最高": prices * (1 + np.random.uniform(0, 0.03, size=len(dates))),
        "最低": prices * (1 - np.random.uniform(0, 0.03, size=len(dates))),
        "涨跌幅": returns,
        "振幅": np.random.uniform(0.01, 0.06, size=len(dates)),
        "换手率": turnover_rates,
        "成交量": volumes,
        "成交额": prices * volumes
    })

    # 按日期排序（升序）
    df = df.sort_values("日期")

    # 保存原始数据
    csv_path = f"../data/etf_data/{code}.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write(f"# code={code}, name={name}\n")
    df.to_csv(csv_path, mode="a", index=False, encoding="utf-8-sig")
    print(f"  生成 {code}.csv: {len(df)} 行")

# 现在生成因子数据（模拟compute_etf_factors.py的输出）
print("\n生成因子数据...")
all_factors = []

for etf in etf_list:
    code = etf["code"]
    name = etf["name"]

    # 读取刚生成的原始数据
    csv_path = f"../data/etf_data/{code}.csv"
    df = pd.read_csv(csv_path, skiprows=1, parse_dates=["日期"])
    df = df.sort_values("日期").reset_index(drop=True)

    # 计算因子（简化版，使用随机值）
    window = 20
    df["momentum_20"] = df["收盘"].pct_change(window)
    df["volatility_20"] = df["收盘"].pct_change().rolling(window).std()
    df["volume_mean_20"] = df["成交量"].rolling(window).mean()
    df["return_mean_20"] = df["涨跌幅"].rolling(window).mean()
    df["amplitude_mean_20"] = df["振幅"].rolling(window).mean()
    df["turnover_mean_20"] = df["换手率"].rolling(window).mean()
    df["MA_5"] = df["收盘"].rolling(5).mean()
    df["MA_10"] = df["收盘"].rolling(10).mean()

    # 计算Y值（目标变量）
    future_days_1 = 5
    future_days_2 = 10

    df["Y_next_day_return"] = df["收盘"].pct_change().shift(-1)
    df["Y_future_5d_return"] = df["收盘"].shift(-future_days_1) / df["收盘"] - 1
    df["Y_future_10d_return"] = df["收盘"].shift(-future_days_2) / df["收盘"] - 1
    df["Y_future_5d_vol_change"] = df["收盘"].pct_change().rolling(future_days_1).std().shift(-future_days_1) / \
                                   df["收盘"].pct_change().rolling(future_days_1).std() - 1
    df["Y_future_10d_vol_change"] = df["收盘"].pct_change().rolling(future_days_2).std().shift(-future_days_2) / \
                                    df["收盘"].pct_change().rolling(future_days_2).std() - 1

    # 添加code和name列
    df.insert(0, "code", code)
    df.insert(1, "name", name)

    # 保存单只ETF因子数据
    factor_path = f"../data/factor_data/{code}_factor.csv"
    df.to_csv(factor_path, index=False)
    print(f"  生成 {code}_factor.csv")

    # 添加到总列表
    all_factors.append(df)

# 合并所有ETF数据
print("\n合并所有ETF因子数据...")
all_df = pd.concat(all_factors, ignore_index=True)

# 删除含NaN的行（由于滚动计算）
all_df = all_df.dropna().reset_index(drop=True)

# 按日期排序
all_df = all_df.sort_values(by=["日期", "code"]).reset_index(drop=True)

# 保存合并后的数据
output_file = "../data/all_etf_factors.csv"
all_df.to_csv(output_file, index=False)
print(f"生成完成: {output_file}")
print(f"数据形状: {all_df.shape}")
print(f"时间范围: {all_df['日期'].min()} 到 {all_df['日期'].max()}")
print(f"ETF数量: {all_df['code'].nunique()}")

# 生成预测数据目录（模拟train_rf_multi_target_backtest.py的输出）
print("\n生成预测数据...")
# 简单复制一部分数据作为测试集预测
split_date = all_df["日期"].quantile(0.8)
test_df = all_df[all_df["日期"] > split_date].copy()

# 添加预测列（随机值，接近实际值）
np.random.seed(42)
for target in ["Y_next_day_return", "Y_future_5d_return", "Y_future_10d_return",
               "Y_future_5d_vol_change", "Y_future_10d_vol_change"]:
    if target in test_df.columns:
        # 预测值 = 实际值 + 小随机噪声
        actual = test_df[target].values
        noise = np.random.randn(len(actual)) * 0.1 * np.std(actual)
        test_df[f"y_pred_{target}"] = actual + noise

predictions_file = "../data/predictions/test_set_with_predictions.csv"
test_df.to_csv(predictions_file, index=False)
print(f"生成预测数据: {predictions_file}")

# 保存分割信息
split_info = {
    "split_date": split_date.strftime("%Y-%m-%d"),
    "train_start": all_df[all_df["日期"] <= split_date]["日期"].min().strftime("%Y-%m-%d"),
    "train_end": all_df[all_df["日期"] <= split_date]["日期"].max().strftime("%Y-%m-%d"),
    "test_start": test_df["日期"].min().strftime("%Y-%m-%d"),
    "test_end": test_df["日期"].max().strftime("%Y-%m-%d"),
    "train_samples": len(all_df[all_df["日期"] <= split_date]),
    "test_samples": len(test_df)
}

import json
with open("../data/predictions/split_info.json", "w", encoding="utf-8") as f:
    json.dump(split_info, f, indent=2, ensure_ascii=False)
print(f"生成分割信息: ../data/predictions/split_info.json")

print("\n模拟数据生成完成！")
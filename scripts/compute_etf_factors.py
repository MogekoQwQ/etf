import os
import pandas as pd

# ------------------ 路径设置 ------------------
data_folder = "../data/etf_data"       # 原始CSV存放位置
factor_folder = "../data/factor_data"  # 新生成因子CSV存放位置
os.makedirs(factor_folder, exist_ok=True)

# ETF列表
etf_list = pd.read_csv("../data/etf_list.csv", dtype={"code": str})  # code,name

# 设置滚动窗口天数
window = 20

# 未来N天收益率和波动率设置
future_days_1 = 5
future_days_2 = 10

for idx, row in etf_list.iterrows():
    code = str(row['code'])
    name = row['name']
    csv_path = os.path.join(data_folder, f"{code}.csv")

    if not os.path.exists(csv_path):
        print(f"{code} 文件不存在，跳过")
        continue

    # ------------------ 读取ETF原始数据 ------------------
    df = pd.read_csv(csv_path, skiprows=1, parse_dates=["日期"])
    df = df.sort_values("日期").reset_index(drop=True)

    # ------------------ 计算量化因子 ------------------
    df["momentum_20"] = df["收盘"].pct_change(window)
    df["volatility_20"] = df["收盘"].pct_change().rolling(window).std()
    df["volume_mean_20"] = df["成交量"].rolling(window).mean()
    df["return_mean_20"] = df["涨跌幅"].rolling(window).mean()
    df["amplitude_mean_20"] = df["振幅"].rolling(window).mean()
    df["turnover_mean_20"] = df["换手率"].rolling(window).mean()
    df["MA_5"] = df["收盘"].rolling(5).mean()
    df["MA_10"] = df["收盘"].rolling(10).mean()

    # ------------------ 计算Y值 ------------------
    df["Y_next_day_return"] = df["收盘"].pct_change().shift(-1)
    df["Y_future_5d_return"] = df["收盘"].shift(-future_days_1) / df["收盘"] - 1
    df["Y_future_10d_return"] = df["收盘"].shift(-future_days_2) / df["收盘"] - 1
    df["Y_future_5d_vol_change"] = df["收盘"].pct_change().rolling(future_days_1).std().shift(-future_days_1) / \
                                   df["收盘"].pct_change().rolling(future_days_1).std() - 1
    df["Y_future_10d_vol_change"] = df["收盘"].pct_change().rolling(future_days_2).std().shift(-future_days_2) / \
                                    df["收盘"].pct_change().rolling(future_days_2).std() - 1

    # ------------------ 保存带因子和Y值的CSV ------------------
    output_path = os.path.join(factor_folder, f"{code}_factor.csv")
    df.to_csv(output_path, index=False)
    print(f"{code} 量化因子 + Y值计算完成，保存至 {output_path}")

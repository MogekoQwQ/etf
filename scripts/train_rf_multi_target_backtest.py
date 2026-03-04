# train_rf_multi_target_backtest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# -------------------- 设置 matplotlib 中文字体 --------------------
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------ 配置 ------------------
DATA_PATH = "../data/all_etf_factors.csv"
TARGETS = [
    "Y_next_day_return",
    "Y_future_5d_return",
    "Y_future_10d_return",
    "Y_future_5d_vol_change",
    "Y_future_10d_vol_change"
]
RETURN_TARGETS = ["Y_next_day_return", "Y_future_5d_return", "Y_future_10d_return"]
VOL_TARGETS    = ["Y_future_5d_vol_change", "Y_future_10d_vol_change"]

TOP_N = 10
OUTPUT_DIR = "../results/multi_target"
PREDICTIONS_DIR = "../data/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

RF_PARAMS = {
    "n_estimators": 800,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "random_state": 42,
}

# ------------------ 数据读取 ------------------
df = pd.read_csv(DATA_PATH, parse_dates=["日期"], encoding='utf-8-sig')
df = df.sort_values("日期").reset_index(drop=True)

exclude_cols = ["code", "name", "日期"] + TARGETS
feature_cols = [c for c in df.columns if c not in exclude_cols]

split_date = df["日期"].quantile(0.8)
train_df = df[df["日期"] <= split_date].copy()
test_df  = df[df["日期"] > split_date].copy()

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

result_file = os.path.join(OUTPUT_DIR, "model_results.txt")
with open(result_file, "w", encoding="utf-8") as f_out:

    for TARGET in TARGETS:
        print(f"\n正在处理目标: {TARGET}")
        f_out.write(f"\n目标: {TARGET}\n")

        y_train = train_df[TARGET]
        y_test  = test_df[TARGET]

        # 训练模型
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train, y_train)

        # 测试集预测
        test_df[f"y_pred_{TARGET}"] = rf.predict(X_test)

        # 模型评估
        r2 = r2_score(y_test, test_df[f"y_pred_{TARGET}"])
        mae = mean_absolute_error(y_test, test_df[f"y_pred_{TARGET}"])
        rmse = np.sqrt(mean_squared_error(y_test, test_df[f"y_pred_{TARGET}"]))
        print(f"R^2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        f_out.write(f"R^2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}\n")

        # ------------------ 可视化预测 vs 实际 ------------------
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, test_df[f"y_pred_{TARGET}"], alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("实际值")
        plt.ylabel("预测值")
        plt.title(f"预测 vs 实际 ({TARGET})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"pred_vs_actual_{TARGET}.png"))
        plt.close()

        # ------------------ 特征重要性 ------------------
        feat_imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        print("前10个重要特征：")
        print(feat_imp.head(10))
        f_out.write("前10个重要特征：\n")
        f_out.write(feat_imp.head(10).to_string(index=False) + "\n")

        plt.figure(figsize=(8,6))
        plt.barh(feat_imp.head(10)["feature"][::-1], feat_imp.head(10)["importance"][::-1])
        plt.xlabel("重要性")
        plt.title(f"前10个重要特征 ({TARGET})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"feature_importance_{TARGET}.png"))
        plt.close()

        # ------------------ 多空组合回测（仅收益类目标） ------------------
        if False and TARGET in RETURN_TARGETS:
            def compute_long_short_return(group):
                top = group.nlargest(min(TOP_N, len(group)), f"y_pred_{TARGET}")
                bottom = group.nsmallest(min(TOP_N, len(group)), f"y_pred_{TARGET}")
                long_ret = top[TARGET].mean()
                short_ret = bottom[TARGET].mean()
                return pd.Series({"daily_return": long_ret - short_ret})

            test_df_sorted = test_df.sort_values(["日期", f"y_pred_{TARGET}"], ascending=[True, False])
            daily_returns = test_df_sorted.groupby("日期", group_keys=False).apply(compute_long_short_return)
            daily_returns = daily_returns.reset_index()

            # 累积收益
            cumulative_returns = (1 + daily_returns["daily_return"]).cumprod()

            # ------------------ 计算回测指标 ------------------
            trading_days = 252  # 一年交易日
            mean_daily_return = daily_returns["daily_return"].mean()
            annual_return = (1 + mean_daily_return) ** trading_days - 1
            annual_vol = daily_returns["daily_return"].std() * np.sqrt(trading_days)
            sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else np.nan
            peak = cumulative_returns.cummax()
            max_drawdown = (cumulative_returns / peak - 1).min()

            f_out.write(f"年化收益率: {annual_return:.4f}, 年化波动率: {annual_vol:.4f}, "
                        f"夏普比率: {sharpe_ratio:.4f}, 最大回撤: {max_drawdown:.4f}\n")

            # 可视化累计收益
            plt.figure(figsize=(10,5))
            plt.plot(daily_returns["日期"], cumulative_returns, label="多空组合累计收益")
            plt.xlabel("日期")
            plt.ylabel("累计收益")
            plt.title(f"测试集多空组合回测 ({TARGET})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"cumulative_return_{TARGET}.png"))
            plt.close()
            print(f"回测完成 ({TARGET})")

print(f"\n所有结果和图表已保存到 {OUTPUT_DIR} 文件夹")

# ------------------ 保存测试集预测数据 ------------------
test_df.to_csv(os.path.join(PREDICTIONS_DIR, "test_set_with_predictions.csv"), index=False)
print(f"测试集预测数据已保存至 {PREDICTIONS_DIR}/test_set_with_predictions.csv")

# ------------------ 保存训练集和测试集分割信息 ------------------
split_info = {
    "split_date": split_date.strftime("%Y-%m-%d"),
    "train_start": train_df["日期"].min().strftime("%Y-%m-%d"),
    "train_end": train_df["日期"].max().strftime("%Y-%m-%d"),
    "test_start": test_df["日期"].min().strftime("%Y-%m-%d"),
    "test_end": test_df["日期"].max().strftime("%Y-%m-%d"),
    "train_samples": len(train_df),
    "test_samples": len(test_df)
}

import json
with open(os.path.join(PREDICTIONS_DIR, "split_info.json"), "w", encoding="utf-8") as f:
    json.dump(split_info, f, indent=2, ensure_ascii=False)
print(f"分割信息已保存至 {PREDICTIONS_DIR}/split_info.json")

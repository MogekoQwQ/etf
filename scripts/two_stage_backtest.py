"""
两阶段策略回测对比
比较仅传统模型排序 vs 传统模型+大语言模型排序
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings("ignore")

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------ 配置 ------------------
DATA_PATH = "../data/all_etf_factors.csv"
PREDICTIONS_PATH = "../data/predictions/test_set_with_predictions.csv"  # 预测数据
OUTPUT_DIR = "../results/two_stage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 策略参数
TARGET = "Y_future_5d_return"  # 预测目标
PRED_COL = f"y_pred_{TARGET}"  # 预测列名
TOP_N_FIRST = 50               # 第一阶段选取数量
TOP_N_FINAL = 10               # 最终组合数量
REBALANCING_FREQ = "W"         # 调仓频率：'D'每日，'W'每周，'M'每月
HOLDING_DAYS = 5               # 持有期（与目标对应）

# 大语言模型配置
USE_LLM = True                 # 是否使用大语言模型排序
LLM_LOG_DIR = os.path.join(OUTPUT_DIR, "llm_logs")

# ------------------ 数据加载 ------------------
def load_data() -> pd.DataFrame:
    """加载数据，优先使用带预测的数据，否则使用原始数据"""
    if os.path.exists(PREDICTIONS_PATH):
        print(f"读取预测数据: {PREDICTIONS_PATH}")
        df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["日期"])
        # 确保有预测列
        if PRED_COL not in df.columns:
            print(f"警告：预测数据中没有列 {PRED_COL}，将使用传统模型重新预测")
            df = run_traditional_prediction(df)
    else:
        print(f"预测数据不存在: {PREDICTIONS_PATH}，使用原始数据并运行传统模型预测")
        df = pd.read_csv(DATA_PATH, parse_dates=["日期"])
        df = run_traditional_prediction(df)

    # 确保 code 列为字符串类型，避免后续类型不匹配问题
    df["code"] = df["code"].astype(str)

    return df

def run_traditional_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """运行传统模型预测（简化版，使用全样本训练）"""
    print("运行传统模型预测...")
    from sklearn.ensemble import RandomForestRegressor

    # 特征列
    exclude_cols = ["code", "name", "日期", "Y_next_day_return", "Y_future_5d_return",
                    "Y_future_10d_return", "Y_future_5d_vol_change", "Y_future_10d_vol_change"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 划分训练集和测试集（按时间80/20）
    split_date = df["日期"].quantile(0.8)
    train_df = df[df["日期"] <= split_date]
    test_df = df[df["日期"] > split_date]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET]

    # 训练模型
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)

    # 预测
    df.loc[df["日期"] > split_date, PRED_COL] = rf.predict(X_test)
    df.loc[df["日期"] <= split_date, PRED_COL] = np.nan

    return df

# ------------------ 调仓日选择 ------------------
def get_rebalancing_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """根据频率选择调仓日期"""
    dates = df["日期"].dropna().unique()
    dates = pd.Series(dates).sort_values()

    if REBALANCING_FREQ == "W":
        # 每周一次（取每周最后一个交易日）
        rebalance_dates = dates[dates.isin(dates + pd.offsets.Week(weekday=4))]
    elif REBALANCING_FREQ == "M":
        # 每月一次（取每月最后一个交易日）
        rebalance_dates = dates.groupby(pd.Grouper(key="日期", freq="M")).last()
    else:
        # 每日
        rebalance_dates = dates

    return pd.DatetimeIndex(rebalance_dates.dropna().unique())

# ------------------ 策略执行 ------------------
def execute_strategy_a(df: pd.DataFrame, date: pd.Timestamp) -> List[str]:
    """策略A：仅传统模型，选择预测值最高的TOP_N_FINAL只ETF"""
    day_data = df[df["日期"] == date].copy()

    # 过滤掉预测值为NaN的行（训练集日期）
    day_data = day_data[day_data[PRED_COL].notna()]

    if len(day_data) < TOP_N_FINAL:
        return []

    # 按预测值排序
    day_data_sorted = day_data.sort_values(PRED_COL, ascending=False)
    selected_codes = day_data_sorted.head(TOP_N_FINAL)["code"].tolist()
    return selected_codes

def execute_strategy_b(df: pd.DataFrame, date: pd.Timestamp) -> List[str]:
    """策略B：两阶段排序"""
    day_data = df[df["日期"] == date].copy()

    # 过滤掉预测值为NaN的行（训练集日期）
    day_data = day_data[day_data[PRED_COL].notna()]

    if len(day_data) < TOP_N_FIRST:
        return []

    # 第一阶段：传统模型选TOP_N_FIRST
    day_data_sorted = day_data.sort_values(PRED_COL, ascending=False)
    top_etfs = day_data_sorted.head(TOP_N_FIRST).copy()

    if USE_LLM:
        # 第二阶段：大语言模型排序
        try:
            from llm_ranking import rank_etfs_by_llm
            ranked_etfs = rank_etfs_by_llm(top_etfs, str(date.date()), log_dir=LLM_LOG_DIR)
            if ranked_etfs is not None:
                selected_codes = ranked_etfs.head(TOP_N_FINAL)["code"].tolist()
            else:
                # LLM失败，回退到传统模型
                print(f"  LLM排序失败，回退到传统模型")
                selected_codes = top_etfs.head(TOP_N_FINAL)["code"].tolist()
        except Exception as e:
            print(f"  LLM排序异常: {e}，回退到传统模型")
            selected_codes = top_etfs.head(TOP_N_FINAL)["code"].tolist()
    else:
        # 不使用LLM，直接传统模型
        selected_codes = top_etfs.head(TOP_N_FINAL)["code"].tolist()

    return selected_codes

# ------------------ 回测计算 ------------------
def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """运行回测，返回每次调仓的结果"""
    rebalance_dates = get_rebalancing_dates(df)
    print(f"共 {len(rebalance_dates)} 个调仓日")

    # 调试：只处理前2个调仓日以快速验证修复（已禁用）
    DEBUG_LIMIT = None  # 设置为None以处理所有调仓日
    if DEBUG_LIMIT and len(rebalance_dates) > DEBUG_LIMIT:
        print(f"调试模式：只处理前 {DEBUG_LIMIT} 个调仓日")
        rebalance_dates = rebalance_dates[:DEBUG_LIMIT]

    results = []

    for i, date in enumerate(rebalance_dates):
        if i % 10 == 0:
            print(f"处理调仓日 {i+1}/{len(rebalance_dates)}: {date.date()}")

        # 执行两种策略
        codes_a = execute_strategy_a(df, date)
        codes_b = execute_strategy_b(df, date)

        if not codes_a or not codes_b:
            continue

        # 计算组合收益（使用实际未来收益）
        # 获取这些ETF在持有期内的实际收益
        day_data = df[df["日期"] == date].copy()

        # 策略A收益
        etf_returns_a = []
        for code in codes_a:
            etf_return = day_data[day_data["code"] == code][TARGET].values
            if len(etf_return) > 0:
                etf_returns_a.append(etf_return[0])

        # 策略B收益
        etf_returns_b = []
        for code in codes_b:
            etf_return = day_data[day_data["code"] == code][TARGET].values
            if len(etf_return) > 0:
                etf_returns_b.append(etf_return[0])

        if not etf_returns_a or not etf_returns_b:
            continue

        # 等权重组合收益
        portfolio_return_a = np.mean(etf_returns_a)
        portfolio_return_b = np.mean(etf_returns_b)

        # 保存结果
        results.append({
            "rebalance_date": date,
            "strategy_a_codes": codes_a,
            "strategy_b_codes": codes_b,
            "strategy_a_return": portfolio_return_a,
            "strategy_b_return": portfolio_return_b,
            "strategy_a_count": len(codes_a),
            "strategy_b_count": len(codes_b),
        })

    results_df = pd.DataFrame(results)
    return results_df

# ------------------ 绩效计算 ------------------
def calculate_performance(returns: pd.Series, freq: str = "W", holding_days: int = None) -> Dict:
    """计算绩效指标
    Args:
        returns: 持有期收益率序列
        freq: 调仓频率，'D'每日，'W'每周，'M'每月
        holding_days: 持有天数，如果为None则使用全局变量HOLDING_DAYS
    """
    if len(returns) == 0:
        return {}

    if holding_days is None:
        holding_days = HOLDING_DAYS

    # 年化因子：基于持有期计算一年中的周期数
    # 一年约252个交易日，除以持有天数得到周期数
    periods_per_year = 252 / holding_days

    # 如果调仓频率低于持有期（如月频调仓但持有5天），按实际调仓频率计算
    # 但收益已经是持有期收益，所以直接按持有期年化更合理
    # 保留原逻辑的注释以供参考：
    # if freq == "W":
    #     periods_per_year = 52
    # elif freq == "M":
    #     periods_per_year = 12
    # else:  # 日频
    #     periods_per_year = 252

    # 年化收益率
    mean_return = returns.mean()
    annual_return = (1 + mean_return) ** periods_per_year - 1

    # 年化波动率
    annual_vol = returns.std() * np.sqrt(periods_per_year)

    # 夏普比率（假设无风险利率为0）
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else np.nan

    # 最大回撤
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative / peak - 1)
    max_drawdown = drawdown.min()

    # 胜率
    win_rate = (returns > 0).mean()

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_periods": len(returns),
        "mean_return": mean_return,
        "std_return": returns.std(),
    }

# ------------------ 可视化 ------------------
def plot_results(results_df: pd.DataFrame, performance_a: Dict, performance_b: Dict):
    """绘制回测结果图表"""
    # 累计收益曲线
    cumulative_a = (1 + results_df["strategy_a_return"]).cumprod()
    cumulative_b = (1 + results_df["strategy_b_return"]).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(results_df["rebalance_date"], cumulative_a, label=f"策略A（仅传统模型）", linewidth=2)
    plt.plot(results_df["rebalance_date"], cumulative_b, label=f"策略B（两阶段）", linewidth=2)
    plt.xlabel("调仓日期")
    plt.ylabel("累计收益")
    plt.title("两阶段策略 vs 仅传统模型策略累计收益对比")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_returns.png"), dpi=300)
    plt.close()

    # 收益分布直方图
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(results_df["strategy_a_return"], bins=30, alpha=0.7, label="策略A")
    plt.xlabel("持有期收益")
    plt.ylabel("频次")
    plt.title("策略A收益分布")
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.hist(results_df["strategy_b_return"], bins=30, alpha=0.7, label="策略B", color='orange')
    plt.xlabel("持有期收益")
    plt.ylabel("频次")
    plt.title("策略B收益分布")
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "returns_distribution.png"), dpi=300)
    plt.close()

    # 绩效指标对比表格
    metrics_df = pd.DataFrame({
        "策略A（仅传统模型）": [
            f"{performance_a.get('annual_return', 0):.2%}",
            f"{performance_a.get('annual_volatility', 0):.2%}",
            f"{performance_a.get('sharpe_ratio', 0):.2f}",
            f"{performance_a.get('max_drawdown', 0):.2%}",
            f"{performance_a.get('win_rate', 0):.2%}",
            f"{performance_a.get('total_periods', 0)}",
        ],
        "策略B（两阶段）": [
            f"{performance_b.get('annual_return', 0):.2%}",
            f"{performance_b.get('annual_volatility', 0):.2%}",
            f"{performance_b.get('sharpe_ratio', 0):.2f}",
            f"{performance_b.get('max_drawdown', 0):.2%}",
            f"{performance_b.get('win_rate', 0):.2%}",
            f"{performance_b.get('total_periods', 0)}",
        ]
    }, index=["年化收益率", "年化波动率", "夏普比率", "最大回撤", "胜率", "调仓周期数"])

    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "performance_metrics.csv"))

    # 绘制绩效对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 年化收益率
    axes[0, 0].bar(["策略A", "策略B"],
                   [performance_a.get("annual_return", 0), performance_b.get("annual_return", 0)])
    axes[0, 0].set_title("年化收益率")
    axes[0, 0].set_ylabel("收益率")

    # 夏普比率
    axes[0, 1].bar(["策略A", "策略B"],
                   [performance_a.get("sharpe_ratio", 0), performance_b.get("sharpe_ratio", 0)])
    axes[0, 1].set_title("夏普比率")
    axes[0, 1].set_ylabel("比率")

    # 最大回撤
    axes[1, 0].bar(["策略A", "策略B"],
                   [performance_a.get("max_drawdown", 0), performance_b.get("max_drawdown", 0)])
    axes[1, 0].set_title("最大回撤")
    axes[1, 0].set_ylabel("回撤")

    # 胜率
    axes[1, 1].bar(["策略A", "策略B"],
                   [performance_a.get("win_rate", 0), performance_b.get("win_rate", 0)])
    axes[1, 1].set_title("胜率")
    axes[1, 1].set_ylabel("胜率")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "performance_comparison.png"), dpi=300)
    plt.close()

# ------------------ 主函数 ------------------
def main():
    print("=" * 60)
    print("两阶段ETF优选策略回测对比")
    print("=" * 60)

    # 加载数据
    df = load_data()
    print(f"数据加载完成，时间范围: {df['日期'].min().date()} 到 {df['日期'].max().date()}")
    print(f"ETF数量: {df['code'].nunique()}")
    print(f"样本总数: {len(df)}")

    # 运行回测
    results_df = run_backtest(df)
    if len(results_df) == 0:
        print("回测未产生有效结果，请检查参数和数据")
        return

    print(f"回测完成，有效调仓周期: {len(results_df)}")

    # 计算绩效指标
    performance_a = calculate_performance(results_df["strategy_a_return"], REBALANCING_FREQ, holding_days=HOLDING_DAYS)
    performance_b = calculate_performance(results_df["strategy_b_return"], REBALANCING_FREQ, holding_days=HOLDING_DAYS)

    # 保存结果
    results_df.to_csv(os.path.join(OUTPUT_DIR, "backtest_results.csv"), index=False)

    # 输出绩效报告
    print("\n" + "=" * 60)
    print("绩效指标对比")
    print("=" * 60)
    print(f"{'指标':<20} {'策略A（仅传统模型）':<25} {'策略B（两阶段）':<25}")
    print("-" * 70)
    print(f"{'年化收益率':<20} {performance_a.get('annual_return', 0):.2%} {performance_b.get('annual_return', 0):.2%}")
    print(f"{'年化波动率':<20} {performance_a.get('annual_volatility', 0):.2%} {performance_b.get('annual_volatility', 0):.2%}")
    print(f"{'夏普比率':<20} {performance_a.get('sharpe_ratio', 0):.2f} {performance_b.get('sharpe_ratio', 0):.2f}")
    print(f"{'最大回撤':<20} {performance_a.get('max_drawdown', 0):.2%} {performance_b.get('max_drawdown', 0):.2%}")
    print(f"{'胜率':<20} {performance_a.get('win_rate', 0):.2%} {performance_b.get('win_rate', 0):.2%}")
    print(f"{'调仓周期数':<20} {performance_a.get('total_periods', 0):<25} {performance_b.get('total_periods', 0):<25}")

    # 可视化
    plot_results(results_df, performance_a, performance_b)
    print(f"\n图表已保存至 {OUTPUT_DIR}")

    # 保存详细绩效报告
    report = {
        "parameters": {
            "target": TARGET,
            "top_n_first": TOP_N_FIRST,
            "top_n_final": TOP_N_FINAL,
            "rebalancing_freq": REBALANCING_FREQ,
            "holding_days": HOLDING_DAYS,
            "use_llm": USE_LLM,
        },
        "performance_strategy_a": performance_a,
        "performance_strategy_b": performance_b,
    }

    with open(os.path.join(OUTPUT_DIR, "performance_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"详细绩效报告已保存至 {OUTPUT_DIR}/performance_report.json")

if __name__ == "__main__":
    main()
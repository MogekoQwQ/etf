import pandas as pd
from pathlib import Path

# ====== 1. 文件路径 ======
input_path = Path("data/processed/all_etf_factors.csv")
output_csv = Path("raw_data_descriptive_stats.csv")
output_md = Path("raw_data_descriptive_stats.md")

# ====== 2. 读取数据 ======
df = pd.read_csv(input_path, encoding="utf-8-sig")

# ====== 3. 需要统计的字段映射 ======
# 左边是论文里想显示的名称，右边是数据表里实际列名候选
column_candidates = {
    "开盘价": ["开盘", "open", "Open"],
    "收盘价": ["收盘", "close", "Close"],
    "最高价": ["最高", "high", "High"],
    "最低价": ["最低", "low", "Low"],
    "成交量": ["成交量", "volume", "Volume"],
    "成交额": ["成交额", "amount", "Amount"],
    "振幅": ["振幅", "amplitude", "Amplitude"],
    "涨跌幅": ["涨跌幅", "pct_change", "PctChange", "pctchg"],
    "换手率": ["换手率", "turnover", "Turnover"],
}

def find_actual_column(df_columns, candidates):
    for c in candidates:
        if c in df_columns:
            return c
    return None

# ====== 4. 自动匹配列名 ======
selected_columns = {}
missing_columns = []

for display_name, candidates in column_candidates.items():
    actual_col = find_actual_column(df.columns, candidates)
    if actual_col is not None:
        selected_columns[display_name] = actual_col
    else:
        missing_columns.append(display_name)

if missing_columns:
    print("以下字段未在数据中找到，请检查列名后补充映射：")
    for col in missing_columns:
        print(f"- {col}")

if not selected_columns:
    raise ValueError("一个目标字段都没有匹配到，无法生成统计表。")

# ====== 5. 生成描述性统计 ======
rows = []
for display_name, actual_col in selected_columns.items():
    series = pd.to_numeric(df[actual_col], errors="coerce").dropna()
    if series.empty:
        continue

    rows.append({
        "项目": display_name,
        "数量": int(series.count()),
        "均值": series.mean(),
        "最小值": series.min(),
        "25%分位点": series.quantile(0.25),
        "75%分位点": series.quantile(0.75),
        "最大值": series.max(),
    })

stats_df = pd.DataFrame(rows)

# ====== 6. 数值格式化 ======
# 价格类和比例类保留4位小数，成交量/成交额如果太大也先统一保留4位小数
for col in ["均值", "最小值", "25%分位点", "75%分位点", "最大值"]:
    stats_df[col] = stats_df[col].map(lambda x: f"{x:.4f}")

stats_df["数量"] = stats_df["数量"].map(lambda x: f"{x:d}")

# ====== 7. 输出 CSV ======
stats_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

# ====== 8. 输出 Markdown 表格 ======
md_text = []
md_text.append("表4-X 原始数据描述性统计")
md_text.append("")
md_text.append(stats_df.to_markdown(index=False))
md_text.append("")
md_text.append("注：统计指标包括数量、均值、最小值、25%分位点、75%分位点及最大值。")

output_md.write_text("\n".join(md_text), encoding="utf-8-sig")

# ====== 9. 控制台输出 ======
print("\n原始数据描述性统计如下：\n")
print(stats_df.to_string(index=False))

print(f"\n已输出 CSV 文件：{output_csv}")
print(f"已输出 Markdown 文件：{output_md}")
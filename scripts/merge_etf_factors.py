import os
import pandas as pd

# ------------------ 路径设置 ------------------
factor_folder = "../data/factor_data"   # 单ETF因子CSV存放位置
output_file = "../data/all_etf_factors.csv"  # 合并后的大表
etf_list_file = "../data/etf_list.csv"  # ETF code-name对照表

# ------------------ 获取所有ETF因子文件 ------------------
factor_files = [f for f in os.listdir(factor_folder) if f.endswith("_factor.csv")]

# 读取ETF列表
etf_list = pd.read_csv(etf_list_file, dtype={"code": str})

all_df_list = []

for file in factor_files:
    file_path = os.path.join(factor_folder, file)
    code = file.split("_")[0]

    # 读取CSV
    df = pd.read_csv(file_path, parse_dates=["日期"])

    # 添加code和name列
    df.insert(0, "code", code)  # 插入到最前面
    name = etf_list.loc[etf_list["code"] == code, "name"].values[0]
    df.insert(1, "name", name)

    all_df_list.append(df)

# ------------------ 合并所有ETF ------------------
all_df = pd.concat(all_df_list, ignore_index=True)

# ------------------ 删除含NaN的行 ------------------
all_df = all_df.dropna().reset_index(drop=True)

# ------------------ 按日期排序 ------------------
all_df = all_df.sort_values(by=["日期", "code"]).reset_index(drop=True)

# ------------------ 保存大表 ------------------
all_df.to_csv(output_file, index=False)
print(f"所有ETF因子已合并，保存至 {output_file}")
print(f"大表形状: {all_df.shape}")

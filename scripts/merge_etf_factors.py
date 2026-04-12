import json
import os
from datetime import datetime

import pandas as pd

from market_data_utils import ensure_columns, read_csv_with_fallback, standardize_date_column_name
from traditional_model_config import (
    get_all_factors_file,
    get_dataset_manifest_file,
    get_etf_list_file,
    get_factor_data_dir,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FACTOR_FOLDER = get_factor_data_dir(PROJECT_ROOT)
OUTPUT_FILE = get_all_factors_file(PROJECT_ROOT)
ETF_LIST_FILE = get_etf_list_file(PROJECT_ROOT)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

FACTOR_FILES = [file_name for file_name in os.listdir(FACTOR_FOLDER) if file_name.endswith("_factor.csv")]
ETF_LIST = read_csv_with_fallback(ETF_LIST_FILE, dtype={"code": str})

all_frames = []

for file_name in FACTOR_FILES:
    file_path = os.path.join(FACTOR_FOLDER, file_name)
    code = file_name.split("_")[0]

    df = read_csv_with_fallback(file_path)
    df = standardize_date_column_name(df)
    ensure_columns(df, ["日期"], file_path)
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).reset_index(drop=True)

    df.insert(0, "code", code)
    matched_name = ETF_LIST.loc[ETF_LIST["code"] == code, "name"]
    name = matched_name.iloc[0] if not matched_name.empty else ""
    df.insert(1, "name", name)
    all_frames.append(df)

if not all_frames:
    raise FileNotFoundError(f"未在 {FACTOR_FOLDER} 找到可合并的因子文件。")

all_df = pd.concat(all_frames, ignore_index=True)
all_df = all_df.dropna().sort_values(by=["日期", "code"]).reset_index(drop=True)
all_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"ETF 因子总表已保存: {OUTPUT_FILE}")
print(f"合并后形状: {all_df.shape}")

dataset_manifest = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "factor_folder": FACTOR_FOLDER,
    "output_file": OUTPUT_FILE,
    "etf_list_file": ETF_LIST_FILE,
    "factor_file_count": len(FACTOR_FILES),
    "row_count": int(len(all_df)),
    "column_count": int(len(all_df.columns)),
}

manifest_path = get_dataset_manifest_file(PROJECT_ROOT)
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(dataset_manifest, handle, ensure_ascii=False, indent=2)

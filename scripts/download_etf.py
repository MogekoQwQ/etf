import os
import random
import time
import traceback

import akshare as ak
import akshare_proxy_patch
import pandas as pd

from market_data_utils import ensure_columns, read_csv_with_fallback, standardize_etf_market_columns
from traditional_model_config import (
    get_download_progress_file,
    get_etf_daily_dir,
    get_etf_list_file,
)


akshare_proxy_patch.install_patch("101.201.173.125", "", 50)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ETF_LIST_FILE = get_etf_list_file(PROJECT_ROOT)
DATA_DIR = get_etf_daily_dir(PROJECT_ROOT)
START_INDEX_FILE = get_download_progress_file(PROJECT_ROOT)

RECENT_DAYS = 1500
MAX_RETRY = 1
RETRY_DELAY_MIN = 3
RETRY_DELAY_MAX = 8
REQUEST_DELAY_MIN = 3.0
REQUEST_DELAY_MAX = 5.0
BATCH_SIZE = 3
BATCH_DELAY_MIN = 5.0
BATCH_DELAY_MAX = 6.0


def ensure_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(START_INDEX_FILE), exist_ok=True)


def load_start_index() -> int:
    if os.path.exists(START_INDEX_FILE):
        try:
            with open(START_INDEX_FILE, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
                if content:
                    return int(content)
        except (ValueError, IOError) as exc:
            print(f"[WARNING] Failed to load download progress: {exc}")
    return 0


def save_start_index(index: int) -> None:
    try:
        with open(START_INDEX_FILE, "w", encoding="utf-8") as handle:
            handle.write(str(index))
        print(f"[INFO] Saved download progress: {index}")
    except IOError as exc:
        print(f"[ERROR] Failed to save download progress: {exc}")


def load_etf_list(file_path: str) -> list[dict]:
    df = read_csv_with_fallback(file_path, dtype={"code": str})
    return df.to_dict(orient="records")


def download_single_etf(etf: dict) -> bool:
    code = str(etf["code"])
    name = str(etf["name"])

    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(f"Downloading {code} - {name} (attempt {attempt}/{MAX_RETRY})...")
            df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
            if df.empty:
                raise ValueError("Downloaded ETF history is empty")

            df = standardize_etf_market_columns(df)
            ensure_columns(df, ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "换手率"], code)
            df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
            df_recent = df.dropna(subset=["日期"]).sort_values("日期").tail(RECENT_DAYS).reset_index(drop=True)
            output_path = os.path.join(DATA_DIR, f"{code}.csv")
            with open(output_path, "w", encoding="utf-8-sig") as handle:
                handle.write(f"# code={code}, name={name}\n")
            df_recent.to_csv(output_path, mode="a", index=False, encoding="utf-8-sig")

            print(f"[OK] Downloaded {code} - {name}")
            return True
        except Exception as exc:
            print(f"[ERROR] Download failed for {code} - {name}: {exc}")
            print("Exception type:", type(exc))
            traceback.print_exc()

            if attempt < MAX_RETRY:
                retry_delay = random.uniform(RETRY_DELAY_MIN, RETRY_DELAY_MAX)
                print(f"Retrying after {retry_delay:.2f} seconds...")
                time.sleep(retry_delay)
            else:
                print("[ERROR] Retry limit reached, stopping this ETF download.")
                return False


def main() -> None:
    ensure_dir()
    etf_list = load_etf_list(ETF_LIST_FILE)

    start_index = load_start_index()
    print(f"[INFO] Resume from ETF index {start_index} / {len(etf_list)}")

    downloaded_count = 0
    last_successful_index = start_index - 1
    all_success = True

    for index, etf in enumerate(etf_list[start_index:], start=start_index):
        success = download_single_etf(etf)
        if not success:
            print(f"[ERROR] Download interrupted at ETF index {index}")
            save_start_index(index)
            all_success = False
            break

        downloaded_count += 1
        last_successful_index = index
        print(f"[INFO] Completed {downloaded_count} ETFs, overall progress {index + 1}/{len(etf_list)}")

        if downloaded_count > 0:
            delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
            print(f"[INFO] Sleeping {delay:.2f} seconds before next request...")
            time.sleep(delay)

        if downloaded_count % BATCH_SIZE == 0 and downloaded_count > 0:
            batch_delay = random.uniform(BATCH_DELAY_MIN, BATCH_DELAY_MAX)
            print(f"[INFO] Batch pause after {downloaded_count} ETFs: {batch_delay:.2f} seconds...")
            time.sleep(batch_delay)

    if all_success and last_successful_index == len(etf_list) - 1:
        if os.path.exists(START_INDEX_FILE):
            os.remove(START_INDEX_FILE)
            print("[INFO] Removed download progress file after successful completion.")
        print("ETF download completed.")
    elif not all_success:
        print(f"[INFO] Next run will resume from ETF index {last_successful_index + 1}")


if __name__ == "__main__":
    main()

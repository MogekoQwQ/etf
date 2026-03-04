import akshare_proxy_patch
akshare_proxy_patch.install_patch("101.201.173.125", "", 50)  # 使用官方提供的代理节点
import akshare as ak
import pandas as pd
import os
import time
import traceback  # 新增，用于打印完整 traceback
import random

ETF_LIST_FILE = "../data/etf_list.csv"  # 样本池 CSV
DATA_DIR = "../data/etf_data"           # 下载 CSV 保存目录
RECENT_DAYS = 750                        # 最近 3 年左右交易日
MAX_RETRY = 1                            # 单支 ETF 最大重试次数
RETRY_DELAY_MIN = 3                      # 重试最小间隔（秒）
RETRY_DELAY_MAX = 8                      # 重试最大间隔（秒）
REQUEST_DELAY_MIN = 3.0                  # 每次成功下载后的最小延迟（秒）
REQUEST_DELAY_MAX = 5.0                  # 每次成功下载后的最大延迟（秒）
BATCH_SIZE = 3                           # 每下载多少个ETF后额外等待
BATCH_DELAY_MIN = 5.0                    # 批次等待最小时间（秒）
BATCH_DELAY_MAX = 6.0                    # 批次等待最大时间（秒）
START_INDEX_FILE = "../data/download_progress.txt"  # 下载进度文件

def ensure_dir():
    """确保 data/etf_data 文件夹存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_start_index():
    """从进度文件读取上次的下载位置"""
    if os.path.exists(START_INDEX_FILE):
        try:
            with open(START_INDEX_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return int(content)
        except (ValueError, IOError) as e:
            print(f"[WARNING] 读取进度文件失败: {e}")
    return 0

def save_start_index(index):
    """保存当前下载位置到进度文件"""
    try:
        with open(START_INDEX_FILE, 'w', encoding='utf-8') as f:
            f.write(str(index))
        print(f"[INFO] 保存下载进度: {index}")
    except IOError as e:
        print(f"[ERROR] 保存进度文件失败: {e}")

def load_etf_list(file_path):
    """从 CSV 文件读取 ETF 样本池"""
    df = pd.read_csv(file_path)
    etf_list = df.to_dict(orient="records")
    return etf_list

def download_single_etf(etf):
    code = str(etf["code"])
    name = str(etf["name"])
    for attempt in range(1, MAX_RETRY + 1):
        try:
            print(f"开始下载 {code} - {name}，尝试 {attempt}/{MAX_RETRY} ...")

            # 下载 ETF 日线数据（前复权）
            df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="qfq")
            if df.empty:
                raise ValueError("返回数据为空")

            # 取最近 RECENT_DAYS 个交易日
            df_recent = df.tail(RECENT_DAYS)

            # 保存 CSV 并在第一行写代码 + 名称
            path = os.path.join(DATA_DIR, f"{code}.csv")
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(f"# code={code}, name={name}\n")
            df_recent.to_csv(path, mode="a", index=False, encoding="utf-8-sig")

            print(f"[OK] 下载完成 {code} - {name}")
            return True

        except Exception as e:
            print(f"[ERROR] 下载失败 {code} - {name}，原因: {e}")
            print("异常类型:", type(e))
            print("完整 traceback：")
            traceback.print_exc()  # 打印详细 traceback

            if attempt < MAX_RETRY:
                retry_delay = random.uniform(RETRY_DELAY_MIN, RETRY_DELAY_MAX)
                print(f"等待 {retry_delay:.2f} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"[ERROR] 达到最大重试次数，停止程序")
                return False

def main():
    ensure_dir()
    etf_list = load_etf_list(ETF_LIST_FILE)

    # 读取上次的下载进度
    start_index = load_start_index()
    print(f"[INFO] 从第 {start_index} 个 ETF 开始下载 (总共 {len(etf_list)} 个)")

    # 下载计数器，用于控制请求频率
    downloaded_count = 0
    last_successful_index = start_index - 1  # 记录最后一个成功下载的索引
    all_success = True

    for i, etf in enumerate(etf_list[start_index:], start=start_index):
        success = download_single_etf(etf)

        if not success:
            print(f"[ERROR] 下载失败，保存当前进度: {i}")
            save_start_index(i)  # 保存当前索引，下次从这里开始
            print("程序停止，未完成剩余 ETF 下载")
            all_success = False
            break

        downloaded_count += 1
        last_successful_index = i
        print(f"[INFO] 已成功下载 {downloaded_count} 个ETF，当前索引: {i+1}/{len(etf_list)}")

        # 每次成功下载后随机等待一小段时间，避免请求过于频繁
        if downloaded_count > 0:  # 第一个ETF下载后也等待
            delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
            print(f"[INFO] 请求间隔，等待 {delay:.2f} 秒...")
            time.sleep(delay)

        # 每下载BATCH_SIZE个ETF，额外随机等待一段时间
        if downloaded_count % BATCH_SIZE == 0 and downloaded_count > 0:
            batch_delay = random.uniform(BATCH_DELAY_MIN, BATCH_DELAY_MAX)
            print(f"[INFO] 已下载 {downloaded_count} 个ETF（批次大小: {BATCH_SIZE}），额外等待 {batch_delay:.2f} 秒...")
            time.sleep(batch_delay)

    # 判断是否全部下载完成
    if all_success and last_successful_index == len(etf_list) - 1:
        if os.path.exists(START_INDEX_FILE):
            os.remove(START_INDEX_FILE)
            print("[INFO] 所有ETF下载完成，已清除进度文件")
        print("下载任务结束")
    elif not all_success:
        print(f"[INFO] 下载中断，下次可从第 {last_successful_index + 1} 个ETF继续")

if __name__ == "__main__":
    main()

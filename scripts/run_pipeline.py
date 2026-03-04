"""
ETF两阶段优选研究框架 - 全流程执行脚本
执行顺序：数据下载 -> 因子计算 -> 传统模型训练 -> 两阶段回测
"""

import os
import sys
import argparse
import subprocess
import time
from typing import List

def run_script(script_name: str, args: List[str] = None, description: str = "") -> bool:
    """运行指定Python脚本"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        print(f"错误：脚本不存在 {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*60}")

    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"脚本执行失败，返回码: {result.returncode}")
            if result.stderr:
                print(f"错误输出:\n{result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"执行脚本时发生异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="ETF两阶段优选研究框架全流程执行")
    parser.add_argument("--skip-download", action="store_true", help="跳过数据下载步骤")
    parser.add_argument("--skip-factors", action="store_true", help="跳过因子计算步骤")
    parser.add_argument("--skip-traditional", action="store_true", help="跳过传统模型训练步骤")
    parser.add_argument("--skip-llm", action="store_true", help="跳过LLM排序步骤")
    parser.add_argument("--target", default="Y_future_5d_return",
                       choices=["Y_next_day_return", "Y_future_5d_return", "Y_future_10d_return"],
                       help="预测目标变量")
    parser.add_argument("--rebalancing-freq", default="W",
                       choices=["D", "W", "M"], help="调仓频率")
    parser.add_argument("--top-n-first", type=int, default=50, help="第一阶段选取数量")
    parser.add_argument("--top-n-final", type=int, default=10, help="最终组合数量")
    parser.add_argument("--use-llm", action="store_true", default=True, help="使用大语言模型排序")
    parser.add_argument("--mock-llm", action="store_true", help="模拟LLM排序（用于测试）")

    args = parser.parse_args()

    print("ETF两阶段优选研究框架 - 全流程执行")
    print("项目: 传统机器学习与大语言模型融合的 ETF 优选研究框架")
    print("=" * 60)

    # 检查必要的数据文件
    data_dir = "../data"
    etf_list_file = os.path.join(data_dir, "etf_list.csv")
    all_factors_file = os.path.join(data_dir, "all_etf_factors.csv")

    # 步骤1: 数据下载（如需要）
    if not args.skip_download:
        if not os.path.exists(etf_list_file):
            print(f"错误：ETF列表文件不存在 {etf_list_file}")
            print("请确保 data/etf_list.csv 存在，或移除 --skip-download 参数")
            return False

        success = run_script(
            "download_etf.py",
            description="下载ETF日线行情数据"
        )
        if not success:
            print("数据下载失败，终止流程")
            return False
    else:
        print("跳过数据下载步骤")

    # 步骤2: 因子计算（如需要）
    if not args.skip_factors:
        if not os.path.exists(etf_list_file):
            print(f"错误：ETF列表文件不存在 {etf_list_file}")
            return False

        # 检查是否已有个别ETF数据
        etf_data_dir = os.path.join(data_dir, "etf_data")
        if not os.path.exists(etf_data_dir) or len(os.listdir(etf_data_dir)) == 0:
            print(f"错误：ETF原始数据目录为空 {etf_data_dir}")
            print("请先运行数据下载步骤或确保数据存在")
            return False

        success = run_script(
            "compute_etf_factors.py",
            description="计算ETF量化因子"
        )
        if not success:
            print("因子计算失败，终止流程")
            return False

        success = run_script(
            "merge_etf_factors.py",
            description="合并所有ETF因子数据"
        )
        if not success:
            print("因子合并失败，终止流程")
            return False
    else:
        print("跳过因子计算步骤")

    # 检查合并后的因子数据
    if not os.path.exists(all_factors_file):
        print(f"错误：合并因子文件不存在 {all_factors_file}")
        print("请先运行因子计算步骤或确保文件存在")
        return False

    # 步骤3: 传统模型训练（如需要）
    if not args.skip_traditional:
        success = run_script(
            "train_rf_multi_target_backtest.py",
            description="训练随机森林模型并进行回测"
        )
        if not success:
            print("传统模型训练失败，终止流程")
            return False
    else:
        print("跳过传统模型训练步骤")

    # 检查预测数据
    predictions_file = "../data/predictions/test_set_with_predictions.csv"
    if not os.path.exists(predictions_file):
        print(f"警告：预测数据不存在 {predictions_file}")
        print("将使用原始数据并重新预测")

    # 步骤4: 两阶段回测
    print(f"\n{'='*60}")
    print("执行两阶段回测对比")
    print(f"目标变量: {args.target}")
    print(f"调仓频率: {args.rebalancing_freq}")
    print(f"第一阶段选取数量: {args.top_n_first}")
    print(f"最终组合数量: {args.top_n_final}")
    print(f"使用LLM排序: {args.use_llm}")
    print(f"模拟LLM: {args.mock_llm}")
    print(f"{'='*60}")

    # 构建参数
    llm_args = [
        "--target", args.target,
        "--rebalancing-freq", args.rebalancing_freq,
        "--top-n-first", str(args.top_n_first),
        "--top-n-final", str(args.top_n_final),
    ]

    if not args.use_llm:
        llm_args.append("--no-llm")
    if args.mock_llm:
        llm_args.append("--mock-llm")

    success = run_script(
        "two_stage_backtest.py",
        args=llm_args,
        description="执行两阶段策略回测对比"
    )

    if success:
        print(f"\n{'='*60}")
        print("全流程执行完成！")
        print("=" * 60)
        print("\n生成的结果文件：")
        print(f"1. 传统模型结果: ../results/multi_target/")
        print(f"2. 两阶段回测结果: ../results/two_stage/")
        print(f"3. LLM排序日志: ../results/two_stage/llm_logs/")
        print("\n请查看 ../results/two_stage/ 目录中的图表和报告。")
    else:
        print("两阶段回测失败")

if __name__ == "__main__":
    main()
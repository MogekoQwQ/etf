"""Run the end-to-end ETF research pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional

from traditional_model_config import (
    DEFAULT_TRADITIONAL_MODEL,
    PLANNED_TRADITIONAL_MODELS,
    ensure_implemented_traditional_model,
    get_traditional_prediction_file,
    get_traditional_two_stage_dir,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def run_script(script_name: str, args: Optional[List[str]] = None, description: str = "") -> bool:
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"错误: 脚本不存在 {script_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"执行步骤: {description}")
    print(f"脚本: {script_name}")
    print(f"{'=' * 60}")

    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
    except Exception as exc:
        print(f"脚本执行异常: {exc}")
        return False

    if result.returncode != 0:
        print(f"脚本执行失败，返回码: {result.returncode}")
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETF 两阶段优选研究项目全流程执行脚本")
    parser.add_argument("--skip-download", action="store_true", help="跳过数据下载步骤")
    parser.add_argument("--skip-factors", action="store_true", help="跳过因子计算步骤")
    parser.add_argument("--skip-traditional", action="store_true", help="跳过第一阶段传统模型训练步骤")
    parser.add_argument("--skip-llm", action="store_true", help="跳过第二阶段 LLM 排序步骤")
    parser.add_argument(
        "--traditional-model",
        default=DEFAULT_TRADITIONAL_MODEL,
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="第一阶段传统模型名称。当前仅 random_forest 已实现，其余选项为后续扩展预留。",
    )
    parser.add_argument(
        "--target",
        default="Y_future_5d_return",
        choices=["Y_next_day_return", "Y_future_5d_return", "Y_future_10d_return"],
        help="预测目标变量",
    )
    parser.add_argument("--rebalancing-freq", default="W", choices=["D", "W", "M"], help="调仓频率")
    parser.add_argument("--top-n-first", type=int, default=50, help="第一阶段候选数量")
    parser.add_argument("--top-n-final", type=int, default=10, help="最终组合数量")
    parser.add_argument("--use-llm", action="store_true", default=True, help="启用 LLM 二次排序")
    parser.add_argument("--mock-llm", action="store_true", help="使用本地 mock LLM 排序")
    parser.add_argument("--enable-explanations", action="store_true", help="启用展示型解释功能")
    parser.add_argument("--explanation-date", type=str, default=None, help="仅对指定调仓日生成解释")
    parser.add_argument(
        "--explanation-sample-size",
        type=int,
        default=3,
        help="启用解释但未指定 explanation-date 时的稳定抽样日期数量",
    )
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM API 超时时间，单位秒")
    return parser.parse_args()


def main() -> bool:
    args = parse_args()
    traditional_model = ensure_implemented_traditional_model(args.traditional_model)

    if args.llm_timeout is not None and args.llm_timeout <= 0:
        raise ValueError("--llm-timeout must be a positive integer")
    if args.explanation_sample_size <= 0:
        raise ValueError("--explanation-sample-size must be a positive integer")

    use_llm = args.use_llm and not args.skip_llm

    print("ETF 两阶段优选研究框架 - 全流程执行")
    print("=" * 60)
    print(f"第一阶段传统模型: {traditional_model}")
    print(f"第二阶段 LLM 排序: {use_llm}")

    data_dir = os.path.join(PROJECT_ROOT, "data")
    etf_list_file = os.path.join(data_dir, "etf_list.csv")
    all_factors_file = os.path.join(data_dir, "all_etf_factors.csv")

    if not args.skip_download:
        if not os.path.exists(etf_list_file):
            print(f"错误: ETF 列表文件不存在 {etf_list_file}")
            return False
        if not run_script("download_etf.py", description="下载 ETF 日线行情数据"):
            print("数据下载失败，终止流程")
            return False
    else:
        print("跳过数据下载步骤")

    if not args.skip_factors:
        etf_data_dir = os.path.join(data_dir, "etf_data")
        if not os.path.exists(etf_list_file):
            print(f"错误: ETF 列表文件不存在 {etf_list_file}")
            return False
        if not os.path.exists(etf_data_dir) or len(os.listdir(etf_data_dir)) == 0:
            print(f"错误: ETF 原始数据目录为空 {etf_data_dir}")
            return False
        if not run_script("compute_etf_factors.py", description="计算 ETF 量化因子"):
            print("因子计算失败，终止流程")
            return False
        if not run_script("merge_etf_factors.py", description="合并 ETF 因子数据"):
            print("因子合并失败，终止流程")
            return False
    else:
        print("跳过因子计算步骤")

    if not os.path.exists(all_factors_file):
        print(f"错误: 合并后的因子文件不存在 {all_factors_file}")
        return False

    if not args.skip_traditional:
        if not run_script(
            "train_traditional_multi_target_backtest.py",
            args=["--traditional-model", traditional_model],
            description="训练第一阶段传统模型并生成评估与预测文件",
        ):
            print("第一阶段传统模型训练失败，终止流程")
            return False
    else:
        print("跳过第一阶段传统模型训练步骤")

    predictions_file = get_traditional_prediction_file(PROJECT_ROOT, traditional_model)
    if not os.path.exists(predictions_file):
        print(f"警告: 预测文件不存在 {predictions_file}")
        print("两阶段回测将尝试使用原始因子数据并在脚本内部回退生成预测。")

    two_stage_args = [
        "--traditional-model",
        traditional_model,
        "--target",
        args.target,
        "--rebalancing-freq",
        args.rebalancing_freq,
        "--top-n-first",
        str(args.top_n_first),
        "--top-n-final",
        str(args.top_n_final),
    ]
    if not use_llm:
        two_stage_args.append("--no-llm")
    if args.mock_llm:
        two_stage_args.append("--mock-llm")
    if args.enable_explanations:
        two_stage_args.append("--enable-explanations")
    if args.explanation_date:
        two_stage_args.extend(["--explanation-date", args.explanation_date])
    if args.enable_explanations and args.explanation_date is None:
        two_stage_args.extend(["--explanation-sample-size", str(args.explanation_sample_size)])
    if args.llm_timeout is not None:
        two_stage_args.extend(["--llm-timeout", str(args.llm_timeout)])

    print(f"\n{'=' * 60}")
    print("执行两阶段策略回测")
    print(f"第一阶段传统模型: {traditional_model}")
    print(f"预测目标: {args.target}")
    print(f"调仓频率: {args.rebalancing_freq}")
    print(f"第一阶段候选数量: {args.top_n_first}")
    print(f"最终组合数量: {args.top_n_final}")
    print(f"使用 LLM 排序: {use_llm}")
    print(f"Mock LLM: {args.mock_llm}")
    print(f"启用解释: {args.enable_explanations}")
    print(f"{'=' * 60}")

    if not run_script("two_stage_backtest.py", args=two_stage_args, description="执行两阶段策略回测对比"):
        print("两阶段策略回测失败")
        return False

    two_stage_dir = get_traditional_two_stage_dir(PROJECT_ROOT, traditional_model)
    if not run_script(
        "two_stage_reporter.py",
        args=[
            "--input-dir",
            two_stage_dir,
            "--output",
            os.path.join(two_stage_dir, "report_summary.html"),
        ],
        description="生成两阶段回测 HTML 汇总报告",
    ):
        print("两阶段汇总报告生成失败")
        return False

    print(f"\n{'=' * 60}")
    print("全流程执行完成")
    print("=" * 60)
    print("\n结果目录:")
    print(f"1. 第一阶段训练评估: ../results/traditional_models/{traditional_model}/training_eval/")
    print(f"2. 第一阶段预测输出: ../data/predictions/{traditional_model}/")
    print(f"3. 两阶段回测结果: ../results/traditional_models/{traditional_model}/two_stage/")
    print(f"4. LLM 排序日志: ../results/traditional_models/{traditional_model}/two_stage/llm_logs/")
    print(
        "5. 若命中解释日期，解释结果位于 "
        f"../results/traditional_models/{traditional_model}/two_stage/llm_logs/explanations/"
    )
    return True


if __name__ == "__main__":
    main()

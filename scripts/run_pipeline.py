"""Run the minimal ETF two-stage experiment pipeline from the delivery root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from traditional_model_config import (
    DEFAULT_TRADITIONAL_MODEL,
    PLANNED_TRADITIONAL_MODELS,
    ensure_implemented_traditional_model,
    get_all_factors_file,
    get_etf_daily_dir,
    get_etf_list_file,
    get_llm_config_local_file,
    get_traditional_llm_log_dir,
    get_traditional_prediction_file,
    get_traditional_training_eval_dir,
    get_traditional_two_stage_output_dir,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def display_path(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return path_obj.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path_obj)


def run_script(script_name: str, args: Sequence[str] | None = None, description: str = "") -> bool:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        print(f"脚本不存在: {display_path(script_path)}")
        return False

    print(f"\n{'=' * 60}")
    print(description or script_name)
    print(f"脚本: {script_name}")
    print(f"{'=' * 60}")

    cmd = [sys.executable, str(script_path), *(args or [])]
    try:
        result = subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"脚本执行失败: {exc}")
        return False

    if result.returncode != 0:
        print(f"脚本退出码非 0: {result.returncode}")
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ETF 两阶段实验主流程。")
    parser.add_argument("--skip-download", action="store_true", help="跳过 ETF 原始数据下载。")
    parser.add_argument("--skip-factors", action="store_true", help="跳过因子计算和合并。")
    parser.add_argument("--skip-traditional", action="store_true", help="跳过第一阶段传统模型训练。")
    parser.add_argument("--skip-llm", action="store_true", help="跳过第二阶段 LLM 重排序。")
    parser.add_argument(
        "--traditional-model",
        default=DEFAULT_TRADITIONAL_MODEL,
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="第一阶段传统模型。",
    )
    parser.add_argument(
        "--target",
        default="Y_future_5d_return",
        choices=["Y_next_day_return", "Y_future_5d_return", "Y_future_10d_return"],
        help="训练目标列。",
    )
    parser.add_argument("--rebalancing-freq", default="W", choices=["D", "W", "M"], help="调仓频率。")
    parser.add_argument("--top-n-first", type=int, default=50, help="第一阶段候选池数量。")
    parser.add_argument("--top-n-final", type=int, default=10, help="最终持仓数量。")
    parser.add_argument("--use-llm", action="store_true", default=True, help="启用第二阶段 LLM 重排序。")
    parser.add_argument("--mock-llm", action="store_true", help="使用 Mock LLM 排序。")
    parser.add_argument("--enable-explanations", action="store_true", help="生成解释输出。")
    parser.add_argument("--explanation-date", type=str, default=None, help="指定生成解释的调仓日。")
    parser.add_argument(
        "--explanation-sample-size",
        type=int,
        default=3,
        help="未指定 explanation-date 时随机抽取的解释样本数。",
    )
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM API 超时时间（秒）。")
    parser.add_argument(
        "--second-stage-llm",
        type=str,
        default="deepseek-chat",
        choices=["deepseek-chat", "gemini-2.5-flash-lite"],
        help="第二阶段 LLM 模型。",
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=str(get_llm_config_local_file(PROJECT_ROOT)),
        help="本地 LLM 配置文件路径。",
    )
    return parser.parse_args()


def main() -> bool:
    args = parse_args()
    traditional_model = ensure_implemented_traditional_model(args.traditional_model)

    if args.llm_timeout is not None and args.llm_timeout <= 0:
        raise ValueError("--llm-timeout 必须为正整数")
    if args.explanation_sample_size <= 0:
        raise ValueError("--explanation-sample-size 必须为正整数")

    use_llm = args.use_llm and not args.skip_llm

    print("ETF 两阶段实验流水线")
    print("=" * 60)
    print(f"第一阶段模型: {traditional_model}")
    print(f"启用第二阶段 LLM: {use_llm}")

    etf_list_file = get_etf_list_file(PROJECT_ROOT)
    etf_daily_dir = get_etf_daily_dir(PROJECT_ROOT)
    all_factors_file = get_all_factors_file(PROJECT_ROOT)

    if not args.skip_download:
        if not etf_list_file.exists():
            print(f"ETF 列表文件不存在: {display_path(etf_list_file)}")
            return False
        if not run_script("download_etf.py", description="下载 ETF 原始数据"):
            print("原始数据下载失败，流程终止。")
            return False
    else:
        print("已跳过原始数据下载。")

    if not args.skip_factors:
        if not etf_list_file.exists():
            print(f"ETF 列表文件不存在: {display_path(etf_list_file)}")
            return False
        if not etf_daily_dir.exists() or not any(etf_daily_dir.iterdir()):
            print(f"ETF 日线数据目录为空: {display_path(etf_daily_dir)}")
            return False
        if not run_script("compute_etf_factors.py", description="计算 ETF 因子"):
            print("因子计算失败，流程终止。")
            return False
        if not run_script("merge_etf_factors.py", description="合并 ETF 因子"):
            print("因子合并失败，流程终止。")
            return False
    else:
        print("已跳过因子计算和合并。")

    if not all_factors_file.exists():
        print(f"合并后的因子文件不存在: {display_path(all_factors_file)}")
        return False

    if not args.skip_traditional:
        if not run_script(
            "train_traditional_multi_target_backtest.py",
            args=["--traditional-model", traditional_model, "--target", args.target],
            description="训练第一阶段传统模型并生成预测结果",
        ):
            print("第一阶段模型训练失败，流程终止。")
            return False
    else:
        print("已跳过第一阶段模型训练。")

    predictions_file = get_traditional_prediction_file(PROJECT_ROOT, traditional_model)
    if not predictions_file.exists():
        print(f"预测文件不存在，将由回测脚本尝试回退生成: {display_path(predictions_file)}")

    two_stage_args = [
        "--traditional-model",
        traditional_model,
        "--target",
        args.target,
        "--second-stage-llm",
        args.second_stage_llm,
        "--llm-config",
        args.llm_config,
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
    print("运行 ETF 两阶段回测")
    print(f"第一阶段模型: {traditional_model}")
    print(f"目标列: {args.target}")
    print(f"调仓频率: {args.rebalancing_freq}")
    print(f"候选池数量: {args.top_n_first}")
    print(f"最终持仓数量: {args.top_n_final}")
    print(f"启用 LLM: {use_llm}")
    print(f"启用 Mock LLM: {args.mock_llm}")
    print(f"启用解释输出: {args.enable_explanations}")
    print(f"{'=' * 60}")

    if not run_script("two_stage_backtest.py", args=two_stage_args, description="执行两阶段回测"):
        print("两阶段回测失败。")
        return False

    two_stage_dir = get_traditional_two_stage_output_dir(
        PROJECT_ROOT,
        traditional_model,
        args.second_stage_llm,
        args.top_n_first,
        args.top_n_final,
    )
    explanation_dir = get_traditional_llm_log_dir(PROJECT_ROOT, traditional_model, args.second_stage_llm) / "explanations"

    print(f"\n{'=' * 60}")
    print("输出目录")
    print("=" * 60)
    print(f"1. 第一阶段评估: {display_path(get_traditional_training_eval_dir(PROJECT_ROOT, traditional_model))}")
    print(f"2. 第一阶段预测: {display_path(predictions_file.parent)}")
    print(f"3. 两阶段结果: {display_path(two_stage_dir)}")
    print(f"4. LLM 日志: {display_path(get_traditional_llm_log_dir(PROJECT_ROOT, traditional_model, args.second_stage_llm))}")
    print(f"5. 解释输出: {display_path(explanation_dir)}")
    return True


if __name__ == "__main__":
    main()

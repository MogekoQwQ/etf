"""Run and archive experiment suites across traditional models."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from experiment_summary import generate_experiment_summary
from traditional_model_config import (
    DEFAULT_TRADITIONAL_MODEL,
    DEFAULT_TOP_N_FINAL,
    DEFAULT_TOP_N_FIRST,
    EXPERIMENT_RESULT_VARIANTS,
    PLANNED_TRADITIONAL_MODELS,
    get_experiment_summary_dir,
    get_traditional_experiment_run_dir,
    get_traditional_model_label,
    get_traditional_two_stage_output_dir,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_VARIANTS = ("baseline", "llm")
ExecutionPlan = List[Tuple[str, str]]


def to_project_relative_path(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path_obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量运行并归档 ETF 论文实验。")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(PLANNED_TRADITIONAL_MODELS),
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="参与实验的传统模型列表。",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        choices=list(EXPERIMENT_RESULT_VARIANTS),
        help="运行结果类型列表。",
    )
    parser.add_argument(
        "--baseline-models",
        nargs="+",
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="仅对这些模型执行 baseline 版本。",
    )
    parser.add_argument(
        "--llm-models",
        nargs="+",
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="仅对这些模型执行 LLM 两阶段版本。",
    )
    parser.add_argument(
        "--explanation-models",
        nargs="+",
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="仅对这些模型生成 explanations。",
    )
    parser.add_argument("--summary-only", action="store_true", help="只生成实验汇总，不运行实验。")
    parser.add_argument("--skip-summary", action="store_true", help="跳过实验汇总。")
    parser.add_argument("--skip-download", action="store_true", help="转发给 run_pipeline.py。")
    parser.add_argument("--skip-factors", action="store_true", help="转发给 run_pipeline.py。")
    parser.add_argument("--skip-traditional", action="store_true", help="转发给 run_pipeline.py。")
    parser.add_argument(
        "--target",
        default="Y_future_5d_return",
        choices=["Y_next_day_return", "Y_future_5d_return", "Y_future_10d_return"],
        help="训练目标列。",
    )
    parser.add_argument("--rebalancing-freq", default="W", choices=["D", "W", "M"], help="调仓频率。")
    parser.add_argument("--top-n-first", type=int, default=50, help="第一阶段候选池数量。")
    parser.add_argument("--top-n-final", type=int, default=10, help="最终持仓数量。")
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM API 超时时间（秒）。")
    parser.add_argument(
        "--enable-explanations",
        action="store_true",
        help="允许 explanations 输出。",
    )
    parser.add_argument(
        "--explanation-model",
        default=DEFAULT_TRADITIONAL_MODEL,
        choices=list(PLANNED_TRADITIONAL_MODELS),
        help="默认生成 explanations 的模型。",
    )
    parser.add_argument("--explanation-date", type=str, default=None, help="指定 explanations 调仓日。")
    parser.add_argument(
        "--explanation-sample-size",
        type=int,
        default=3,
        help="未指定 explanation-date 时的 explanation 样本数。",
    )
    parser.add_argument(
        "--summary-output-dir",
        default=str(get_experiment_summary_dir(PROJECT_ROOT)),
        help="实验汇总输出目录。",
    )
    return parser.parse_args()


def run_pipeline(pipeline_args: List[str], description: str) -> None:
    script_path = SCRIPT_DIR / "run_pipeline.py"
    cmd = [sys.executable, str(script_path), *pipeline_args]

    print("\n" + "=" * 72)
    print(description)
    print("Command:", " ".join(cmd))
    print("=" * 72)

    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Pipeline run failed with exit code {completed.returncode}")


def ordered_unique(models: Sequence[str]) -> List[str]:
    selected = {model for model in models}
    return [model for model in PLANNED_TRADITIONAL_MODELS if model in selected]


def use_explicit_variant_model_args(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.baseline_models,
            args.llm_models,
            args.explanation_models,
        )
    )


def resolve_execution_config(args: argparse.Namespace) -> Dict[str, object]:
    explicit_groups = use_explicit_variant_model_args(args)

    if explicit_groups:
        baseline_models = ordered_unique(args.baseline_models or [])
        llm_models = ordered_unique(args.llm_models or [])
        mock_llm_models: List[str] = []
        if not baseline_models and not llm_models:
            raise ValueError("显式模式下至少需要传入 --baseline-models 或 --llm-models。")
    else:
        selected_models = ordered_unique(args.models)
        baseline_models = selected_models if "baseline" in args.variants else []
        llm_models = selected_models if "llm" in args.variants else []
        mock_llm_models = selected_models if "mock_llm" in args.variants else []

    if args.explanation_models is not None:
        explanation_models = ordered_unique(args.explanation_models)
    elif args.enable_explanations:
        explanation_models = [args.explanation_model]
    else:
        explanation_models = []

    invalid_explanation_models = [model for model in explanation_models if model not in llm_models]
    if invalid_explanation_models:
        raise ValueError(
            "--explanation-models 中的模型必须同时包含在 --llm-models 中: "
            + ", ".join(invalid_explanation_models)
        )

    execution_plan: ExecutionPlan = []
    execution_plan.extend((model, "baseline") for model in baseline_models)
    execution_plan.extend((model, "llm") for model in llm_models)
    execution_plan.extend((model, "mock_llm") for model in mock_llm_models)

    summary_models = ordered_unique([*baseline_models, *llm_models, *mock_llm_models])
    return {
        "baseline_models": baseline_models,
        "llm_models": llm_models,
        "mock_llm_models": mock_llm_models,
        "explanation_models": explanation_models,
        "execution_plan": execution_plan,
        "summary_models": summary_models,
        "explicit_groups": explicit_groups,
    }


def copy_two_stage_snapshot(
    model_name: str,
    variant: str,
    pipeline_args: List[str],
    llm_model: str = "deepseek-chat",
    top_n_first: int = DEFAULT_TOP_N_FIRST,
    top_n_final: int = DEFAULT_TOP_N_FINAL,
) -> Path:
    source_dir = get_traditional_two_stage_output_dir(
        PROJECT_ROOT,
        model_name,
        llm_model,
        top_n_first,
        top_n_final,
    )
    destination_dir = get_traditional_experiment_run_dir(
        PROJECT_ROOT,
        model_name,
        variant,
        llm_model=llm_model,
        top_n_first=top_n_first,
        top_n_final=top_n_final,
    )

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Active two-stage result directory not found: {source_dir}")

    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)

    manifest = {
        "traditional_model": model_name,
        "model_label": get_traditional_model_label(model_name),
        "variant": variant,
        "archived_at": dt.datetime.now().isoformat(timespec="seconds"),
        "pipeline_args": pipeline_args,
        "source_two_stage_dir": to_project_relative_path(source_dir),
        "archived_dir": to_project_relative_path(destination_dir),
    }
    manifest_path = destination_dir / "experiment_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return destination_dir


def should_enable_explanations(
    explanation_models: Sequence[str],
    model_name: str,
    variant: str,
) -> bool:
    return variant == "llm" and model_name in explanation_models


def build_pipeline_args(
    args: argparse.Namespace,
    explanation_models: Sequence[str],
    model_name: str,
    variant: str,
    skip_download: bool,
    skip_factors: bool,
    skip_traditional: bool,
) -> List[str]:
    pipeline_args = [
        "--traditional-model",
        model_name,
        "--target",
        args.target,
        "--rebalancing-freq",
        args.rebalancing_freq,
        "--top-n-first",
        str(args.top_n_first),
        "--top-n-final",
        str(args.top_n_final),
    ]
    if skip_download:
        pipeline_args.append("--skip-download")
    if skip_factors:
        pipeline_args.append("--skip-factors")
    if skip_traditional:
        pipeline_args.append("--skip-traditional")
    if variant == "baseline":
        pipeline_args.append("--skip-llm")
    elif variant == "mock_llm":
        pipeline_args.append("--mock-llm")

    if should_enable_explanations(explanation_models, model_name, variant):
        pipeline_args.append("--enable-explanations")
        if args.explanation_date:
            pipeline_args.extend(["--explanation-date", args.explanation_date])
        else:
            pipeline_args.extend(["--explanation-sample-size", str(args.explanation_sample_size)])

    if args.llm_timeout is not None:
        pipeline_args.extend(["--llm-timeout", str(args.llm_timeout)])
    return pipeline_args


def run_suite(args: argparse.Namespace, execution_plan: ExecutionPlan, explanation_models: Sequence[str]) -> List[Path]:
    archived_dirs: List[Path] = []
    shared_data_ready = False
    model_ready: Dict[str, bool] = {}

    for model_name, variant in execution_plan:
        pipeline_args = build_pipeline_args(
            args,
            explanation_models,
            model_name,
            variant,
            skip_download=args.skip_download or shared_data_ready,
            skip_factors=args.skip_factors or shared_data_ready,
            skip_traditional=args.skip_traditional or model_ready.get(model_name, False),
        )
        description = f"运行 {model_name} [{variant}]"
        run_pipeline(pipeline_args, description)
        archived_dir = copy_two_stage_snapshot(
            model_name,
            variant,
            pipeline_args,
            top_n_first=args.top_n_first,
            top_n_final=args.top_n_final,
        )
        archived_dirs.append(archived_dir)
        print(f"已归档到: {to_project_relative_path(archived_dir)}")

        shared_data_ready = True
        model_ready[model_name] = True

    return archived_dirs


def generate_summary(args: argparse.Namespace, summary_models: Sequence[str]) -> None:
    output_paths = generate_experiment_summary(summary_models, args.summary_output_dir)
    print("\n实验汇总已生成:")
    for label, path in output_paths.items():
        print(f"- {label}: {to_project_relative_path(path)}")


def print_execution_summary(config: Dict[str, object]) -> None:
    def format_models(models: Sequence[str]) -> str:
        return ", ".join(models) if models else "无"

    print("\n执行计划")
    print(f"- baseline: {format_models(config['baseline_models'])}")
    print(f"- llm: {format_models(config['llm_models'])}")
    if config["mock_llm_models"]:
        print(f"- mock_llm: {format_models(config['mock_llm_models'])}")
    print(f"- explanations: {format_models(config['explanation_models'])}")
    print(f"- 汇总模型: {format_models(config['summary_models'])}")


def main() -> None:
    args = parse_args()
    if args.llm_timeout is not None and args.llm_timeout <= 0:
        raise ValueError("--llm-timeout 必须为正整数")
    if args.explanation_sample_size <= 0:
        raise ValueError("--explanation-sample-size 必须为正整数")

    config = resolve_execution_config(args)
    print_execution_summary(config)

    if not args.summary_only:
        archived_dirs = run_suite(
            args,
            config["execution_plan"],
            config["explanation_models"],
        )
        print("\n已归档目录")
        for directory in archived_dirs:
            print(f"- {to_project_relative_path(directory)}")

    if not args.skip_summary:
        generate_summary(args, config["summary_models"])


if __name__ == "__main__":
    main()

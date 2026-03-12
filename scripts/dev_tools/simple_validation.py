"""Lightweight validation script for key research modules."""

from __future__ import annotations

import ast
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPTS_DIR)


def script_path(filename: str) -> str:
    return os.path.join(SCRIPTS_DIR, filename)


def check_syntax() -> bool:
    print("=" * 60)
    print("语法检查")
    print("=" * 60)

    files_to_check = [
        "traditional_model_config.py",
        "train_traditional_multi_target_backtest.py",
        "llm_ranking.py",
        "two_stage_backtest.py",
        "run_pipeline.py",
        "explanation_utils.py",
        "explanation_reporter.py",
        "explanation_config.py",
    ]

    all_good = True
    for filename in files_to_check:
        filepath = script_path(filename)
        if not os.path.exists(filepath):
            print(f"  [FAIL] {filename}: 文件不存在")
            all_good = False
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as handle:
                ast.parse(handle.read())
            print(f"  [OK] {filename}: 语法正确")
        except SyntaxError as exc:
            print(f"  [FAIL] {filename}: {exc}")
            all_good = False
    return all_good


def check_pipeline_references() -> bool:
    print("\n" + "=" * 60)
    print("路径与命名检查")
    print("=" * 60)

    targets = {
        "run_pipeline.py": [
            "train_traditional_multi_target_backtest.py",
            "--traditional-model",
            "results/traditional_models",
            "data/predictions",
        ],
        "two_stage_backtest.py": [
            "--traditional-model",
            "traditional_model",
            "get_traditional_prediction_file",
        ],
    }

    all_good = True
    for filename, expected_tokens in targets.items():
        with open(script_path(filename), "r", encoding="utf-8") as handle:
            content = handle.read()
        for token in expected_tokens:
            if token not in content:
                print(f"  [FAIL] {filename}: 缺少关键内容 {token}")
                all_good = False
            else:
                print(f"  [OK] {filename}: 已包含 {token}")
    return all_good


def check_dev_tool_locations() -> bool:
    print("\n" + "=" * 60)
    print("辅助脚本目录检查")
    print("=" * 60)

    tool_files = [
        "check_env.py",
        "generate_mock_data.py",
        "simple_validation.py",
        "test_api.py",
        "test_explanation_feature.py",
    ]
    all_good = True
    for filename in tool_files:
        filepath = os.path.join(SCRIPT_DIR, filename)
        if os.path.exists(filepath):
            print(f"  [OK] scripts/dev_tools/{filename}")
        else:
            print(f"  [FAIL] scripts/dev_tools/{filename}: 文件不存在")
            all_good = False
    return all_good


def main() -> bool:
    print("ETF 研究项目轻量验证")
    print("=" * 60)

    checks = [
        ("语法检查", check_syntax),
        ("路径与命名检查", check_pipeline_references),
        ("辅助脚本目录检查", check_dev_tool_locations),
    ]

    results = []
    for check_name, check_func in checks:
        try:
            success = check_func()
        except Exception as exc:
            print(f"  [FAIL] {check_name}: {exc}")
            success = False
        results.append((check_name, success))

    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    all_passed = True
    for check_name, success in results:
        print(f"{check_name}: {'[PASS]' if success else '[FAIL]'}")
        all_passed = all_passed and success
    return all_passed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

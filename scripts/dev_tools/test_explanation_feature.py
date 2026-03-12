"""Lightweight self-checks for explanation-related modules."""

from __future__ import annotations

import os
import shutil
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPTS_DIR)


def build_mock_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["510300", "510500", "159915"],
            "name": ["沪深300ETF", "中证500ETF", "创业板ETF"],
            "momentum_20": [0.10, 0.20, 0.15],
            "volatility_20": [0.05, 0.06, 0.07],
            "return_mean_20": [0.01, -0.02, 0.03],
            "y_pred_Y_future_5d_return": [0.03, 0.01, 0.02],
        }
    )


def cleanup_logs() -> None:
    if os.path.exists("./test_logs"):
        shutil.rmtree("./test_logs")


def test_ranking_only() -> bool:
    from llm_ranking import rank_etfs_by_llm

    mock_data = build_mock_data()
    mock_response = """{"rankings": [
        {"code": "510300", "score": 0.95},
        {"code": "159915", "score": 0.88},
        {"code": "510500", "score": 0.81}
    ]}"""

    with patch("llm_ranking.call_deepseek_api") as mock_api:
        mock_api.return_value = mock_response
        result = rank_etfs_by_llm(mock_data, "2024-01-01", enable_explanations=True)

    assert result is not None
    assert "llm_score" in result.columns
    assert "llm_explanation" not in result.columns
    assert result["llm_score"].dtype in [np.float64, np.float32, float]
    return True


def test_explanation_generation() -> bool:
    from llm_ranking import generate_explanations_for_date

    ranked_data = build_mock_data().copy()
    ranked_data["llm_score"] = [0.95, 0.81, 0.88]

    mock_response = """{
        "rankings": [
            {"code": "510300", "score": 0.95, "explanation": "动量与排序分数均领先"},
            {"code": "510500", "score": 0.81, "explanation": "波动率适中，但排序靠后"},
            {"code": "159915", "score": 0.88, "explanation": "收益均值较高，综合得分较优"}
        ],
        "summary": {
            "market_context": "候选 ETF 呈现分化。",
            "key_factors": ["动量因子", "波动率约束"],
            "risk_considerations": "需关注高波动品种。"
        }
    }"""

    with patch("llm_ranking.call_deepseek_api") as mock_api:
        mock_api.return_value = mock_response
        result = generate_explanations_for_date(ranked_data, "2024-01-01", log_dir="./test_logs")

    assert result is not None
    assert "llm_explanation" in result.columns
    assert result["llm_explanation"].str.len().gt(0).all()
    assert os.path.exists("./test_logs/explanations/20240101/structured_explanation.json")
    cleanup_logs()
    return True


def test_mock_mode() -> bool:
    from llm_ranking import generate_explanations_for_date, rank_etfs_by_llm

    mock_data = build_mock_data()
    with patch("llm_ranking.call_deepseek_api") as mock_api:
        ranked = rank_etfs_by_llm(
            mock_data,
            "2024-01-01",
            log_dir="./test_logs",
            mock=True,
            score_reference_col="y_pred_Y_future_5d_return",
        )
        explained = generate_explanations_for_date(
            ranked.head(2),
            "2024-01-01",
            log_dir="./test_logs",
            mock=True,
        )

    assert ranked is not None
    assert explained is not None
    assert not mock_api.called
    cleanup_logs()
    return True


def test_configuration() -> bool:
    from two_stage_backtest import parse_args

    original_argv = sys.argv[:]
    try:
        sys.argv = ["two_stage_backtest.py"]
        args = parse_args()
        assert args.enable_explanations is False
        assert args.explanation_date is None
        assert args.explanation_sample_size == 3

        sys.argv = [
            "two_stage_backtest.py",
            "--enable-explanations",
            "--explanation-date",
            "2024-01-05",
            "--explanation-sample-size",
            "5",
        ]
        args = parse_args()
        assert args.enable_explanations is True
        assert args.explanation_date == "2024-01-05"
        assert args.explanation_sample_size == 5
    finally:
        sys.argv = original_argv
    return True


def test_utility_modules() -> bool:
    from explanation_utils import extract_key_insights, validate_explanation_format

    test_result = {
        "rankings": [
            {"code": "510300", "score": 0.95, "explanation": "动量强劲"},
            {"code": "510500", "score": 0.87, "explanation": "波动率可控"},
        ],
        "summary": {
            "market_context": "测试",
            "key_factors": ["动量"],
            "risk_considerations": "测试风险",
        },
    }

    validation = validate_explanation_format(test_result)
    insights = extract_key_insights(test_result)

    assert validation["is_valid"] is True
    assert "top_etfs" in insights
    return True


def main() -> bool:
    tests = [
        ("排序不附带解释", test_ranking_only),
        ("解释独立调用", test_explanation_generation),
        ("Mock 模式", test_mock_mode),
        ("参数解析", test_configuration),
        ("工具模块", test_utility_modules),
    ]

    all_passed = True
    for test_name, test_func in tests:
        try:
            success = test_func()
            print(f"{test_name}: {'[PASS]' if success else '[FAIL]'}")
        except Exception as exc:
            print(f"{test_name}: [FAIL] {exc}")
            success = False
        all_passed = all_passed and success

    cleanup_logs()
    return all_passed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

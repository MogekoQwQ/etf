"""Manual DeepSeek API connectivity check."""

from __future__ import annotations

import json
import os
import sys

import requests


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from llm_ranking import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, MODEL_NAME, rank_etfs_by_llm
from traditional_model_config import DEFAULT_TRADITIONAL_MODEL, get_traditional_llm_log_dir


def test_simple_chat() -> None:
    print("测试 DeepSeek API 调用")
    print(f"API URL: {DEEPSEEK_API_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"API Key prefix: {DEEPSEEK_API_KEY[:10]}...")

    messages = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "请说 hello"},
    ]

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 100,
            },
            timeout=30,
        )
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print(response.text)
    except Exception as exc:
        print(f"请求异常: {exc}")


def test_llm_ranking_function() -> None:
    import pandas as pd

    data = {
        "code": ["510300", "510050", "510500"],
        "name": ["沪深300ETF", "上证50ETF", "中证500ETF"],
        "日期": ["2025-01-01"] * 3,
        "momentum_20": [0.05, 0.03, 0.07],
        "volatility_20": [0.02, 0.015, 0.025],
        "volume_mean_20": [1000000, 800000, 1200000],
        "涨跌幅": [0.01, 0.005, 0.015],
        "振幅": [0.03, 0.02, 0.04],
    }
    df = pd.DataFrame(data)

    log_dir = os.path.join(
        get_traditional_llm_log_dir(PROJECT_ROOT, DEFAULT_TRADITIONAL_MODEL),
        "manual_api_test",
    )
    result = rank_etfs_by_llm(df, "2025-01-01", log_dir=log_dir)
    if result is None:
        print("rank_etfs_by_llm 返回 None")
    else:
        print(result[["code", "name", "llm_score"]])


if __name__ == "__main__":
    print("=" * 60)
    test_simple_chat()
    print("\n" + "=" * 60)
    test_llm_ranking_function()

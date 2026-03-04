"""
测试 DeepSeek API 调用
"""

import os
import sys
import json
import requests

# 使用 llm_ranking.py 中的配置
from llm_ranking import DEEPSEEK_API_URL, DEEPSEEK_API_KEY, MODEL_NAME, call_deepseek_api

def test_simple_chat():
    """测试简单的聊天请求"""
    print("测试 DeepSeek API 调用")
    print(f"API URL: {DEEPSEEK_API_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"API Key (前10位): {DEEPSEEK_API_KEY[:10]}...")

    messages = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "请说 'hello'"}
    ]

    print("\n发送请求...")
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 100
            },
            timeout=30
        )

        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")

        if response.status_code == 200:
            result = response.json()
            print(f"成功! 响应: {json.dumps(result, indent=2, ensure_ascii=False)}")

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"\n模型回复: {content}")
            else:
                print("响应中没有 choices 字段")
        else:
            print(f"错误! 响应: {response.text}")

    except Exception as e:
        print(f"请求异常: {e}")
        import traceback
        traceback.print_exc()

def test_llm_ranking_function():
    """测试 llm_ranking.py 中的函数"""
    print("\n测试 llm_ranking 函数...")

    # 创建模拟的 ETF 数据
    import pandas as pd
    import numpy as np

    # 创建3只ETF的模拟数据
    data = {
        'code': ['510300', '510050', '510500'],
        'name': ['沪深300ETF', '上证50ETF', '中证500ETF'],
        '日期': ['2025-01-01'] * 3,
        'momentum_20': [0.05, 0.03, 0.07],
        'volatility_20': [0.02, 0.015, 0.025],
        'volume_mean_20': [1000000, 800000, 1200000],
        '涨跌幅': [0.01, 0.005, 0.015],
        '振幅': [0.03, 0.02, 0.04],
    }

    df = pd.DataFrame(data)

    from llm_ranking import rank_etfs_by_llm

    print("调用 rank_etfs_by_llm...")
    result = rank_etfs_by_llm(df, "2025-01-01", log_dir="../results/two_stage/llm_logs_test")

    if result is not None:
        print("成功! 排序结果:")
        print(result[['code', 'name', 'llm_score']])
    else:
        print("rank_etfs_by_llm 返回 None")

if __name__ == "__main__":
    print("=" * 60)
    test_simple_chat()
    print("\n" + "=" * 60)
    test_llm_ranking_function()
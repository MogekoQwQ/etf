"""LLM ranking utilities for ETF cross-sectional sorting."""

from __future__ import annotations

import datetime
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yaml

from market_data_utils import detect_date_column, read_csv_with_fallback, standardize_date_column_name
from traditional_model_config import get_llm_config_local_file


LAST_GEMINI_REQUEST_TS: Optional[float] = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_LLM_CONFIG_PATH = get_llm_config_local_file(PROJECT_ROOT)
DEFAULT_LLM_MODEL = "deepseek-chat"

SUPPORTED_LLM_MODELS = (
    "deepseek-chat",
    "gemini-2.5-flash-lite",
)
MODEL_PROVIDER_MAP = {
    "deepseek-chat": "deepseek",
    "gemini-2.5-flash-lite": "gemini",
}
DEFAULT_API_URLS = {
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
ENV_API_KEY_MAP = {
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

MAX_RETRY = 3
RETRY_DELAY = 5
DEFAULT_TIMEOUT = 60
MIN_TIMEOUT = 60
MAX_TIMEOUT = 300
GEMINI_MIN_REQUEST_INTERVAL_SECONDS = 5.0
GEMINI_RATE_LIMIT_BACKOFF_SECONDS = 65
RANKING_MAX_COMPLETION_TOKENS = 1200
EXPLANATION_MAX_COMPLETION_TOKENS = 2400
RANKING_PROMPT_TEMPLATE = """你是一个用于 ETF 横截面排序的投资研究助手。

请基于给定 ETF 因子数据，在相同输入数据、相同排序目标下完成二阶段排序。

要求：
1. 仅根据输入的 ETF 因子与分数进行横截面比较，不要编造不存在的数据。
2. 输出必须是合法 JSON。
3. 只返回 `rankings` 字段，不要输出额外解释文本。
4. `rankings` 中只返回指定数量的 top ETF（具体数量由参数top_n_final指定），不要包含其他候选 ETF。
5. `score` 越大表示排序越靠前。
6. `code` 必须与输入 ETF 代码完全一致。

输出格式：
{{
  "rankings": [
    {{"code": "510300", "score": 0.95}}
  ]
}}

调仓日期: {date}
候选 ETF 数量: {candidate_count}
ETF 数据:
{data}
"""


EXPLANATION_PROMPT_TEMPLATE = """你是一个用于 ETF 二阶段排序解释的投资研究助手。

请基于已给出的 ETF 排序结果和因子信息，为每个 ETF 生成简洁解释，并同时输出结构化总结。

要求：
1. 输出必须是合法 JSON。
2. `rankings` 中每个 ETF 都必须包含 `code`、`score`、`explanation`。
3. `summary` 中包含 `market_context`、`key_factors`、`risk_considerations`。
4. 不要输出 JSON 之外的文本。

输出格式：
{{
  "rankings": [
    {{"code": "510300", "score": 0.95, "explanation": "解释文本"}}
  ],
  "summary": {{
    "market_context": "市场背景",
    "key_factors": ["因素1", "因素2"],
    "risk_considerations": "风险提示"
  }}
}}

调仓日期: {date}
ETF 数据:
{data}
"""


def load_llm_runtime_config(
    llm_model: str = DEFAULT_LLM_MODEL,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    if llm_model not in SUPPORTED_LLM_MODELS:
        raise ValueError(
            f"Unsupported second-stage LLM: {llm_model}. "
            f"Available options: {', '.join(SUPPORTED_LLM_MODELS)}"
        )

    resolved_config_path = config_path or DEFAULT_LLM_CONFIG_PATH
    file_config: Dict[str, Any] = {}
    legacy_json_config_path = os.path.join(PROJECT_ROOT, "config", "llm_config.json")
    effective_config_path = resolved_config_path
    if not os.path.exists(effective_config_path) and os.path.exists(legacy_json_config_path):
        effective_config_path = legacy_json_config_path

    if os.path.exists(effective_config_path):
        with open(effective_config_path, "r", encoding="utf-8") as handle:
            if effective_config_path.endswith((".yaml", ".yml")):
                file_config = yaml.safe_load(handle) or {}
            else:
                file_config = json.load(handle)

    model_config = file_config.get("models", {}).get(llm_model, {})
    provider = str(model_config.get("provider") or MODEL_PROVIDER_MAP[llm_model]).strip().lower()
    model_name = str(model_config.get("model") or llm_model).strip()
    api_key = str(
        model_config.get("api_key")
        or os.environ.get(ENV_API_KEY_MAP.get(provider, ""), "")
    ).strip()
    api_url = str(model_config.get("api_url") or DEFAULT_API_URLS.get(provider, "")).strip()
    temperature = float(model_config.get("temperature", 0.1))

    if provider not in {"deepseek", "gemini", "openrouter"}:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    if not api_key:
        raise ValueError(
            f"Missing API key for {llm_model}. "
            f"Configure `{effective_config_path}` or set `{ENV_API_KEY_MAP.get(provider, 'API_KEY')}`."
        )
    if not api_url:
        raise ValueError(f"Missing api_url for {llm_model}.")

    return {
        "provider": provider,
        "model": model_name,
        "api_key": api_key,
        "api_url": api_url,
        "temperature": temperature,
        "config_path": effective_config_path,
    }


def estimate_llm_timeout(
    messages: List[Dict[str, str]],
    max_tokens: int,
    etf_count: Optional[int] = None,
    mode: str = "ranking",
) -> int:
    total_chars = sum(len(message.get("content", "")) for message in messages)
    estimated_input_tokens = max(1, total_chars // 4)

    if etf_count is None:
        etf_count = max(1, total_chars // 300)

    if mode == "explanation":
        per_etf_output_tokens = 180
        summary_tokens = 420
    else:
        per_etf_output_tokens = 35
        summary_tokens = 120

    estimated_output_tokens = min(max_tokens, etf_count * per_etf_output_tokens + summary_tokens)

    estimated_seconds = DEFAULT_TIMEOUT
    estimated_seconds += estimated_input_tokens / 80.0
    estimated_seconds += estimated_output_tokens / 50.0
    return max(MIN_TIMEOUT, min(MAX_TIMEOUT, int(round(estimated_seconds))))


def throttle_gemini_requests() -> None:
    global LAST_GEMINI_REQUEST_TS

    now = time.monotonic()
    if LAST_GEMINI_REQUEST_TS is not None:
        elapsed = now - LAST_GEMINI_REQUEST_TS
        wait_seconds = GEMINI_MIN_REQUEST_INTERVAL_SECONDS - elapsed
        if wait_seconds > 0:
            print(f"Gemini throttle sleep: {wait_seconds:.2f}s")
            time.sleep(wait_seconds)

    LAST_GEMINI_REQUEST_TS = time.monotonic()


def get_gemini_retry_wait_seconds(exc: requests.exceptions.RequestException) -> Optional[float]:
    response = getattr(exc, "response", None)
    if response is None or response.status_code != 429:
        return None

    retry_after_header = str(response.headers.get("Retry-After", "")).strip()
    if retry_after_header:
        try:
            return max(float(retry_after_header), GEMINI_MIN_REQUEST_INTERVAL_SECONDS)
        except ValueError:
            pass

    return float(GEMINI_RATE_LIMIT_BACKOFF_SECONDS)


def call_deepseek_api(
    runtime_config: Dict[str, Any],
    messages: List[Dict[str, str]],
    max_retry: int = MAX_RETRY,
    timeout: Optional[int] = None,
    max_tokens: int = RANKING_MAX_COMPLETION_TOKENS,
    etf_count: Optional[int] = None,
    mode: str = "ranking",
) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {runtime_config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": runtime_config["model"],
        "messages": messages,
        "temperature": runtime_config["temperature"],
        "max_tokens": max_tokens,
    }

    effective_timeout = timeout or estimate_llm_timeout(messages, max_tokens, etf_count=etf_count, mode=mode)
    for attempt in range(1, max_retry + 1):
        try:
            response = requests.post(
                runtime_config["api_url"],
                headers=headers,
                json=payload,
                timeout=effective_timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as exc:
            print(f"DeepSeek API request failed: {exc}")
            if attempt < max_retry:
                time.sleep(RETRY_DELAY)
            else:
                return None
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"DeepSeek response parsing failed: {exc}")
            return None
    return None


def _resolve_openai_compatible_api_url(runtime_config: Dict[str, Any]) -> str:
    api_url = str(runtime_config["api_url"]).rstrip("/")
    if api_url.endswith("/chat/completions"):
        return api_url
    return f"{api_url}/chat/completions"


def call_openrouter_api(
    runtime_config: Dict[str, Any],
    messages: List[Dict[str, str]],
    max_retry: int = MAX_RETRY,
    timeout: Optional[int] = None,
    max_tokens: int = RANKING_MAX_COMPLETION_TOKENS,
    etf_count: Optional[int] = None,
    mode: str = "ranking",
) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {runtime_config['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": runtime_config["model"],
        "messages": messages,
        "temperature": runtime_config["temperature"],
        "max_tokens": max_tokens,
    }

    effective_timeout = timeout or estimate_llm_timeout(messages, max_tokens, etf_count=etf_count, mode=mode)
    api_url = _resolve_openai_compatible_api_url(runtime_config)
    for attempt in range(1, max_retry + 1):
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=effective_timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as exc:
            print(f"OpenRouter API request failed: {exc}")
            if attempt < max_retry:
                time.sleep(RETRY_DELAY)
            else:
                return None
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"OpenRouter response parsing failed: {exc}")
            return None
    return None


def call_gemini_api(
    runtime_config: Dict[str, Any],
    messages: List[Dict[str, str]],
    max_retry: int = MAX_RETRY,
    timeout: Optional[int] = None,
    max_tokens: int = RANKING_MAX_COMPLETION_TOKENS,
    etf_count: Optional[int] = None,
    mode: str = "ranking",
) -> Optional[str]:
    system_text = "\n\n".join(message["content"] for message in messages if message.get("role") == "system")
    user_text = "\n\n".join(message["content"] for message in messages if message.get("role") == "user")
    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "generationConfig": {
            "temperature": runtime_config["temperature"],
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    effective_timeout = timeout or estimate_llm_timeout(messages, max_tokens, etf_count=etf_count, mode=mode)
    for attempt in range(1, max_retry + 1):
        try:
            throttle_gemini_requests()
            response = requests.post(
                runtime_config["api_url"],
                headers={"Content-Type": "application/json"},
                params={"key": runtime_config["api_key"]},
                json=payload,
                timeout=effective_timeout,
            )
            response.raise_for_status()
            result = response.json()
            parts = result["candidates"][0]["content"]["parts"]
            return "\n".join(str(part.get("text", "")) for part in parts if part.get("text"))
        except requests.exceptions.RequestException as exc:
            print(f"Gemini API request failed: {exc}")
            if attempt < max_retry:
                retry_wait_seconds = get_gemini_retry_wait_seconds(exc)
                if retry_wait_seconds is not None:
                    print(f"Gemini rate limit backoff: {retry_wait_seconds:.2f}s")
                    time.sleep(retry_wait_seconds)
                else:
                    time.sleep(RETRY_DELAY)
            else:
                return None
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"Gemini response parsing failed: {exc}")
            return None
    return None


def call_llm_api(
    messages: List[Dict[str, str]],
    llm_model: str = DEFAULT_LLM_MODEL,
    config_path: Optional[str] = None,
    max_retry: int = MAX_RETRY,
    timeout: Optional[int] = None,
    max_tokens: int = RANKING_MAX_COMPLETION_TOKENS,
    etf_count: Optional[int] = None,
    mode: str = "ranking",
) -> Optional[str]:
    runtime_config = load_llm_runtime_config(llm_model=llm_model, config_path=config_path)
    if runtime_config["provider"] == "deepseek":
        return call_deepseek_api(
            runtime_config=runtime_config,
            messages=messages,
            max_retry=max_retry,
            timeout=timeout,
            max_tokens=max_tokens,
            etf_count=etf_count,
            mode=mode,
        )
    if runtime_config["provider"] == "openrouter":
        return call_openrouter_api(
            runtime_config=runtime_config,
            messages=messages,
            max_retry=max_retry,
            timeout=timeout,
            max_tokens=max_tokens,
            etf_count=etf_count,
            mode=mode,
        )
    return call_gemini_api(
        runtime_config=runtime_config,
        messages=messages,
        max_retry=max_retry,
        timeout=timeout,
        max_tokens=max_tokens,
        etf_count=etf_count,
        mode=mode,
    )


def _normalized_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return pd.Series(0.5, index=series.index, dtype=float)

    ranked = numeric.rank(method="average", pct=True, ascending=not higher_is_better)
    return ranked.fillna(0.5).astype(float)


def _resolve_date_column(df: pd.DataFrame) -> Optional[str]:
    return detect_date_column(df)


def _extract_factor_columns(etf_data: pd.DataFrame) -> List[str]:
    date_column = _resolve_date_column(etf_data)
    exclude_cols = {"code", "name", "llm_score", "llm_explanation"}
    if date_column:
        exclude_cols.add(date_column)
    exclude_cols.update(
        {
            "Y_next_day_return",
            "Y_future_5d_return",
            "Y_future_10d_return",
            "Y_future_5d_vol_change",
            "Y_future_10d_vol_change",
        }
    )
    return [col for col in etf_data.columns if col not in exclude_cols]


def _serialize_etf_payload(
    etf_data: pd.DataFrame,
    factor_cols: List[str],
    include_llm_score: bool = False,
) -> str:
    data_lines: List[str] = []
    for _, row in etf_data.iterrows():
        parts = []
        if include_llm_score and "llm_score" in row.index:
            parts.append(f"llm_score: {round(float(row['llm_score']), 6)}")

        for col in factor_cols:
            value = row[col]
            if isinstance(value, (float, np.floating, int, np.integer)):
                display_value = round(float(value), 6)
            else:
                display_value = value
            parts.append(f"{col}: {display_value}")

        data_lines.append(f"ETF {row['code']}: " + ", ".join(parts))
    return "\n".join(data_lines)


def _build_json_parse_candidates(response_text: str) -> List[str]:
    stripped_text = str(response_text or "").strip()
    if not stripped_text:
        return [stripped_text]

    # Handle common Gemini wrappers such as ```json ... ``` or a leading `json` label.
    code_fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped_text, flags=re.IGNORECASE | re.DOTALL)
    if code_fence_match:
        stripped_text = code_fence_match.group(1).strip()
    stripped_text = re.sub(r"^\s*json\s*(?=[{\[])", "", stripped_text, count=1, flags=re.IGNORECASE)

    json_match = re.search(r"\{.*\}", stripped_text, re.DOTALL)
    json_str = json_match.group(0).strip() if json_match else stripped_text

    parse_candidates = [json_str]
    trailing_comma_fixed = re.sub(r",(\s*[}\]])", r"\1", json_str)
    if trailing_comma_fixed != json_str:
        parse_candidates.append(trailing_comma_fixed)
    return parse_candidates


def _parse_json_response(response_text: str) -> Dict[str, Any]:
    parse_candidates = _build_json_parse_candidates(response_text)
    if not parse_candidates or not parse_candidates[0]:
        raise json.JSONDecodeError("Empty LLM response", str(response_text or ""), 0)

    last_error: Optional[json.JSONDecodeError] = None
    for candidate in parse_candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError(
        "Failed to parse LLM JSON response",
        parse_candidates[0] if parse_candidates else str(response_text or ""),
        0,
    )


def _save_json_parse_debug_artifact(
    log_dir: Optional[str],
    response_text: str,
    exc: Exception,
    llm_model: str,
) -> None:
    if not log_dir:
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidates = _build_json_parse_candidates(response_text)
    sections = [
        f"debug_type: ranking_json_parse_failure",
        f"llm_model: {llm_model}",
        f"error_type: {type(exc).__name__}",
        f"error_message: {exc}",
        "",
        "[raw_response]",
        str(response_text or ""),
    ]
    for idx, candidate in enumerate(candidates, start=1):
        sections.extend(
            [
                "",
                f"[parse_candidate_{idx}]",
                candidate,
            ]
        )
    _save_text_artifact(log_dir, f"parse_debug_{timestamp}.txt", "\n".join(sections))


def _build_mock_llm_result(
    etf_data: pd.DataFrame,
    score_reference_col: Optional[str] = None,
) -> Dict[str, Any]:
    df = etf_data.copy()
    score_parts: List[pd.Series] = []

    if score_reference_col and score_reference_col in df.columns:
        score_parts.append(_normalized_rank(df[score_reference_col], higher_is_better=True) * 0.45)

    weighted_factors = [
        ("momentum_20", 0.20, True),
        ("return_mean_20", 0.15, True),
        ("MA_5", 0.05, True),
        ("MA_10", 0.05, True),
        ("volatility_20", 0.10, False),
        ("amplitude_mean_20", 0.05, False),
    ]

    for column, weight, higher_is_better in weighted_factors:
        if column in df.columns:
            score_parts.append(_normalized_rank(df[column], higher_is_better=higher_is_better) * weight)

    if score_parts:
        combined_score = sum(score_parts)
    else:
        combined_score = pd.Series(np.linspace(1.0, 0.0, len(df)), index=df.index, dtype=float)

    min_score = float(combined_score.min())
    max_score = float(combined_score.max())
    if max_score > min_score:
        normalized_score = (combined_score - min_score) / (max_score - min_score)
    else:
        normalized_score = pd.Series(0.5, index=df.index, dtype=float)

    rankings: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        rankings.append(
            {
                "code": str(row["code"]),
                "score": round(float(normalized_score.loc[idx]), 6),
            }
        )
    rankings.sort(key=lambda item: item["score"], reverse=True)
    return {"rankings": rankings}


def _build_mock_explanation_result(etf_data: pd.DataFrame) -> Dict[str, Any]:
    df = etf_data.copy().reset_index(drop=True)
    rankings: List[Dict[str, Any]] = []
    key_factors: List[str] = []

    if "momentum_20" in df.columns:
        key_factors.append("momentum_20")
    if "volatility_20" in df.columns:
        key_factors.append("volatility_20")
    if "return_mean_20" in df.columns:
        key_factors.append("return_mean_20")
    if "llm_score" in df.columns:
        key_factors.append("llm_score")

    for _, row in df.iterrows():
        explanation_parts = []
        if "llm_score" in df.columns:
            explanation_parts.append(f"llm_score={float(row['llm_score']):.3f}")
        if "momentum_20" in df.columns:
            explanation_parts.append("momentum is relatively strong")
        if "volatility_20" in df.columns:
            explanation_parts.append("volatility is under review")

        rankings.append(
            {
                "code": str(row["code"]),
                "score": round(float(row.get("llm_score", 0.0)), 6),
                "explanation": "; ".join(explanation_parts[:3]),
            }
        )

    return {
        "rankings": rankings,
        "summary": {
            "market_context": "Mock explanation result for local fallback.",
            "key_factors": key_factors[:5] or ["llm_score"],
            "risk_considerations": "Mock explanations are only for local testing.",
        },
    }


def _save_text_artifact(log_dir: Optional[str], filename: str, content: str) -> None:
    if not log_dir:
        return
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _save_ranking_artifacts(
    result: Dict[str, Any],
    response_text: str,
    etf_data: pd.DataFrame,
    factor_cols: List[str],
    log_dir: Optional[str],
    llm_model: str,
) -> pd.DataFrame:
    rankings = result.get("rankings", [])
    if not isinstance(rankings, list):
        raise ValueError("rankings must be a list.")

    code_to_score: Dict[str, float] = {}
    for item in rankings:
        if "code" in item and "score" in item:
            code_to_score[str(item["code"])] = float(item["score"])

    missing_codes = set(etf_data["code"].astype(str)) - set(code_to_score.keys())
    if missing_codes:
        min_score = min(code_to_score.values()) if code_to_score else 0.0
        for code in missing_codes:
            code_to_score[code] = min_score - 0.01

    ranked = etf_data.copy()
    ranked["code"] = ranked["code"].astype(str)
    ranked["llm_score"] = ranked["code"].map(code_to_score)
    ranked["second_stage_llm"] = llm_model
    ranked = ranked.sort_values("llm_score", ascending=False).reset_index(drop=True)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_text_artifact(log_dir, f"response_{timestamp}.txt", response_text)
        result_file = os.path.join(log_dir, f"result_{timestamp}.csv")
        cols_to_save = ["code", "name", "llm_score", "second_stage_llm"]
        cols_to_save.extend([col for col in factor_cols if col not in cols_to_save])
        ranked[cols_to_save].to_csv(result_file, index=False, encoding="utf-8-sig")

    return ranked


def _merge_explanations_into_dataframe(
    result: Dict[str, Any],
    etf_data: pd.DataFrame,
) -> pd.DataFrame:
    rankings = result.get("rankings", [])
    if not isinstance(rankings, list):
        raise ValueError("rankings must be a list.")

    code_to_explanation: Dict[str, str] = {}
    code_to_score: Dict[str, float] = {}
    for item in rankings:
        code = str(item.get("code", "") or "")
        if not code:
            continue
        code_to_explanation[code] = str(item.get("explanation", "") or "")
        if "score" in item:
            code_to_score[code] = float(item["score"])

    explained = etf_data.copy()
    explained["code"] = explained["code"].astype(str)
    explained["llm_explanation"] = explained["code"].map(code_to_explanation).fillna("")
    if "llm_score" not in explained.columns and code_to_score:
        explained["llm_score"] = explained["code"].map(code_to_score)
    return explained


def _save_explanation_artifacts(
    result: Dict[str, Any],
    response_text: str,
    etf_data: pd.DataFrame,
    log_dir: Optional[str],
    date: str,
) -> None:
    if not log_dir:
        return

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_text_artifact(log_dir, f"explanation_response_{timestamp}.txt", response_text)

    explanation_file = os.path.join(log_dir, f"explanation_{timestamp}.json")
    with open(explanation_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    from explanation_utils import ExplanationStorage

    storage = ExplanationStorage(log_dir)
    storage.save_explanation(
        date=date,
        result=result,
        etf_data=etf_data,
        enable_explanations=True,
    )


def rank_etfs_by_llm(
    etf_data: pd.DataFrame,
    date: str,
    log_dir: Optional[str] = None,
    enable_explanations: bool = False,
    mock: bool = False,
    top_n_final: int = 10,
    score_reference_col: Optional[str] = None,
    api_timeout: Optional[int] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    config_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    if "code" not in etf_data.columns:
        print("Missing `code` column in ETF data.")
        return None

    if enable_explanations:
        print("Ranking step received enable_explanations=True; explanations are still generated separately.")

    ranked_input = etf_data.copy()
    ranked_input["code"] = ranked_input["code"].astype(str)

    factor_cols = _extract_factor_columns(ranked_input)
    data_str = _serialize_etf_payload(ranked_input, factor_cols)
    prompt = RANKING_PROMPT_TEMPLATE.format(
        date=date,
        data=data_str,
        candidate_count=len(ranked_input),
    )
    prompt += (
        f"\nOnly return the top {int(top_n_final)} ETFs in `rankings`. "
        "Do not include the remaining candidates."
    )

    if log_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_text_artifact(log_dir, f"prompt_{timestamp}.txt", prompt)

    if mock:
        result = _build_mock_llm_result(
            etf_data=ranked_input,
            score_reference_col=score_reference_col,
        )
        response_text = json.dumps(result, ensure_ascii=False, indent=2)
        return _save_ranking_artifacts(
            result=result,
            response_text=response_text,
            etf_data=ranked_input,
            factor_cols=factor_cols,
            log_dir=log_dir,
            llm_model=llm_model,
        )

    messages = [
        {"role": "system", "content": "You are an ETF ranking assistant. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    response_text = call_llm_api(
        messages=messages,
        llm_model=llm_model,
        config_path=config_path,
        timeout=api_timeout,
        max_tokens=RANKING_MAX_COMPLETION_TOKENS,
        etf_count=len(ranked_input),
        mode="ranking",
    )
    if response_text is None:
        print("LLM ranking request failed.")
        return None

    try:
        result = _parse_json_response(response_text)
        return _save_ranking_artifacts(
            result=result,
            response_text=response_text,
            etf_data=ranked_input,
            factor_cols=factor_cols,
            log_dir=log_dir,
            llm_model=llm_model,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        _save_json_parse_debug_artifact(
            log_dir=log_dir,
            response_text=response_text,
            exc=exc,
            llm_model=llm_model,
        )
        print(f"Failed to parse LLM ranking response: {exc}")
        return None


def generate_explanations_for_date(
    etf_data: pd.DataFrame,
    date: str,
    log_dir: Optional[str] = None,
    mock: bool = False,
    api_timeout: Optional[int] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    config_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    if etf_data.empty:
        print("No ETF data available for explanation generation.")
        return None

    explained_input = etf_data.copy().reset_index(drop=True)
    explained_input["code"] = explained_input["code"].astype(str)
    if "llm_score" not in explained_input.columns:
        print("Explanation generation requires `llm_score` in the ETF dataframe.")
        return None

    factor_cols = _extract_factor_columns(explained_input)
    data_str = _serialize_etf_payload(explained_input, factor_cols, include_llm_score=True)
    prompt = EXPLANATION_PROMPT_TEMPLATE.format(date=date, data=data_str)

    if log_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _save_text_artifact(log_dir, f"explanation_prompt_{timestamp}.txt", prompt)

    if mock:
        result = _build_mock_explanation_result(explained_input)
        response_text = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        messages = [
            {"role": "system", "content": "You are an ETF explanation assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        response_text = call_llm_api(
            messages=messages,
            llm_model=llm_model,
            config_path=config_path,
            timeout=api_timeout,
            max_tokens=EXPLANATION_MAX_COMPLETION_TOKENS,
            etf_count=len(explained_input),
            mode="explanation",
        )
        if response_text is None:
            print("LLM explanation request failed.")
            return None

        try:
            result = _parse_json_response(response_text)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"Failed to parse LLM explanation response: {exc}")
            return None

    explained_output = _merge_explanations_into_dataframe(result, explained_input)

    try:
        _save_explanation_artifacts(
            result=result,
            response_text=response_text,
            etf_data=explained_output,
            log_dir=log_dir,
            date=date,
        )
    except Exception as exc:
        print(f"Failed to save explanation artifacts: {exc}")
        return None

    return explained_output


def process_rebalancing_dates(
    data_path: str,
    output_dir: str,
    target: str = "Y_future_5d_return",
    top_n_first: int = 50,
    top_n_final: int = 10,
    rebalancing_freq: str = "W",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    enable_explanations: bool = False,
    mock: bool = False,
    api_timeout: Optional[int] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    config_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    df = read_csv_with_fallback(data_path)
    df = standardize_date_column_name(df)
    date_column = _resolve_date_column(df)
    if date_column is None:
        print("Date column is missing from the input dataset.")
        return None

    df[date_column] = pd.to_datetime(df[date_column])
    if start_date:
        df = df[df[date_column] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_column] <= pd.to_datetime(end_date)]

    pred_col = f"y_pred_{target}"
    if pred_col not in df.columns:
        print(f"Prediction column is missing: {pred_col}")
        return None

    dates = pd.Series(pd.to_datetime(df[date_column].unique())).sort_values().reset_index(drop=True)
    if rebalancing_freq == "W":
        rebalance_dates = dates.groupby(dates.dt.to_period("W-FRI")).max()
    elif rebalancing_freq == "M":
        rebalance_dates = dates.groupby(dates.dt.to_period("M")).max()
    else:
        rebalance_dates = dates
    rebalance_dates = pd.to_datetime(pd.Series(rebalance_dates).dropna().unique())

    all_results: List[Dict[str, Any]] = []
    for date in rebalance_dates:
        day_data = df[df[date_column] == date].copy()
        if len(day_data) < top_n_first:
            continue

        top_etfs = day_data.sort_values(pred_col, ascending=False).head(top_n_first).copy()
        ranked_etfs = rank_etfs_by_llm(
            top_etfs,
            str(date.date()),
            log_dir=output_dir,
            enable_explanations=False,
            mock=mock,
            top_n_final=top_n_final,
            score_reference_col=pred_col,
            api_timeout=api_timeout,
            llm_model=llm_model,
            config_path=config_path,
        )
        if ranked_etfs is None:
            ranked_etfs = top_etfs.copy()
            ranked_etfs["llm_score"] = ranked_etfs[pred_col]

        final_etfs = ranked_etfs.head(top_n_final).copy()
        if enable_explanations:
            generate_explanations_for_date(
                final_etfs,
                str(date.date()),
                log_dir=output_dir,
                mock=mock,
                api_timeout=api_timeout,
                llm_model=llm_model,
                config_path=config_path,
            )

        all_results.append(
            {
                "rebalance_date": date,
                "second_stage_llm": llm_model,
                "traditional_top_codes": list(top_etfs["code"]),
                "llm_ranked_codes": list(ranked_etfs["code"]),
                "final_codes": list(final_etfs["code"]),
            }
        )

        result_df = pd.DataFrame(
            {
                "date": date,
                "code": final_etfs["code"],
                "name": final_etfs["name"] if "name" in final_etfs.columns else "",
                "llm_score": final_etfs["llm_score"],
                "traditional_score": final_etfs[pred_col],
                "second_stage_llm": llm_model,
            }
        )
        result_file = os.path.join(output_dir, f"rebalance_{date.strftime('%Y%m%d')}.csv")
        result_df.to_csv(result_file, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(output_dir, "rebalancing_summary.csv"), index=False, encoding="utf-8-sig")
    return summary_df


if __name__ == "__main__":
    process_rebalancing_dates(
        data_path="./data/processed/all_etf_factors.csv",
        output_dir="./runs/backtests/<traditional_model>/logs/deepseek_chat",
        target="Y_future_5d_return",
        top_n_first=50,
        top_n_final=10,
        rebalancing_freq="W",
        start_date="2023-01-01",
        end_date="2024-12-31",
    )

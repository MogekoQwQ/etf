"""CSV encoding helpers and ETF market column normalization."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd


CSV_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "gb18030", "gbk")

STANDARD_MARKET_COLUMNS = (
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌幅",
    "换手率",
)

FULL_MARKET_SCHEMA = (
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌幅",
    "涨跌额",
    "换手率",
)

MARKET_COLUMN_ALIASES = {
    "日期": ("日期", "date", "datetime", "交易日期", "时间"),
    "开盘": ("开盘", "开盘价", "今开", "open"),
    "收盘": ("收盘", "收盘价", "close", "最新价"),
    "最高": ("最高", "最高价", "high"),
    "最低": ("最低", "最低价", "low"),
    "成交量": ("成交量", "成交量(手)", "volume", "vol"),
    "成交额": ("成交额", "成交金额", "amount", "turnover"),
    "振幅": ("振幅", "amplitude"),
    "涨跌幅": ("涨跌幅", "涨跌幅%", "changepercent", "pct_chg"),
    "涨跌额": ("涨跌额", "change", "price_change"),
    "换手率": ("换手率", "turnoverrate", "turnover_rate"),
}


def read_csv_with_fallback(
    file_path: str,
    *args,
    encoding_candidates: Sequence[str] = CSV_ENCODING_CANDIDATES,
    **kwargs,
) -> pd.DataFrame:
    if kwargs.get("encoding"):
        return pd.read_csv(file_path, *args, **kwargs)

    last_error: Optional[Exception] = None
    for encoding in encoding_candidates:
        try:
            return pd.read_csv(file_path, *args, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return pd.read_csv(file_path, *args, **kwargs)


def normalize_column_name(column_name: object) -> str:
    return str(column_name).replace("\ufeff", "").strip()


def _normalize_label(label: object) -> str:
    return normalize_column_name(label).lower().replace(" ", "").replace("_", "").replace("-", "")


def standardize_date_column_name(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: normalize_column_name(column) for column in df.columns}
    standardized = df.rename(columns=renamed).copy()

    if "日期" in standardized.columns:
        return standardized

    for column in standardized.columns:
        if _normalize_label(column) in {_normalize_label(alias) for alias in MARKET_COLUMN_ALIASES["日期"]}:
            return standardized.rename(columns={column: "日期"})

    for column in standardized.columns:
        parsed = pd.to_datetime(standardized[column], errors="coerce")
        valid_ratio = float(parsed.notna().mean())
        if valid_ratio < 0.95:
            continue
        valid_values = parsed.dropna()
        if valid_values.empty:
            continue
        if valid_values.dt.year.min() < 2000 or valid_values.dt.year.max() > 2100:
            continue
        if valid_values.nunique() < 10:
            continue
        return standardized.rename(columns={column: "日期"})

    return standardized


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    standardized = standardize_date_column_name(df)
    if "日期" in standardized.columns:
        return "日期"
    return None


def standardize_etf_market_columns(
    df: pd.DataFrame,
    *,
    allow_position_fallback: bool = False,
) -> pd.DataFrame:
    standardized = standardize_date_column_name(df)
    rename_map: dict[str, str] = {}

    for column in standardized.columns:
        normalized_column = _normalize_label(column)
        for canonical_name, aliases in MARKET_COLUMN_ALIASES.items():
            alias_labels = {_normalize_label(canonical_name), *(_normalize_label(alias) for alias in aliases)}
            if normalized_column in alias_labels:
                rename_map[column] = canonical_name
                break

    standardized = standardized.rename(columns=rename_map)

    recognized_columns = [column for column in FULL_MARKET_SCHEMA if column in standardized.columns]
    if allow_position_fallback and len(recognized_columns) < 5 and len(standardized.columns) >= len(STANDARD_MARKET_COLUMNS):
        schema = FULL_MARKET_SCHEMA if len(standardized.columns) >= len(FULL_MARKET_SCHEMA) else STANDARD_MARKET_COLUMNS
        positional_map: dict[str, str] = {}
        for index, canonical_name in enumerate(schema):
            source_column = standardized.columns[index]
            if source_column == canonical_name:
                continue
            positional_map[source_column] = canonical_name
        standardized = standardized.rename(columns=positional_map)

    return standardized


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], source_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"{source_name} 缺少列: {', '.join(missing_columns)}")

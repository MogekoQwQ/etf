"""
Generate an offline HTML summary report for two-stage ETF backtest results.
"""

import argparse
import datetime
import html
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


METRIC_ALIASES = {
    "annual_return": [
        "annual_return",
        "annualized_return",
        "annualreturn",
        "annualizedreturn",
        "年化收益率",
        "年化收益",
    ],
    "annual_volatility": [
        "annual_volatility",
        "annualized_volatility",
        "annualvolatility",
        "annualizedvolatility",
        "volatility",
        "年化波动率",
        "波动率",
    ],
    "sharpe_ratio": [
        "sharpe_ratio",
        "sharperatio",
        "sharpe",
        "夏普比率",
        "夏普",
    ],
    "max_drawdown": [
        "max_drawdown",
        "maxdrawdown",
        "drawdown",
        "最大回撤",
        "回撤",
    ],
    "win_rate": [
        "win_rate",
        "winrate",
        "胜率",
    ],
    "total_periods": [
        "total_periods",
        "periods",
        "rebalancing_periods",
        "调仓周期数",
        "调仓期数",
        "周期数",
        "总周期数",
    ],
}

STRATEGY_ALIASES = {
    "traditional": [
        "strategy_a",
        "strategya",
        "a",
        "traditional",
        "baseline",
        "传统",
        "传统模型",
        "仅传统模型",
        "策略a",
    ],
    "two_stage": [
        "strategy_b",
        "strategyb",
        "b",
        "two_stage",
        "twostage",
        "两阶段",
        "二阶段",
        "策略b",
    ],
}

DISPLAY_METRICS = [
    ("annual_return", "年化收益率", "percent"),
    ("annual_volatility", "年化波动率", "percent"),
    ("sharpe_ratio", "夏普比率", "float"),
    ("max_drawdown", "最大回撤", "percent"),
    ("win_rate", "胜率", "percent"),
    ("total_periods", "调仓周期数", "integer"),
]

PARAMETER_LABELS = {
    "target": "预测目标",
    "top_n_first": "第一阶段候选数",
    "top_n_final": "最终持仓数",
    "rebalancing_freq": "调仓频率",
    "holding_days": "持有天数",
    "use_llm": "是否启用 LLM 排序",
    "mock_llm": "是否使用 Mock 排序",
    "enable_explanations": "是否启用解释功能",
    "explanation_date": "解释指定日期",
    "explanation_sample_size": "解释抽样数量",
    "selected_explanation_dates": "计划解释日期",
    "generated_explanation_dates": "实际生成解释日期",
    "generated_explanations_count": "实际解释日期数",
    "llm_timeout": "LLM 超时设置",
}


def normalize_text(value: Any) -> str:
    """Normalize labels for fuzzy matching."""
    text = str(value or "").strip().lower()
    for old, new in {
        "（": "(",
        "）": ")",
        "，": ",",
        "：": ":",
        " ": "",
        "_": "",
        "-": "",
        "、": "",
        "/": "",
        ".": "",
        "%": "",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        ":": "",
        ",": "",
    }.items():
        text = text.replace(old, new)
    return text


def match_alias(value: Any, alias_map: Dict[str, List[str]]) -> Optional[str]:
    """Return canonical key when a label matches aliases."""
    normalized = normalize_text(value)
    if not normalized:
        return None

    for canonical, aliases in alias_map.items():
        candidates = [canonical] + aliases
        for alias in candidates:
            alias_normalized = normalize_text(alias)
            if normalized == alias_normalized or alias_normalized in normalized or normalized in alias_normalized:
                return canonical
    return None


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def coerce_number(value: Any) -> Optional[float]:
    """Convert textual percentages or numeric strings to floats."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a", "null"}:
        return None

    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1]

    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None

    if is_percent:
        return number / 100
    return number


def format_value(value: Any, value_type: str) -> str:
    if is_missing(value):
        return "缺失"
    if value_type == "percent":
        return f"{float(value):.2%}"
    if value_type == "integer":
        return str(int(round(float(value))))
    return f"{float(value):.2f}"


def html_table(df: pd.DataFrame) -> str:
    """Render a compact HTML table with escaped cells."""
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrapper"><table class="report-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


class TwoStageReporter:
    """Build an HTML summary report from two-stage backtest artifacts."""

    def __init__(self, input_dir: str):
        self.input_dir = os.path.abspath(input_dir)
        self.metrics_path = os.path.join(self.input_dir, "performance_metrics.csv")
        self.report_path = os.path.join(self.input_dir, "performance_report.json")
        self.backtest_path = os.path.join(self.input_dir, "backtest_results.csv")
        self.image_names = [
            ("累计收益曲线", "cumulative_returns.png"),
            ("收益分布图", "returns_distribution.png"),
            ("绩效指标对比图", "performance_comparison.png"),
        ]
        self.warnings: List[str] = []

    def generate_report(self, output_path: str) -> str:
        metrics_comparison = self.load_metrics_comparison()
        report_data = self.load_json_report()
        backtest_preview = self.load_backtest_preview()
        html_content = self.build_html(metrics_comparison, report_data, backtest_preview, output_path)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html_content)

        print(f"HTML 汇总报告已生成: {output_path}")
        return output_path

    def load_json_report(self) -> Dict[str, Any]:
        if not os.path.exists(self.report_path):
            self.warnings.append("未找到 performance_report.json，参数信息与部分绩效补充内容不可用。")
            return {}

        try:
            with open(self.report_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            self.warnings.append(f"performance_report.json 读取失败: {exc}")
            return {}

    def load_metrics_comparison(self) -> Dict[str, Dict[str, Optional[float]]]:
        comparison = {"traditional": {}, "two_stage": {}}

        if os.path.exists(self.metrics_path):
            try:
                metrics_df = pd.read_csv(self.metrics_path)
                self._parse_metrics_dataframe(metrics_df, comparison)
            except Exception as exc:
                self.warnings.append(f"performance_metrics.csv 读取失败: {exc}")
        else:
            self.warnings.append("未找到 performance_metrics.csv，绩效指标表格将尽量使用 JSON 补全。")

        report_data = self.load_json_report()
        self._merge_report_metrics(report_data, comparison)

        for metric_key, label, _ in DISPLAY_METRICS:
            if is_missing(comparison["traditional"].get(metric_key)) or is_missing(comparison["two_stage"].get(metric_key)):
                self.warnings.append(f"绩效指标“{label}”缺少完整数据，报告中将显示为缺失。")

        return comparison

    def _parse_metrics_dataframe(
        self,
        metrics_df: pd.DataFrame,
        comparison: Dict[str, Dict[str, Optional[float]]],
    ) -> None:
        if metrics_df.empty:
            self.warnings.append("performance_metrics.csv 为空。")
            return

        frame = metrics_df.copy()
        if frame.columns.size > 0 and str(frame.columns[0]).startswith("Unnamed"):
            frame = frame.rename(columns={frame.columns[0]: "metric_label"})

        row_labels = frame.iloc[:, 0].tolist() if frame.shape[1] > 0 else []
        row_has_metrics = sum(1 for label in row_labels if match_alias(label, METRIC_ALIASES)) >= 2
        column_has_metrics = sum(1 for col in frame.columns if match_alias(col, METRIC_ALIASES)) >= 2
        row_has_strategies = sum(1 for label in row_labels if match_alias(label, STRATEGY_ALIASES)) >= 2
        column_has_strategies = sum(1 for col in frame.columns if match_alias(col, STRATEGY_ALIASES)) >= 2

        if row_has_metrics and column_has_strategies:
            metric_col = frame.columns[0]
            for _, row in frame.iterrows():
                metric_key = match_alias(row.get(metric_col), METRIC_ALIASES)
                if not metric_key:
                    continue
                for column in frame.columns[1:]:
                    strategy_key = match_alias(column, STRATEGY_ALIASES)
                    if strategy_key:
                        comparison[strategy_key][metric_key] = coerce_number(row.get(column))
            return

        if row_has_strategies and column_has_metrics:
            strategy_col = frame.columns[0]
            for _, row in frame.iterrows():
                strategy_key = match_alias(row.get(strategy_col), STRATEGY_ALIASES)
                if not strategy_key:
                    continue
                for column in frame.columns[1:]:
                    metric_key = match_alias(column, METRIC_ALIASES)
                    if metric_key:
                        comparison[strategy_key][metric_key] = coerce_number(row.get(column))
            return

        self.warnings.append("performance_metrics.csv 字段结构未完全识别，将优先使用 JSON 中的指标数据。")

    def _merge_report_metrics(
        self,
        report_data: Dict[str, Any],
        comparison: Dict[str, Dict[str, Optional[float]]],
    ) -> None:
        report_mapping = {
            "traditional": report_data.get("performance_strategy_a", {}),
            "two_stage": report_data.get("performance_strategy_b", {}),
        }

        for strategy_key, metrics in report_mapping.items():
            if not isinstance(metrics, dict):
                continue
            for metric_key, _, _ in DISPLAY_METRICS:
                existing_value = comparison[strategy_key].get(metric_key)
                if not is_missing(existing_value):
                    continue
                json_value = coerce_number(metrics.get(metric_key))
                if json_value is not None:
                    comparison[strategy_key][metric_key] = json_value

    def load_backtest_preview(self) -> pd.DataFrame:
        if not os.path.exists(self.backtest_path):
            self.warnings.append("未找到 backtest_results.csv，无法展示回测明细示例。")
            return pd.DataFrame()

        try:
            frame = pd.read_csv(self.backtest_path)
        except Exception as exc:
            self.warnings.append(f"backtest_results.csv 读取失败: {exc}")
            return pd.DataFrame()

        if frame.empty:
            self.warnings.append("backtest_results.csv 为空。")
            return frame

        preview = frame.head(10).copy()
        for column in preview.columns:
            if "return" in normalize_text(column):
                preview[column] = preview[column].map(
                    lambda value: format_value(value, "percent") if coerce_number(value) is not None else value
                )
        return preview

    def build_metric_rows(self, comparison: Dict[str, Dict[str, Optional[float]]]) -> str:
        rows = []
        for metric_key, label, value_type in DISPLAY_METRICS:
            traditional_value = comparison["traditional"].get(metric_key)
            two_stage_value = comparison["two_stage"].get(metric_key)
            rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{html.escape(format_value(traditional_value, value_type))}</td>"
                f"<td>{html.escape(format_value(two_stage_value, value_type))}</td>"
                "</tr>"
            )
        return "".join(rows)

    def build_summary(self, comparison: Dict[str, Dict[str, Optional[float]]]) -> str:
        phrases = []

        def append_change(metric_key: str, label: str, relation: str) -> None:
            traditional_value = comparison["traditional"].get(metric_key)
            two_stage_value = comparison["two_stage"].get(metric_key)
            if is_missing(traditional_value) or is_missing(two_stage_value):
                return

            old_text = format_value(traditional_value, "percent" if metric_key in {"annual_return", "max_drawdown", "annual_volatility", "win_rate"} else "float")
            new_text = format_value(two_stage_value, "percent" if metric_key in {"annual_return", "max_drawdown", "annual_volatility", "win_rate"} else "float")

            if metric_key == "total_periods":
                old_text = format_value(traditional_value, "integer")
                new_text = format_value(two_stage_value, "integer")

            if float(two_stage_value) == float(traditional_value):
                phrases.append(f"{label}保持不变，均为{new_text}")
                return

            if relation == "higher_better":
                action = "提升" if float(two_stage_value) > float(traditional_value) else "回落"
            else:
                action = "收敛" if float(two_stage_value) > float(traditional_value) else "扩大"

            phrases.append(f"{label}由{old_text}{action}至{new_text}")

        append_change("annual_return", "年化收益率", "higher_better")
        append_change("sharpe_ratio", "夏普比率", "higher_better")

        traditional_drawdown = comparison["traditional"].get("max_drawdown")
        two_stage_drawdown = comparison["two_stage"].get("max_drawdown")
        if not is_missing(traditional_drawdown) and not is_missing(two_stage_drawdown):
            old_text = format_value(traditional_drawdown, "percent")
            new_text = format_value(two_stage_drawdown, "percent")
            if float(two_stage_drawdown) == float(traditional_drawdown):
                phrases.append(f"最大回撤保持不变，均为{new_text}")
            elif float(two_stage_drawdown) > float(traditional_drawdown):
                phrases.append(f"最大回撤由{old_text}收窄至{new_text}")
            else:
                phrases.append(f"最大回撤由{old_text}扩大至{new_text}")

        append_change("win_rate", "胜率", "higher_better")
        append_change("total_periods", "调仓周期数", "higher_better")

        if not phrases:
            return "当前结果文件中缺少可用于自动生成摘要的完整对比数据。"

        return "；".join(phrases) + "。"

    def build_parameter_items(self, report_data: Dict[str, Any]) -> str:
        parameters = report_data.get("parameters", {}) if isinstance(report_data, dict) else {}
        if not isinstance(parameters, dict) or not parameters:
            return '<div class="empty-state">未提供可展示的参数信息。</div>'

        items = []
        for key, label in PARAMETER_LABELS.items():
            if key not in parameters:
                continue
            value = parameters[key]
            if isinstance(value, bool):
                display_value = "是" if value else "否"
            elif value is None:
                display_value = "未设置"
            else:
                display_value = str(value)
            items.append(
                '<div class="param-card">'
                f'<div class="param-label">{html.escape(label)}</div>'
                f'<div class="param-value">{html.escape(display_value)}</div>'
                "</div>"
            )

        if not items:
            return '<div class="empty-state">未提供可展示的参数信息。</div>'
        return "".join(items)

    def build_image_sections(self, output_path: str) -> str:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        sections = []

        for title, image_name in self.image_names:
            image_path = os.path.join(self.input_dir, image_name)
            if os.path.exists(image_path):
                relative_path = os.path.relpath(image_path, output_dir).replace("\\", "/")
                content = f'<img src="{html.escape(relative_path)}" alt="{html.escape(title)}" class="chart-image">'
            else:
                content = (
                    '<div class="missing-file">'
                    f"未找到图像文件：{html.escape(image_name)}"
                    "</div>"
                )

            sections.append(
                '<div class="chart-card">'
                f"<h3>{html.escape(title)}</h3>"
                f"{content}"
                "</div>"
            )

        return "".join(sections)

    def build_warnings(self) -> str:
        if not self.warnings:
            return ""
        unique_warnings = []
        for item in self.warnings:
            if item not in unique_warnings:
                unique_warnings.append(item)

        warning_items = "".join(f"<li>{html.escape(item)}</li>" for item in unique_warnings)
        return (
            '<div class="section">'
            "<h2>数据完整性提示</h2>"
            f'<ul class="warning-list">{warning_items}</ul>'
            "</div>"
        )

    def build_html(
        self,
        comparison: Dict[str, Dict[str, Optional[float]]],
        report_data: Dict[str, Any],
        backtest_preview: pd.DataFrame,
        output_path: str,
    ) -> str:
        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary_text = self.build_summary(comparison)
        backtest_table = (
            html_table(backtest_preview)
            if not backtest_preview.empty
            else '<div class="empty-state">未找到可展示的回测明细示例。</div>'
        )

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF 两阶段优选策略回测结果汇总</title>
    <style>
        body {{
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f7fa;
        }}
        .header {{
            background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin-top: 10px;
            opacity: 0.9;
            font-size: 16px;
        }}
        .meta {{
            margin-top: 18px;
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .meta-item {{
            background: rgba(255, 255, 255, 0.12);
            padding: 10px 14px;
            border-radius: 8px;
            min-width: 220px;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border-left: 4px solid #3f51b5;
        }}
        .section h2 {{
            color: #1a237e;
            margin-top: 0;
            border-bottom: 2px solid #e8eaf6;
            padding-bottom: 10px;
        }}
        .param-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-top: 16px;
        }}
        .param-card {{
            background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
            border-radius: 8px;
            padding: 16px;
        }}
        .param-label {{
            font-size: 13px;
            color: #3f51b5;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .param-value {{
            margin-top: 8px;
            font-size: 20px;
            font-weight: 600;
            color: #1a237e;
            word-break: break-word;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }}
        .report-table th {{
            background-color: #3f51b5;
            color: white;
            text-align: left;
            padding: 14px;
            font-weight: 500;
        }}
        .report-table td {{
            padding: 14px;
            border-bottom: 1px solid #e8eaf6;
            vertical-align: top;
        }}
        .report-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .table-wrapper {{
            overflow-x: auto;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border-left: 3px solid #3f51b5;
            padding: 14px;
            border-radius: 0 4px 4px 0;
        }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 18px;
        }}
        .chart-card {{
            background: #fafbff;
            border: 1px solid #e8eaf6;
            border-radius: 8px;
            padding: 18px;
        }}
        .chart-card h3 {{
            margin-top: 0;
            color: #1a237e;
        }}
        .chart-image {{
            width: 100%;
            height: auto;
            border-radius: 6px;
            display: block;
        }}
        .missing-file {{
            background: #fff3e0;
            border-left: 4px solid #fb8c00;
            padding: 12px;
            border-radius: 0 4px 4px 0;
        }}
        .empty-state {{
            background: #f8f9fa;
            border-left: 3px solid #90a4ae;
            padding: 12px;
            border-radius: 0 4px 4px 0;
            color: #546e7a;
        }}
        .warning-list {{
            margin: 0;
            padding-left: 18px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e8eaf6;
            color: #757575;
            font-size: 14px;
        }}
        @media (max-width: 768px) {{
            body {{
                padding: 14px;
            }}
            .header {{
                padding: 22px;
            }}
            .header h1 {{
                font-size: 24px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ETF 两阶段优选策略回测结果汇总</h1>
        <div class="subtitle">传统机器学习策略与两阶段策略的回测结果对比报告</div>
        <div class="meta">
            <div class="meta-item"><strong>生成时间：</strong>{html.escape(generated_at)}</div>
            <div class="meta-item"><strong>数据目录：</strong>{html.escape(self.input_dir)}</div>
            <div class="meta-item"><strong>输出文件：</strong>{html.escape(os.path.abspath(output_path))}</div>
        </div>
    </div>

    <div class="section">
        <h2>关键参数</h2>
        <div class="param-grid">
            {self.build_parameter_items(report_data)}
        </div>
    </div>

    <div class="section">
        <h2>策略绩效指标对比</h2>
        <div class="table-wrapper">
            <table class="report-table">
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>传统模型策略</th>
                        <th>两阶段策略</th>
                    </tr>
                </thead>
                <tbody>
                    {self.build_metric_rows(comparison)}
                </tbody>
            </table>
        </div>
    </div>

    <div class="section">
        <h2>绩效摘要</h2>
        <div class="summary-box">{html.escape(summary_text)}</div>
    </div>

    <div class="section">
        <h2>图表展示</h2>
        <div class="charts-grid">
            {self.build_image_sections(output_path)}
        </div>
    </div>

    <div class="section">
        <h2>回测明细示例</h2>
        <p>以下表格展示 `backtest_results.csv` 的前 10 行，用于说明回测结果字段结构。</p>
        {backtest_table}
    </div>

    {self.build_warnings()}

    <div class="footer">
        <p>本报告由 ETF 两阶段优选研究框架自动生成，仅用于研究结果展示。</p>
        <p>生成时间：{html.escape(generated_at)}</p>
    </div>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 ETF 两阶段回测 HTML 汇总报告")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../results/traditional_models/random_forest/two_stage",
        help="两阶段回测结果目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/traditional_models/random_forest/two_stage/report_summary.html",
        help="HTML 汇总报告输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reporter = TwoStageReporter(args.input_dir)

    try:
        reporter.generate_report(args.output)
        print("[OK] 两阶段汇总报告生成成功")
    except Exception as exc:
        print(f"[FAIL] 两阶段汇总报告生成失败: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

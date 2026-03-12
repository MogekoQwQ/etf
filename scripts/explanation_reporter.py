"""
Generate readable HTML reports for stored explanation artifacts.
"""

import argparse
import datetime
import html
import json
import os
from typing import Any, Dict, Optional


class ExplanationReporter:
    """Generate single-date and summary HTML reports."""

    def __init__(self, explanations_dir: str):
        self.explanations_dir = explanations_dir

    def _report_href(self, report_path: str) -> str:
        """Return a relative href from the summary page to a daily report."""
        return os.path.relpath(report_path, self.explanations_dir).replace("\\", "/")

    def list_available_dates(self) -> list[str]:
        """Return dates that actually have explanation artifacts."""
        if not os.path.isdir(self.explanations_dir):
            return []

        available_dates = []
        for item in os.listdir(self.explanations_dir):
            item_path = os.path.join(self.explanations_dir, item)
            if not (os.path.isdir(item_path) and item.isdigit() and len(item) == 8):
                continue

            structured_path = os.path.join(item_path, "structured_explanation.json")
            full_response_path = os.path.join(item_path, "full_response.json")
            if os.path.exists(structured_path) or os.path.exists(full_response_path):
                available_dates.append(f"{item[:4]}-{item[4:6]}-{item[6:8]}")

        available_dates.sort()
        return available_dates

    def generate_report_for_date(self, date: str, output_path: Optional[str] = None) -> str:
        explanation_data = self._load_explanation_data(date)
        if not explanation_data:
            raise ValueError(f"未找到 {date} 的解释数据")

        html_content = self._generate_html_content(date, explanation_data)

        if output_path is None:
            date_dir = os.path.join(self.explanations_dir, date.replace("-", ""))
            os.makedirs(date_dir, exist_ok=True)
            output_path = os.path.join(date_dir, f"report_{date}.html")

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html_content)

        print(f"HTML报告已生成: {output_path}")
        return output_path

    def _load_explanation_data(self, date: str) -> Optional[Dict[str, Any]]:
        date_dir = os.path.join(self.explanations_dir, date.replace("-", ""))
        structured_path = os.path.join(date_dir, "structured_explanation.json")

        if os.path.exists(structured_path):
            with open(structured_path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        full_response_path = os.path.join(date_dir, "full_response.json")
        if os.path.exists(full_response_path):
            with open(full_response_path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        return None

    def _generate_html_content(self, date: str, data: Dict[str, Any]) -> str:
        rankings = data.get("rankings", [])
        summary = data.get("summary", {})
        etf_count = data.get("etf_count", len(rankings))
        top_etfs = rankings[:10]

        key_factors = summary.get("key_factors") or ["动量因子", "波动率", "流动性", "趋势强度"]
        key_factor_tags = "\n".join(
            f'            <span class="factor-tag">{html.escape(str(factor))}</span>'
            for factor in key_factors
        )

        rows = []
        for i, etf in enumerate(top_etfs, 1):
            code = html.escape(str(etf.get("code", "N/A")))
            score = float(etf.get("score", 0) or 0)
            explanation = html.escape(str(etf.get("explanation", "") or "无详细解释"))

            if score >= 0.8:
                score_class = "score-high"
            elif score >= 0.6:
                score_class = "score-medium"
            else:
                score_class = "score-low"

            rows.append(
                f"""                <tr>
                    <td>#{i}</td>
                    <td><strong>{code}</strong></td>
                    <td class="{score_class}">{score:.3f}</td>
                    <td><div class="explanation-box">{explanation}</div></td>
                </tr>"""
            )

        market_context = html.escape(str(summary.get("market_context", "LLM基于当日因子数据进行的市场环境分析。")))
        risk_considerations = html.escape(str(summary.get("risk_considerations", "基于因子数据的风险分析。")))
        risk_content = summary.get("risk_considerations") or (
            "1. 市场波动风险：ETF价格受市场波动影响。<br>"
            "2. 因子失效风险：量化因子可能阶段性失效。<br>"
            "3. 模型风险：LLM排序结果存在不确定性。<br>"
            "4. 流动性风险：部分ETF交易活跃度较低。"
        )
        generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF可解释性分析报告 - {date}</title>
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
        .metrics-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            margin: 10px;
            flex: 1;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #1a237e;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            color: #5c6bc0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .etf-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .etf-table th {{
            background-color: #3f51b5;
            color: white;
            text-align: left;
            padding: 15px;
            font-weight: 500;
        }}
        .etf-table td {{
            padding: 15px;
            border-bottom: 1px solid #e8eaf6;
        }}
        .etf-table tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .etf-table tr:hover {{
            background-color: #e8eaf6;
        }}
        .score-high {{ color: #4caf50; font-weight: bold; }}
        .score-medium {{ color: #ff9800; font-weight: bold; }}
        .score-low {{ color: #f44336; font-weight: bold; }}
        .explanation-box {{
            background-color: #f8f9fa;
            border-left: 3px solid #3f51b5;
            padding: 12px;
            border-radius: 0 4px 4px 0;
            font-size: 14px;
        }}
        .factor-tag {{
            display: inline-block;
            background: #e3f2fd;
            color: #1565c0;
            padding: 5px 12px;
            border-radius: 20px;
            margin: 3px;
            font-size: 13px;
        }}
        .risk-warning {{
            background: #ffebee;
            border-left: 4px solid #f44336;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 4px 4px 0;
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
            .metrics-container {{ flex-direction: column; }}
            .metric-card {{ margin: 10px 0; }}
            .etf-table {{ display: block; overflow-x: auto; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ETF可解释性分析报告</h1>
        <div class="subtitle">基于大语言模型的ETF排序解释 | 日期: {date}</div>
    </div>

    <div class="section">
        <h2>执行摘要</h2>
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-label">ETF总数</div>
                <div class="metric-value">{etf_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">报告生成时间</div>
                <div class="metric-value">{datetime.datetime.now().strftime('%H:%M')}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">数据源</div>
                <div class="metric-value">DeepSeek API</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>市场环境分析</h2>
        <p>{market_context}</p>
        <h3>关键因子关注度</h3>
        <div>
{key_factor_tags}
        </div>
    </div>

    <div class="section">
        <h2>ETF详细排名与解释</h2>
        <p>以下展示LLM排序前10名ETF及其解释。</p>
        <table class="etf-table">
            <thead>
                <tr>
                    <th>排名</th>
                    <th>ETF代码</th>
                    <th>LLM评分</th>
                    <th>解释分析</th>
                </tr>
            </thead>
            <tbody>
{chr(10).join(rows)}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>风险提示与注意事项</h2>
        <div class="risk-warning">
            <p><strong>重要风险提示：</strong></p>
            <p>{risk_content}</p>
        </div>
        <p><strong>LLM分析的风险考虑：</strong> {risk_considerations}</p>
    </div>

    <div class="footer">
        <p>报告生成系统：ETF两阶段优选研究框架 - LLM可解释性模块</p>
        <p>版本: 1.0 | 生成时间: {generated_at}</p>
    </div>
</body>
</html>"""

    def generate_summary_report(self, start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> str:
        date_dirs = []
        for date_str in self.list_available_dates():
            item = date_str.replace("-", "")
            item_path = os.path.join(self.explanations_dir, item)
            date_dirs.append((date_str, item_path))

        if start_date:
            date_dirs = [(d, p) for d, p in date_dirs if d >= start_date]
        if end_date:
            date_dirs = [(d, p) for d, p in date_dirs if d <= end_date]

        date_dirs.sort(key=lambda item: item[0])

        if not date_dirs:
            raise ValueError("指定日期范围内没有解释数据")

        cards = []
        for date_str, date_dir in date_dirs:
            report_path = os.path.join(date_dir, f"report_{date_str}.html")
            has_report = os.path.exists(report_path)
            report_link = (
                f'<a href="{self._report_href(report_path)}">查看HTML报告</a>'
                if has_report else "未生成"
            )

            cards.append(
                f"""
    <div class="date-card">
        <h3>{date_str}</h3>
        <p>数据目录: {html.escape(os.path.basename(date_dir))}</p>
        <p>报告: {report_link}</p>
        <p>文件列表: {html.escape(self._list_files_in_dir(date_dir))}</p>
    </div>"""
            )

        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>ETF可解释性分析汇总报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; padding: 20px; }}
        .header {{ background: #1a237e; color: white; padding: 20px; border-radius: 5px; }}
        .date-card {{
            background: white;
            border: 1px solid #ddd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #3f51b5;
        }}
        .date-card a {{ color: #1a237e; text-decoration: none; font-weight: bold; }}
        .date-card a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ETF可解释性分析汇总报告</h1>
        <p>日期范围: {start_date or '最早'} 至 {end_date or '最新'} | 总计: {len(date_dirs)} 个交易日</p>
    </div>
{''.join(cards)}
    <div style="margin-top: 30px; color: #666; font-size: 14px;">
        <p>生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>"""

        summary_path = os.path.join(self.explanations_dir, "summary_report.html")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(html_content)

        print(f"汇总报告已生成: {summary_path}")
        return summary_path

    def _list_files_in_dir(self, dir_path: str) -> str:
        try:
            files = os.listdir(dir_path)
            file_links = []
            for file_name in files:
                if file_name.endswith((".json", ".txt", ".csv", ".html")):
                    file_path = os.path.join(dir_path, file_name)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path) / 1024
                        file_links.append(f"{file_name} ({size:.1f}KB)")
            return ", ".join(file_links[:5]) + ("..." if len(file_links) > 5 else "")
        except Exception:
            return "无法访问"


def main():
    parser = argparse.ArgumentParser(description="生成ETF可解释性分析报告")
    parser.add_argument("--date", type=str, help="生成指定日期的报告 (格式: YYYY-MM-DD)")
    parser.add_argument("--start-date", type=str, help="汇总报告开始日期")
    parser.add_argument("--end-date", type=str, help="汇总报告结束日期")
    parser.add_argument(
        "--explanations-dir",
        type=str,
        default="../results/traditional_models/random_forest/two_stage/llm_logs/explanations",
        help="解释内容目录路径",
    )
    parser.add_argument("--output", type=str, help="输出文件路径")

    args = parser.parse_args()
    reporter = ExplanationReporter(args.explanations_dir)

    try:
        if args.date:
            output_path = reporter.generate_report_for_date(args.date, args.output)
            print(f"[OK] 单日报告生成成功: {output_path}")
        else:
            output_path = reporter.generate_summary_report(args.start_date, args.end_date)
            print(f"[OK] 汇总报告生成成功: {output_path}")
    except Exception as exc:
        print(f"[FAIL] 报告生成失败: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()

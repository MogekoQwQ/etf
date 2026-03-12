"""
解释内容处理工具模块
提供结构化存储、读取和验证解释内容的功能
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any
import datetime


class ExplanationStorage:
    """
    解释内容存储管理类
    负责解释内容的保存、组织和检索
    """

    def __init__(self, base_dir: str):
        """
        初始化存储管理器

        Args:
            base_dir: 基础目录路径，如 "../results/traditional_models/random_forest/two_stage/llm_logs"
        """
        self.base_dir = base_dir
        self.explanations_dir = os.path.join(base_dir, "explanations")

    def save_explanation(self, date: str, result: Dict[str, Any],
                         etf_data: pd.DataFrame, enable_explanations: bool = True) -> str:
        """
        保存解释内容到结构化文件

        Args:
            date: 交易日字符串 (YYYY-MM-DD)
            result: 完整的API响应结果（JSON解析后的字典）
            etf_data: 包含llm_score和llm_explanation的DataFrame
            enable_explanations: 是否启用解释功能

        Returns:
            保存的目录路径
        """
        if not enable_explanations:
            return ""

        # 创建日期目录
        date_dir = os.path.join(self.explanations_dir, date.replace("-", ""))
        os.makedirs(date_dir, exist_ok=True)

        # 1. 保存完整API响应
        full_response_path = os.path.join(date_dir, "full_response.json")
        with open(full_response_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 2. 保存摘要文本
        summary_path = os.path.join(date_dir, "summary.txt")
        self._save_summary_text(date, result, summary_path)

        # 3. 保存ETF详细解释（CSV格式）
        details_path = os.path.join(date_dir, "etf_details.csv")
        self._save_etf_details(etf_data, details_path)

        # 4. 保存结构化解释（JSON格式）
        structured_path = os.path.join(date_dir, "structured_explanation.json")
        self._save_structured_explanation(date, result, etf_data, structured_path)

        # 5. 更新索引
        self._update_index(date, date_dir, etf_data)

        print(f"解释内容已保存到: {date_dir}")
        return date_dir

    def _save_summary_text(self, date: str, result: Dict[str, Any], summary_path: str):
        """保存摘要文本文件"""
        summary_content = []

        # 添加日期标题
        summary_content.append(f"=== ETF可解释性分析报告 ===\n")
        summary_content.append(f"日期: {date}\n")

        # 添加市场分析
        if "summary" in result and "market_context" in result["summary"]:
            summary_content.append(f"\n[市场分析]")
            summary_content.append(f"{result['summary']['market_context']}\n")

        # 添加关键因子
        if "summary" in result and "key_factors" in result["summary"]:
            summary_content.append(f"\n[关键因子]")
            for factor in result["summary"]["key_factors"]:
                summary_content.append(f"- {factor}")

        # 添加风险考虑
        if "summary" in result and "risk_considerations" in result["summary"]:
            summary_content.append(f"\n[风险考虑]")
            summary_content.append(f"{result['summary']['risk_considerations']}\n")

        # 添加前3名ETF分析
        if "rankings" in result:
            summary_content.append(f"\n[前3名ETF分析]")
            for i, item in enumerate(result["rankings"][:3], 1):
                code = item.get("code", "")
                score = item.get("score", 0)
                explanation = item.get("explanation", "")
                summary_content.append(f"{i}. {code} (评分: {score:.3f})")
                if explanation:
                    summary_content.append(f"   解释: {explanation}")
                summary_content.append("")

        # 写入文件
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_content))

    def _save_etf_details(self, etf_data: pd.DataFrame, details_path: str):
        """保存ETF详细解释到CSV文件"""
        # 选择要保存的列
        cols_to_save = ["code", "name", "llm_score"]
        if "llm_explanation" in etf_data.columns:
            cols_to_save.append("llm_explanation")

        # 保存到CSV
        etf_data[cols_to_save].to_csv(details_path, index=False, encoding="utf-8-sig")

    def _save_structured_explanation(self, date: str, result: Dict[str, Any],
                                     etf_data: pd.DataFrame, structured_path: str):
        """保存结构化解释到JSON文件"""
        structured_data = {
            "date": date,
            "generation_time": datetime.datetime.now().isoformat(),
            "etf_count": len(etf_data),
            "rankings": []
        }

        # 添加排名数据
        for _, row in etf_data.iterrows():
            ranking = {
                "code": row["code"],
                "score": float(row["llm_score"]),
                "explanation": row.get("llm_explanation", "") if "llm_explanation" in etf_data.columns else ""
            }
            structured_data["rankings"].append(ranking)

        # 添加摘要信息
        if "summary" in result:
            structured_data["summary"] = result["summary"]

        # 保存到文件
        with open(structured_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)

    def _update_index(self, date: str, date_dir: str, etf_data: pd.DataFrame):
        """更新索引文件"""
        index_path = os.path.join(self.explanations_dir, "index.csv")

        # 创建索引行
        index_row = {
            "date": date,
            "date_dir": os.path.basename(date_dir),
            "etf_count": len(etf_data),
            "has_explanations": "llm_explanation" in etf_data.columns,
            "full_response_path": os.path.join(date_dir, "full_response.json"),
            "summary_path": os.path.join(date_dir, "summary.txt"),
            "details_path": os.path.join(date_dir, "etf_details.csv"),
            "structured_path": os.path.join(date_dir, "structured_explanation.json")
        }

        # 读取或创建索引
        if os.path.exists(index_path):
            index_df = pd.read_csv(index_path)
        else:
            index_df = pd.DataFrame(columns=index_row.keys())

        if not index_df.empty and "date" in index_df.columns:
            index_df = index_df[index_df["date"] != date]

        # 添加新行
        index_df = pd.concat([index_df, pd.DataFrame([index_row])], ignore_index=True)
        index_df.to_csv(index_path, index=False, encoding="utf-8-sig")

    def load_explanation(self, date: str) -> Optional[Dict[str, Any]]:
        """
        加载指定日期的解释内容

        Args:
            date: 交易日字符串 (YYYY-MM-DD)

        Returns:
            结构化解释数据，如果不存在则返回None
        """
        date_dir = os.path.join(self.explanations_dir, date.replace("-", ""))
        structured_path = os.path.join(date_dir, "structured_explanation.json")

        if not os.path.exists(structured_path):
            return None

        with open(structured_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_dates_with_explanations(self) -> List[str]:
        """
        列出所有有解释内容的日期

        Returns:
            日期字符串列表
        """
        index_path = os.path.join(self.explanations_dir, "index.csv")
        if not os.path.exists(index_path):
            return []

        index_df = pd.read_csv(index_path)
        dates = []
        for _, row in index_df.iterrows():
            date = row.get("date")
            structured_path = row.get("structured_path")
            if isinstance(date, str) and isinstance(structured_path, str) and os.path.exists(structured_path):
                dates.append(date)
        return dates


def extract_key_insights(result: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
    """
    从解释结果中提取关键洞察

    Args:
        result: API响应结果
        top_n: 提取前N名ETF的洞察

    Returns:
        关键洞察字典
    """
    insights = {
        "top_etfs": [],
        "common_strengths": [],
        "common_weaknesses": [],
        "factor_importance": {},
        "risk_patterns": []
    }

    if "rankings" not in result:
        return insights

    # 提取前N名ETF的洞察
    top_rankings = result["rankings"][:top_n]
    explanations = [r.get("explanation", "") for r in top_rankings if "explanation" in r]

    # 简单的关键词提取（实际项目中可以使用更复杂的NLP技术）
    strength_keywords = ["强劲", "优势", "优秀", "良好", "稳定", "高", "强", "积极"]
    weakness_keywords = ["偏弱", "不足", "风险", "波动", "低", "弱", "负面", "谨慎"]

    for explanation in explanations:
        if not explanation:
            continue

        # 检查优势和劣势关键词
        for keyword in strength_keywords:
            if keyword in explanation:
                if keyword not in insights["common_strengths"]:
                    insights["common_strengths"].append(keyword)

        for keyword in weakness_keywords:
            if keyword in explanation:
                if keyword not in insights["common_weaknesses"]:
                    insights["common_weaknesses"].append(keyword)

    # 添加排名前5的ETF
    for i, ranking in enumerate(top_rankings[:5], 1):
        insights["top_etfs"].append({
            "rank": i,
            "code": ranking.get("code", ""),
            "score": ranking.get("score", 0),
            "has_explanation": "explanation" in ranking and bool(ranking["explanation"])
        })

    # 从摘要中提取因子重要性
    if "summary" in result and "key_factors" in result["summary"]:
        insights["factor_importance"] = {
            "key_factors": result["summary"]["key_factors"]
        }

    return insights


def validate_explanation_format(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证解释格式是否符合预期

    Args:
        result: API响应结果

    Returns:
        验证结果字典
    """
    validation = {
        "is_valid": False,
        "errors": [],
        "warnings": [],
        "stats": {}
    }

    # 检查必需字段
    if "rankings" not in result:
        validation["errors"].append("缺少必需字段: rankings")
        return validation

    rankings = result["rankings"]
    if not isinstance(rankings, list):
        validation["errors"].append("rankings 必须是列表")
        return validation

    # 统计信息
    validation["stats"]["etf_count"] = len(rankings)

    # 检查每个排名的格式
    explanation_count = 0
    for i, item in enumerate(rankings):
        if not isinstance(item, dict):
            validation["errors"].append(f"排名 {i} 不是字典")
            continue

        # 检查必需字段
        if "code" not in item:
            validation["errors"].append(f"排名 {i} 缺少 code 字段")
        if "score" not in item:
            validation["errors"].append(f"排名 {i} 缺少 score 字段")

        # 检查解释字段
        if "explanation" in item:
            explanation_count += 1
            explanation = item["explanation"]
            if not isinstance(explanation, str):
                validation["warnings"].append(f"排名 {i} 的 explanation 不是字符串")
            elif len(explanation.strip()) == 0:
                validation["warnings"].append(f"排名 {i} 的 explanation 为空")
            elif len(explanation) > 500:  # 检查长度
                validation["warnings"].append(f"排名 {i} 的 explanation 过长 ({len(explanation)} 字符)")

    validation["stats"]["explanation_count"] = explanation_count
    validation["stats"]["explanation_coverage"] = explanation_count / len(rankings) if rankings else 0

    # 检查摘要字段
    if "summary" in result:
        summary = result["summary"]
        if not isinstance(summary, dict):
            validation["warnings"].append("summary 字段不是字典")
        else:
            # 检查摘要中的关键字段
            for field in ["market_context", "key_factors", "risk_considerations"]:
                if field not in summary:
                    validation["warnings"].append(f"summary 中缺少 {field} 字段")

    # 如果没有错误，标记为有效
    if not validation["errors"]:
        validation["is_valid"] = True

    return validation


# 示例用法
if __name__ == "__main__":
    # 创建存储管理器
    storage = ExplanationStorage("../results/traditional_models/random_forest/two_stage/llm_logs")

    # 示例：加载现有解释
    dates = storage.list_dates_with_explanations()
    print(f"有解释内容的日期: {dates}")

    if dates:
        # 加载第一个日期的解释
        explanation = storage.load_explanation(dates[0])
        if explanation:
            print(f"加载到 {explanation['date']} 的解释，包含 {explanation['etf_count']} 只ETF")

            # 提取关键洞察
            insights = extract_key_insights(explanation)
            print(f"关键洞察: {insights}")

            # 验证格式
            validation = validate_explanation_format(explanation)
            print(f"验证结果: {validation}")

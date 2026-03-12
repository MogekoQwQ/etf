"""
解释功能配置管理
提供成本优先的配置管理，确保API成本可控
"""

import json
import os
from typing import Dict, Any, Optional


class ExplanationConfig:
    """
    解释功能配置类
    强调成本控制，默认关闭解释功能
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config_path = config_path

        # 默认配置（成本优先）
        self.default_config = {
            # 核心控制（成本优先）
            "enable": False,  # 默认关闭，需要时手动开启
            "enable_by_default": False,  # 永远默认关闭，防止意外成本
            "explanation_mode": "sampled_display_only",
            "explanation_date": None,
            "explanation_sample_size": 3,

            # 内容控制
            "explanation_level": "standard",  # standard/brief
            "max_explanation_length": 300,  # 每只ETF最多300字符
            "summary_max_length": 500,  # 摘要最多500字符

            # 选择性生成（成本优化）
            "score_threshold": 0.7,  # 只对评分>0.7的ETF生成详细解释
            "top_n_detailed": 10,  # 最多为前10名生成详细解释
            "skip_low_score": True,  # 跳过低分ETF

            # 格式控制
            "save_json": True,  # 保存JSON格式（程序处理）
            "generate_html": True,  # 生成HTML报告（用户查看）
            "save_text_summary": True,  # 保存文本摘要

            # 成本跟踪
            "enable_cost_tracking": True,
            "max_tokens_per_call": 8000,  # 每次API调用token限制
            "estimated_cost_per_call": 0.002,  # 估计每次调用成本（美元）
        }

        # 加载或使用默认配置
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
        else:
            self.config = self.default_config.copy()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """从文件加载配置"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)

            # 合并配置，确保所有字段都有值
            config = self.default_config.copy()
            config.update(loaded_config)
            return config
        except Exception as e:
            print(f"加载配置文件失败 {config_path}: {e}，使用默认配置")
            return self.default_config.copy()

    def save_config(self, config_path: Optional[str] = None):
        """保存配置到文件"""
        save_path = config_path or self.config_path
        if not save_path:
            print("未指定配置文件路径，无法保存")
            return

        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到 {save_path}")
        except Exception as e:
            print(f"保存配置文件失败 {save_path}: {e}")

    def get(self, key: str, default=None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value

    def validate_config(self) -> Dict[str, Any]:
        """验证配置的有效性"""
        validation = {
            "is_valid": True,
            "warnings": [],
            "cost_estimates": {}
        }

        # 检查成本相关配置
        if self.get("enable") and not self.get("enable_by_default"):
            validation["warnings"].append("解释功能已启用，但默认关闭，符合成本控制原则")

        # 检查长度限制
        max_len = self.get("max_explanation_length")
        if max_len > 500:
            validation["warnings"].append(f"解释长度限制较高 ({max_len}字符)，可能增加API成本")

        # 计算估计成本
        if self.get("enable_cost_tracking"):
            tokens_per_etf = max_len / 4  # 粗略估计：每字符约0.25个token
            estimated_tokens = tokens_per_etf * self.get("top_n_detailed", 10)
            estimated_cost = estimated_tokens / 1000 * 0.002  # 粗略成本估计

            validation["cost_estimates"] = {
                "tokens_per_etf": round(tokens_per_etf),
                "estimated_total_tokens": round(estimated_tokens),
                "estimated_cost_per_call": round(estimated_cost, 6),
                "max_tokens_per_call": self.get("max_tokens_per_call")
            }

            if estimated_tokens > self.get("max_tokens_per_call"):
                validation["is_valid"] = False
                validation["warnings"].append(
                    f"估计token使用量 ({estimated_tokens}) 超过限制 ({self.get('max_tokens_per_call')})"
                )

        return validation

    def generate_cost_report(self, call_count: int = 1) -> Dict[str, Any]:
        """
        生成成本报告

        Args:
            call_count: API调用次数

        Returns:
            成本报告
        """
        if not self.get("enable_cost_tracking"):
            return {"message": "成本跟踪未启用"}

        validation = self.validate_config()
        if not validation["cost_estimates"]:
            return {"message": "无法生成成本估计"}

        cost_per_call = validation["cost_estimates"].get("estimated_cost_per_call", 0)
        total_cost = cost_per_call * call_count

        return {
            "cost_per_call_usd": round(cost_per_call, 6),
            "estimated_total_cost_usd": round(total_cost, 6),
            "call_count": call_count,
            "config_summary": {
                "enable": self.get("enable"),
                "explanation_mode": self.get("explanation_mode"),
                "explanation_date": self.get("explanation_date"),
                "explanation_sample_size": self.get("explanation_sample_size"),
                "top_n_detailed": self.get("top_n_detailed"),
                "max_explanation_length": self.get("max_explanation_length"),
                "max_tokens_per_call": self.get("max_tokens_per_call"),
            }
        }

    def apply_to_prompt(self, prompt_template: str) -> str:
        """
        将配置应用到prompt模板

        Args:
            prompt_template: 原始prompt模板

        Returns:
            应用配置后的prompt模板
        """
        if not self.get("enable"):
            return prompt_template

        # 这里可以添加根据配置调整prompt的逻辑
        # 例如，添加长度限制提示等
        modified_prompt = prompt_template

        # 添加成本控制提示
        if self.get("max_explanation_length"):
            limit_msg = f"\n## 特别提醒（成本控制）\n请确保每只ETF的解释不超过{self.get('max_explanation_length')}字符，整体摘要不超过{self.get('summary_max_length')}字符。"
            modified_prompt = modified_prompt.replace("## 特别要求（成本控制）", f"## 特别要求（成本控制）{limit_msg}")

        return modified_prompt


# 全局配置实例
_default_config = None


def get_global_config(config_path: Optional[str] = None) -> ExplanationConfig:
    """获取全局配置实例"""
    global _default_config
    if _default_config is None:
        _default_config = ExplanationConfig(config_path)
    return _default_config


# 示例用法
if __name__ == "__main__":
    # 创建配置实例
    config = ExplanationConfig()

    # 打印默认配置
    print("默认配置:")
    print(json.dumps(config.config, indent=2, ensure_ascii=False))

    # 验证配置
    validation = config.validate_config()
    print("\n配置验证:")
    print(f"是否有效: {validation['is_valid']}")
    if validation['warnings']:
        print("警告:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    if validation['cost_estimates']:
        print("成本估计:")
        for key, value in validation['cost_estimates'].items():
            print(f"  {key}: {value}")

    # 生成成本报告
    cost_report = config.generate_cost_report(call_count=10)
    print("\n成本报告 (10次调用):")
    print(json.dumps(cost_report, indent=2, ensure_ascii=False))

    # 保存配置示例
    config.save_config("explanation_config.json")

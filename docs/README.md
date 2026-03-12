# ETF Project

## Environment
- Python 3.10

### 环境设置
```bash
# 1. 克隆项目
git clone git@github.com:MogekoQwQ/etf.git
cd etf-main

# 2. 创建虚拟环境
py -3.10 -m venv .venv

# 3. 激活虚拟环境
# PowerShell:
.\.venv\Scripts\Activate.ps1

# 4. 安装依赖
pip install -r requirements.txt

# 5. 进行环境诊断
python scripts/dev_tools/check_env.py

# 6. 设置DeepSeek API密钥（如需在线LLM排序）
# PowerShell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

### 快速开始
```bash
# 进入脚本目录
cd scripts

# 全流程执行
python run_pipeline.py

# 跳过数据下载（当原始数据已存在时）
python run_pipeline.py --skip-download

# 跳过LLM，仅执行传统模型回测链路
python run_pipeline.py --skip-download --skip-factors --skip-traditional --skip-llm

# 使用本地mock排序验证两阶段流程
python run_pipeline.py --skip-download --skip-factors --skip-traditional --mock-llm

# 启用解释功能
python run_pipeline.py --enable-explanations

# 启用解释功能并显式设置 LLM 超时（秒）
python run_pipeline.py --skip-download --skip-factors --skip-traditional --enable-explanations --llm-timeout 180

# 自定义参数示例（轻量测试）
python run_pipeline.py --skip-download --skip-factors --skip-traditional --enable-explanations --rebalancing-freq M --top-n-first 10 --top-n-final 3

# 自定义参数示例（轻量测试 + 显式超时）
python run_pipeline.py --skip-download --skip-factors --skip-traditional --enable-explanations --rebalancing-freq M --top-n-first 10 --top-n-final 3 --llm-timeout 180
```

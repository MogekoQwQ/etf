#!/usr/bin/env python
"""Simple environment diagnostics for local development."""

from __future__ import annotations

import os
import platform
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VENV_PATH = os.path.join(PROJECT_ROOT, ".venv")
IN_VENV = (hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix) or bool(os.environ.get("VIRTUAL_ENV"))

print("=" * 60)
print("环境诊断报告")
print("=" * 60)

print("\n1. 系统信息:")
print(f"  操作系统: {platform.platform()}")
print(f"  架构: {platform.architecture()}")
print(f"  当前目录: {os.getcwd()}")
print(f"  项目根目录: {PROJECT_ROOT}")

print("\n2. Python 信息:")
print(f"  Python 版本: {sys.version}")
print(f"  Python 可执行文件: {sys.executable}")
print(f"  是否位于虚拟环境: {IN_VENV}")

print("\n3. 虚拟环境检查:")
if os.path.exists(VENV_PATH):
    print(f"  [OK] 虚拟环境目录存在: {VENV_PATH}")
    python_exe = os.path.join(VENV_PATH, "Scripts", "python.exe")
    if os.path.exists(python_exe):
        print(f"  [OK] 虚拟环境 Python 存在: {python_exe}")
        try:
            result = subprocess.run([python_exe, "--version"], capture_output=True, text=True, shell=False)
            print(f"  虚拟环境 Python 版本: {(result.stdout or result.stderr).strip()}")
        except Exception as exc:
            print(f"  [FAIL] 无法获取虚拟环境版本: {exc}")
    else:
        print("  [FAIL] 虚拟环境 Python 不存在")
else:
    print("  [FAIL] 虚拟环境目录不存在")

print("\n4. 关键依赖检查:")
for module_name, display_name in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
]:
    try:
        module = __import__(module_name)
        print(f"  [OK] {display_name}: {getattr(module, '__version__', 'unknown')}")
    except ImportError as exc:
        print(f"  [FAIL] {display_name}: {exc}")

print("\n5. 环境变量检查:")
for env_var in ["PATH", "VIRTUAL_ENV", "PYTHONPATH"]:
    value = os.environ.get(env_var, "<未设置>")
    if env_var == "PATH" and len(value) > 200:
        value = value[:200] + "...[截断]"
    print(f"  {env_var}: {value}")

print("\n6. 建议:")
print("  1. 建议使用 Python 3.10+")
print("  2. 在项目根目录运行: py -3.10 -m venv .venv")
print("  3. 激活环境后执行: pip install -r requirements.txt")
print("  4. 当前辅助脚本已移至 scripts/dev_tools/")

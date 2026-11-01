#!/usr/bin/env python3
"""Captioner 生成多个 caption；Offline QA Solver 逐 caption 批量答题。

用法与输入输出见 README.md。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_qa.common import run_stage

if __name__ == '__main__':
    run_stage(6)

#!/usr/bin/env python3
"""Weak Verifier 初筛，未通过再升级 Teacher；保留逐模型证据和争议题。

用法与输入输出见 README.md。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_qa.common import run_stage

if __name__ == '__main__':
    run_stage(5)

#!/usr/bin/env python3
"""Offline QA Maker：直接从原始媒体生成 QA。

用法与输入输出见 README.md。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_qa.common import run_stage

if __name__ == '__main__':
    run_stage(2)

#!/usr/bin/env python3
"""校验 QA 格式、去重，并用固定 seed 打乱选项。

用法与输入输出见 README.md。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from offline_qa.common import run_stage

if __name__ == '__main__':
    run_stage(3)

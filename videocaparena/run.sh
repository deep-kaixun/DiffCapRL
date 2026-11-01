#!/usr/bin/env bash
set -euo pipefail
ARENA_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ARENA_REPO_ROOT/videocaparena/benchmark.py" "$@"

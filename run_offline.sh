#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo '请先按README完成联网准备；离线启动不会自动安装依赖。' >&2
  exit 1
fi
export MODEL_BACKEND=local
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765

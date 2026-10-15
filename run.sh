#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.lock
fi
if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
fi
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8765

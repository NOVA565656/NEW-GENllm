#!/usr/bin/env bash
set -euo pipefail

echo "=== Agent Bootstrap ==="

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade "openai>=1.0" python-dotenv "amcp-agent[telegram]==0.13.0"

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo "Fill in the keys, then run ./start.sh"
    exit 0
fi
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== Installing agent dependencies ==="

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
    "amcp-agent[telegram]==0.13.0" \
    "openai>=1.0" \
    "python-dotenv>=1.0"

echo "=== Applying AMCP tool-loop integration ==="
python patch_amcp.py

if [[ ! -f .env ]]; then
    cp .env.example .env
    chmod 600 .env
    echo "Created .env. Add your provider and Telegram settings before starting."
else
    chmod 600 .env
fi

echo "=== Bootstrap complete ==="
echo "Edit .env, then run ./start.sh"
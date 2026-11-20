#!/usr/bin/env bash
set -euo pipefail

echo "=== Agent Bootstrap ==="

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --upgrade \
    "openai>=1.0" \
    "python-dotenv>=1.0" \
    "amcp-agent[telegram]==0.13.0"

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Created .env from .env.example"
    echo "Fill in provider keys and model IDs, then run ./bootstrap.sh again."
    exit 0
fi

set -a
source .env
set +a

required=(
    TELEGRAM_BOT_TOKEN
    TELEGRAM_ADMIN_USER_ID
    GROQ_API_KEY
    GEMINI_API_KEY
    NVIDIA_API_KEY
    NVIDIA_MODEL
    CLOUDFLARE_API_TOKEN
    CLOUDFLARE_ACCOUNT_ID
    CLOUDFLARE_MODEL
)

missing=0
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required setting: ${name}" >&2
        missing=1
    fi
done

if [[ "$missing" -ne 0 ]]; then
    exit 1
fi

echo "Bootstrap configuration is complete."
echo "Run ./start.sh to start AMCP."
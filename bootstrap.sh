#!/usr/bin/env bash
set -euo pipefail

echo "=== Agent Bootstrap ==="

# 1. Python env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2. Core tools
pip install freellmpool openai python-dotenv amcp-agent[telegram]

# 3. Load keys from .env
if [ ! -f .env ]; then
  cp .env.example .env
  echo ">>> Edit .env with your API keys, then re-run."
  exit 0
fi
set -a; source .env; set +a

# 4. Point freellmpool at our config
export FREELLMPOOL_CONFIG="$(pwd)/config/freellmpool.toml"

# 5. Validate
echo "--- Capacity check ---"
freellmpool capacity status --target 4 || true

# 6. AMCP config: single OpenAI-compatible endpoint
cat > config/amcp.env <<EOF
AMCP_BASE_URL=http://localhost:8080/v1
AMCP_API_KEY=anything
AMCP_CHAT_MODEL=auto
AMCP_TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
AMCP_TELEGRAM_ADMIN_USERS=${TELEGRAM_ADMIN_USER_ID}
EOF

echo "=== Done. Run ./start.sh ==="
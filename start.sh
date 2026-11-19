#!/usr/bin/env bash
set -euo pipefail

echo "=== Starting Agent ==="

# 1. Activate environment
source .venv/bin/activate
set -a; source .env; set +a
export FREELLMPOOL_CONFIG="$(pwd)/config/freellmpool.toml"

# 2. Validate Telegram User ID
if [ -z "${TELEGRAM_ADMIN_USER_ID:-}" ]; then
    echo "ERROR: TELEGRAM_ADMIN_USER_ID is empty in your .env file."
    exit 1
fi

# 3. Kill any process already using port 8080
echo "Checking port 8080..."
if command -v fuser >/dev/null 2>&1; then
    fuser -k 8080/tcp 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
    lsof -t -i:8080 | xargs -r kill -9 2>/dev/null || true
fi
sleep 1

# 4. Start the free-provider pool
echo "Starting freellmpool on port 8080..."
freellmpool proxy --port 8080 &
POOL_PID=$!
sleep 3

# 5. Start the Telegram bot with Workarounds
echo "Starting Telegram bot for User ID: ${TELEGRAM_ADMIN_USER_ID}..."

# WORKAROUND A: Force the ID into the environment variable that FastMCP reads
export FASTMCP_SERVER_AUTH_ALLOW_USER="${TELEGRAM_ADMIN_USER_ID}"
export AMCP_TELEGRAM_ALLOWED_USERS="${TELEGRAM_ADMIN_USER_ID}"

# WORKAROUND B: Generate the config.toml file that AMCP looks for
mkdir -p ~/.config/amcp
cat > ~/.config/amcp/config.toml <<EOF
[llm]
base_url = "http://localhost:8080/v1"
model = "auto"
api_key = "anything"

[telegram]
token = "${TELEGRAM_BOT_TOKEN}"
allowed_users = ["${TELEGRAM_ADMIN_USER_ID}"]
EOF

# Start the bot
amcp telegram start --allow-user "${TELEGRAM_ADMIN_USER_ID}" || {
    echo "Bot stopped. Shutting down freellmpool..."
    kill $POOL_PID
    exit 1
}
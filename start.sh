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

# 5. Configure Telegram bot
echo "Configuring Telegram bot for User ID: ${TELEGRAM_ADMIN_USER_ID}..."

# Set environment variables expected by AMCP's underlying framework
export AMCP_TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}"
export AMCP_TELEGRAM_ALLOWED_USERS="${TELEGRAM_ADMIN_USER_ID}"
export AMCP_TELEGRAM_ADMIN_USERS="${TELEGRAM_ADMIN_USER_ID}"

# Write the config file exactly as AMCP expects it
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

# 6. Start the bot using the correct plural flag
echo "Starting Telegram bot..."
amcp telegram start --allowed-users "${TELEGRAM_ADMIN_USER_ID}" || {
    echo "Bot stopped. Shutting down freellmpool..."
    kill $POOL_PID
    exit 1
}
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
    echo "Get your ID from @userinfobot on Telegram, then set it in .env"
    exit 1
fi

if ! [[ "${TELEGRAM_ADMIN_USER_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TELEGRAM_ADMIN_USER_ID must be a number (e.g., 123456789)."
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

# Wait for the proxy to be ready
sleep 3

# 5. Start the Telegram bot with the validated User ID
echo "Starting Telegram bot for User ID: ${TELEGRAM_ADMIN_USER_ID}..."
set -a; source config/amcp.env; set +a

# Pass the flag explicitly to avoid empty arguments
amcp telegram start --allow-user "${TELEGRAM_ADMIN_USER_ID}" || {
    echo "Bot stopped. Shutting down freellmpool..."
    kill $POOL_PID
    exit 1
}
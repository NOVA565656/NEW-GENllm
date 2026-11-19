#!/usr/bin/env bash
set -euo pipefail

echo "=== Starting Agent ==="

# 1. Activate environment
source .venv/bin/activate
set -a; source .env; set +a
export FREELLMPOOL_CONFIG="$(pwd)/config/freellmpool.toml"

# 2. Kill any process already using port 8080 (fixes the Errno 98 error)
echo "Checking port 8080..."
if command -v fuser >/dev/null 2>&1; then
    fuser -k 8080/tcp 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
    lsof -t -i:8080 | xargs -r kill -9 2>/dev/null || true
fi
sleep 1

# 3. Start the free-provider pool
echo "Starting freellmpool on port 8080..."
freellmpool proxy --port 8080 &
POOL_PID=$!

# Give the proxy a moment to bind
sleep 2

# 4. Start the Telegram bot with your User ID
echo "Starting Telegram bot..."
set -a; source config/amcp.env; set +a

# Ensure the admin ID is set
if [ -z "${TELEGRAM_ADMIN_USER_ID:-}" ]; then
    echo "ERROR: TELEGRAM_ADMIN_USER_ID is not set in your .env file."
    kill $POOL_PID
    exit 1
fi

# Start AMCP with the allowed user flag
amcp telegram start --allow-user "${TELEGRAM_ADMIN_USER_ID}" || {
    echo "Bot stopped. Shutting down freellmpool..."
    kill $POOL_PID
    exit 1
}
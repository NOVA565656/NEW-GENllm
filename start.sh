#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
set -a; source .env; set +a
export FREELLMPOOL_CONFIG="$(pwd)/config/freellmpool.toml"

# Start the free-provider pool
freellmpool proxy --port 8080 &

# Give the proxy a moment to bind
sleep 2

# Start the Telegram bot
set -a; source config/amcp.env; set +a
amcp telegram start
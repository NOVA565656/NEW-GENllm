#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
set -a
source .env
set +a

: "${TELEGRAM_BOT_TOKEN:?Set TELEGRAM_BOT_TOKEN in .env}"
: "${TELEGRAM_ADMIN_USER_ID:?Set TELEGRAM_ADMIN_USER_ID in .env}"
: "${GROQ_API_KEY:?Set GROQ_API_KEY in .env}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in .env}"
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY in .env}"
: "${CLOUDFLARE_API_TOKEN:?Set CLOUDFLARE_API_TOKEN in .env}"
: "${CLOUDFLARE_ACCOUNT_ID:?Set CLOUDFLARE_ACCOUNT_ID in .env}"
: "${NVIDIA_MODEL:?Set NVIDIA_MODEL in .env}"
: "${CLOUDFLARE_MODEL:?Set CLOUDFLARE_MODEL in .env}"

mkdir -p ~/.config/amcp
chmod 700 ~/.config/amcp

cat > ~/.config/amcp/config.toml <<EOF
[chat]
active_provider = "groq"
base_url = "https://api.groq.com/openai/v1"
model = "${GROQ_MODEL:-llama-3.3-70b-versatile}"
api_key = "${GROQ_API_KEY}"
api_type = "openai"
tool_loop_limit = 300
bash_tool_limit = 100
default_max_lines = 400
mcp_tools_enabled = true
write_tool_enabled = true
edit_tool_enabled = true
default_agent = "coder"

[chat.providers.groq]
base_url = "https://api.groq.com/openai/v1"
model = "${GROQ_MODEL:-llama-3.3-70b-versatile}"
api_key = "${GROQ_API_KEY}"
api_type = "openai"

[chat.providers.gemini]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model = "${GEMINI_MODEL:-gemini-2.5-flash}"
api_key = "${GEMINI_API_KEY}"
api_type = "openai"

[chat.providers.nvidia]
base_url = "https://integrate.api.nvidia.com/v1"
model = "${NVIDIA_MODEL}"
api_key = "${NVIDIA_API_KEY}"
api_type = "openai"

[chat.providers.cloudflare]
base_url = "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1"
model = "${CLOUDFLARE_MODEL}"
api_key = "${CLOUDFLARE_API_TOKEN}"
api_type = "openai"

[chat.providers.venice]
base_url = "https://api.venice.ai/api/v1"
model = "${VENICE_MODEL:-cognitivecomputations/dolphin-mistral-24b-venice-edition}"
api_key = "${VENICE_API_KEY:-}"
api_type = "openai"

[telegram]
enabled = true
bot_token = "${TELEGRAM_BOT_TOKEN}"
allowed_users = ["${TELEGRAM_ADMIN_USER_ID}"]
admin_users = ["${TELEGRAM_ADMIN_USER_ID}"]
dm_policy = "allowlist"
group_policy = "mention"
typing_indicator = true
typing_interval_seconds = 4
max_queue_size = 20
streaming_enabled = true
streaming_interval_seconds = 1.5
streaming_min_chars = 50
streaming_max_chars = 3900
EOF

chmod 600 ~/.config/amcp/config.toml

export AMCP_TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN}"
export AMCP_TELEGRAM_ALLOWED_USERS="${TELEGRAM_ADMIN_USER_ID}"
export AMCP_TELEGRAM_ADMIN_USERS="${TELEGRAM_ADMIN_USER_ID}"

echo "Starting Telegram bot..."
amcp telegram start --allow-user "${TELEGRAM_ADMIN_USER_ID}"
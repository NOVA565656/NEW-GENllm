# Private Telegram Coding Agent

A private, free-tier coding agent that runs in GitHub Codespaces.
It chats via Telegram, browses the web, pushes to Git, and uses a
4-stage verification chain for secure, clean code.

## Quick Start

1. **Create the files** in a new GitHub repository and open it in Codespaces.
2. **Copy the template:** `cp .env.example .env`
3. **Fill in `.env`** with your API keys (see the table below).
4. **Run the installer:** `chmod +x bootstrap.sh && ./bootstrap.sh`
5. **Start the agent:** `./start.sh`

## Keys You Need (All Free Tiers)

| Variable | Provider | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | Groq (Stage 1) | `console.groq.com/keys` |
| `MISTRAL_API_KEY` | Mistral/Codestral (Stage 2) | `console.mistral.ai` |
| `CEREBRAS_API_KEY` | Cerebras (Stage 3) | `cloud.cerebras.ai` |
| `CLOUDFLARE_API_TOKEN` | Cloudflare (Stage 4) | `dash.cloudflare.com` → Workers AI |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare | Same dashboard |
| `VENICE_API_KEY` | Venice (on-demand) | `venice.ai/settings/api` |
| `TELEGRAM_BOT_TOKEN` | Telegram | `@BotFather` |

## Using the Agent

- **Normal coding turn:** Send any message. It runs the 4-stage chain.
- **Uncensored on-demand:** Send `/venice <your task>`.
- **Shut down:** Send `/sleep` to stop the Codespace.
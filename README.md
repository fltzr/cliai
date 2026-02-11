# ⚡ CLIAI

A robust CLI AI chat platform that works seamlessly with **any OpenAI-compatible endpoint** and API key.

Plug in OpenAI, Groq, Together, Ollama, vLLM, LM Studio, or any custom endpoint — just set your URL and key.

## Features

- **Provider-agnostic** — any OpenAI-compatible `/v1/chat/completions` endpoint
- **Named profiles** — switch between endpoints/models with `--profile`
- **Streaming SSE** — live markdown rendering with Rich
- **Session persistence** — auto-save, `/save`, `/load` conversations
- **Slash commands** — `/help`, `/clear`, `/model`, `/system`, `/retry`, `/info`
- **Retry with backoff** — handles rate limits, transient errors, timeouts
- **Config layering** — YAML config → env vars → CLI flags
- **Rich TUI** — styled prompts, spinners, token usage display

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Local Ollama (default)
python chat_cli.py chat

# OpenAI
python chat_cli.py chat -e https://api.openai.com/v1 -k sk-... -m gpt-4o-mini

# Groq
python chat_cli.py chat -e https://api.groq.com/openai/v1 -k gsk_... -m llama-3.3-70b-versatile

# Using a named profile
python chat_cli.py chat --profile openai
```

## Configuration

Config file: `~/.config/cliai/config.yaml` (created on first run)

```yaml
default_profile: default

profiles:
  default:
    endpoint: "http://localhost:11434/v1"
    model: "llama3"
    temperature: 0.7
    stream: true

  openai:
    endpoint: "https://api.openai.com/v1"
    api_key: "sk-..."
    model: "gpt-4o-mini"

  groq:
    endpoint: "https://api.groq.com/openai/v1"
    api_key: "gsk_..."
    model: "llama-3.3-70b-versatile"
```

### Environment Variables

Override any setting with `CLIAI_*` env vars:

| Variable | Description |
|----------|-------------|
| `CLIAI_ENDPOINT` | API endpoint URL |
| `CLIAI_API_KEY` | API key |
| `CLIAI_MODEL` | Model name |
| `CLIAI_TEMPERATURE` | Sampling temperature |
| `CLIAI_MAX_TOKENS` | Max response tokens |
| `CLIAI_SYSTEM_PROMPT` | System prompt |
| `CLIAI_STREAM` | Enable streaming (`true`/`false`) |

## CLI Options

```
python chat_cli.py chat [OPTIONS]

  -e, --endpoint TEXT      API endpoint URL
  -k, --api-key TEXT       API key
  -m, --model TEXT         Model name
  -p, --profile TEXT       Named profile from config
  -s, --system TEXT        System prompt
  -t, --temperature FLOAT  Sampling temperature (0.0-2.0)
  --max-tokens INTEGER     Max response tokens
  --no-stream              Disable streaming
  -l, --load TEXT          Load a saved session
```

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/save [name]` | Save session to file |
| `/load <name>` | Load session from file |
| `/sessions` | List saved sessions |
| `/delete <name>` | Delete a saved session |
| `/model [name]` | Show or switch model |
| `/system [prompt]` | Show or set system prompt |
| `/retry` | Re-send last user message |
| `/info` | Show current configuration |
| `/exit` | Quit |

## Other Commands

```bash
# Show current config
python chat_cli.py config

# Initialize config file
python chat_cli.py config --init

# List saved sessions
python chat_cli.py sessions
```

## Security

CLIAI uses **defense in depth** to control where your data goes:

### Host Allowlisting

Restrict outbound connections to specific hosts. Add to `config.yaml`:

```yaml
security:
  enforce_allowlist: true
  allowed_hosts:
    - "localhost"
    - "api.openai.com"
    - "*.groq.com"          # wildcard subdomains
    - "192.168.1.0/24"      # CIDR ranges
```

Or via environment variables:

```bash
export CLIAI_ENFORCE_ALLOWLIST=true
export CLIAI_ALLOWED_HOSTS="localhost,api.openai.com"
```

When enabled, any request to a host not in the list is **blocked before data leaves the process**.

### Docker Deployment

For OS-level network isolation as a second layer:

```bash
# Build and run
docker compose up --build

# Or run directly with network restrictions
docker run -it --network=none cliai chat -e http://host.docker.internal:11434/v1
```

The included `docker-compose.yaml` provides DNS control, resource limits, and persistent volumes.

## Running Tests

```bash
python -m pytest tests/ -v
```

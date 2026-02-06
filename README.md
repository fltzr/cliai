# Ollama Pro CLI (Interceptable, Production-Grade)

A hardened CLI chat client for Ollama endpoints with a plugin interception pipeline designed for prompt governance, redaction, and request rewriting before network send.

## Why this is production-grade

- **Plugin pipeline** with clear lifecycle hooks:
  - `pre_send(payload, context)`
  - `post_receive(response_json, context)`
- **Retry strategy** with linear backoff for transient failures.
- **Streaming support** for real-time token output (`stream=true`) and non-stream mode.
- **Operational controls**: model switching, history management, save/export, transcript logging.
- **Backward compatibility** with legacy `intercept_payload(payload)` hook.
- **Zero external dependencies** (Python standard library only).

## Files

- `chat_cli.py` — CLI runtime, HTTP client, retry logic, interceptor loader.
- `interceptor_example.py` — ready-to-use plugin for outbound/inbound modifications.

## Quick start

```bash
python3 chat_cli.py \
  --endpoint http://192.168.1.25:11434/api/chat \
  --model llama3.1 \
  --interceptor interceptor_example.py
```

## Interceptor API

Create a file and pass it using `--interceptor <file.py>` (repeatable).

```python
def pre_send(payload: dict, context) -> dict:
    # mutate outgoing request payload
    return payload

def post_receive(response_json: dict, context) -> dict:
    # mutate parsed response before session state update
    return response_json
```

`context` includes:
- `turn_index`
- `started_at`
- `endpoint`

Legacy mode is still supported:

```python
def intercept_payload(payload: dict) -> dict:
    return payload
```

## CLI options

```text
--endpoint URL
--model NAME
--timeout SECONDS
--retries N
--retry-backoff SECONDS
--interceptor PATH   (repeatable)
--system TEXT
--stream / --no-stream
--transcript PATH
--no-transcript
-v / --verbose
```

Environment variables:
- `OLLAMA_ENDPOINT`
- `OLLAMA_MODEL`
- `OLLAMA_TIMEOUT`
- `OLLAMA_RETRIES`
- `OLLAMA_RETRY_BACKOFF`
- `OLLAMA_SYSTEM_PROMPT`
- `OLLAMA_TRANSCRIPT`

## REPL commands

- `/help`
- `/history`
- `/system <prompt>`
- `/model <name>`
- `/pop`
- `/save [path]`
- `/quit`

## Recommended deployment pattern

For a LAN Ollama server:
1. Keep model/policy routing in `pre_send` plugin(s).
2. Add redaction before send and provenance tags after receive.
3. Keep transcripts enabled and ship JSONL logs to your SIEM or data lake.
4. Run with `--verbose` in staging and disable in prod.

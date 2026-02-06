# Ollama CLI Chat (Interceptable)

This repo now contains a production-ready Python CLI chat client for an Ollama endpoint on your LAN, with a **first-class prompt interception layer** so you can inspect and mutate payloads before they are sent.

## Why this approach

Open-source UIs/CLIs exist, but most hide request construction inside framework callbacks, making prompt interception awkward. This tool keeps interception explicit and easy:

- Payload is assembled in one place.
- `intercept_payload(payload)` is called on every turn.
- You can change model, messages, options, routing, tags, etc. before the HTTP request.

## Files

- `chat_cli.py` — interactive CLI chat app.
- `interceptor_example.py` — example hook you can customize.

## Quick start

```bash
python3 chat_cli.py \
  --endpoint http://192.168.1.25:11434/api/chat \
  --model llama3.1
```

Optional environment variables:

- `OLLAMA_ENDPOINT`
- `OLLAMA_MODEL`
- `OLLAMA_TIMEOUT`
- `OLLAMA_INTERCEPTOR`
- `OLLAMA_SYSTEM_PROMPT`
- `OLLAMA_TRANSCRIPT`

## Interception API

Point `--interceptor` at a Python file that defines:

```python
def intercept_payload(payload: dict) -> dict:
    # modify payload
    return payload
```

`payload` starts as:

```json
{
  "model": "llama3.1",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "stream": false
}
```

You can:

- Rewrite or filter prompts.
- Insert policy/system messages.
- Redact secrets before send.
- Adjust `model`/`options` dynamically.

## CLI commands

Inside the REPL:

- `/help`
- `/history`
- `/pop`
- `/system <prompt>`
- `/save [path]`
- `/quit`

## Notes

- Uses the Ollama `/api/chat` endpoint.
- Transcript logging defaults to `chat_transcript.jsonl` (disable with `--no-transcript`).
- Works with any Ollama-reachable host on your local network.

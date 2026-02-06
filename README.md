# Ollama Pro CLI (Click-powered)

A production-focused CLI chat app for Ollama that uses **Click** (a mature open-source CLI framework) plus a plugin pipeline to intercept and rewrite prompts before they are sent.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 chat_cli.py chat \
  --endpoint http://192.168.1.25:11434/api/chat \
  --model llama3.1 \
  --interceptor interceptor_example.py
```

## Why this is better

- Built on **Click** rather than hand-rolled argparse/input loops
- Structured command model (`chat` command + typed options + envvar support)
- Streaming and non-streaming responses
- Retries with backoff for transient failures
- Interceptor lifecycle:
  - `pre_send(payload, context)`
  - `post_receive(response_json, context)`
- Backward compatibility with legacy `intercept_payload(payload)`

## Interceptor API

Create a Python file and pass it with `--interceptor` (repeatable):

```python
def pre_send(payload: dict, context) -> dict:
    return payload

def post_receive(response_json: dict, context) -> dict:
    return response_json
```

Legacy compatibility:

```python
def intercept_payload(payload: dict) -> dict:
    return payload
```

## REPL commands

- `/help`
- `/history`
- `/system <prompt>`
- `/model <name>`
- `/pop`
- `/save [path]`
- `/quit`

## Notes

- Transcript log defaults to `chat_transcript.jsonl`
- Disable transcript logging with `--no-transcript`
- Environment variables supported: `OLLAMA_ENDPOINT`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`, `OLLAMA_RETRIES`, `OLLAMA_RETRY_BACKOFF`, `OLLAMA_SYSTEM_PROMPT`, `OLLAMA_TRANSCRIPT`

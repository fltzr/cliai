"""Example payload interceptor for chat_cli.py.

Define intercept_payload(payload) and return the modified payload.
"""

from __future__ import annotations

from datetime import datetime


def intercept_payload(payload: dict) -> dict:
    payload = dict(payload)
    messages = list(payload.get("messages", []))

    # Inject a lightweight audit prefix into the latest user message.
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            messages[i] = {
                **messages[i],
                "content": f"[sent {datetime.utcnow().isoformat()}Z] {messages[i]['content']}",
            }
            break

    payload["messages"] = messages

    # You can also override model/options per request if needed:
    # payload["model"] = "llama3.1:8b"
    # payload["options"] = {"temperature": 0.2}

    return payload

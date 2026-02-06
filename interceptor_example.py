"""Production interceptor example for chat_cli.py.

Supports both outbound mutation (pre_send) and inbound mutation (post_receive).
"""

from __future__ import annotations

from datetime import datetime, timezone


def pre_send(payload: dict, context) -> dict:
    messages = list(payload.get("messages", []))

    # Example: prepend trace metadata to the latest user turn.
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            stamp = datetime.now(timezone.utc).isoformat()
            messages[i] = {
                **messages[i],
                "content": f"[trace turn={context.turn_index} ts={stamp}] {messages[i]['content']}",
            }
            break

    payload = dict(payload)
    payload["messages"] = messages

    # Example dynamic override hooks:
    # payload["model"] = "llama3.1:8b"
    # payload["options"] = {"temperature": 0.1, "num_ctx": 8192}

    return payload


def post_receive(response_json: dict, context) -> dict:
    # Example: annotate inbound response for local audit.
    msg = response_json.get("message", {})
    content = msg.get("content", "")
    msg["content"] = f"{content}\n\n[received turn={context.turn_index}]"
    response_json["message"] = msg
    return response_json

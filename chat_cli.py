#!/usr/bin/env python3
"""Interactive CLI chat client for Ollama with prompt interception hooks."""

from __future__ import annotations

import argparse
import json
import os
import readline  # noqa: F401  # enables shell history/editing in input()
import sys
import textwrap
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib import error, request


Message = dict[str, str]
Payload = dict[str, Any]
Interceptor = Callable[[Payload], Payload]


DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1"
DEFAULT_TIMEOUT_SECS = 300


@dataclass
class SessionConfig:
    endpoint: str
    model: str
    timeout_secs: int
    interceptor_path: Path | None
    transcript_path: Path | None
    stream: bool


class OllamaChatCLI:
    def __init__(self, config: SessionConfig, system_prompt: str | None = None):
        self.config = config
        self.messages: list[Message] = []
        self.interceptor = self._load_interceptor(config.interceptor_path)
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def run(self) -> None:
        self._print_banner()
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    break
                continue

            self.messages.append({"role": "user", "content": user_input})
            payload = {
                "model": self.config.model,
                "messages": deepcopy(self.messages),
                "stream": self.config.stream,
            }

            payload = self.interceptor(payload)
            self._validate_payload(payload)

            assistant_reply = self._send(payload)
            if assistant_reply is None:
                self.messages.pop()
                continue

            self.messages.append({"role": "assistant", "content": assistant_reply})
            if self.config.transcript_path:
                self._append_transcript(user_input, assistant_reply)

    def _load_interceptor(self, path: Path | None) -> Interceptor:
        if path is None:
            return lambda payload: payload
        if not path.exists():
            raise FileNotFoundError(f"Interceptor file not found: {path}")

        namespace: dict[str, Any] = {}
        code = path.read_text(encoding="utf-8")
        exec(compile(code, str(path), "exec"), namespace)

        interceptor = namespace.get("intercept_payload")
        if not callable(interceptor):
            raise ValueError(
                f"{path} must define a callable function intercept_payload(payload)"
            )
        return interceptor

    @staticmethod
    def _validate_payload(payload: Payload) -> None:
        if "model" not in payload:
            raise ValueError("Intercepted payload missing required key: 'model'")
        if "messages" not in payload or not isinstance(payload["messages"], list):
            raise ValueError("Intercepted payload must include list key: 'messages'")

    def _send(self, payload: Payload) -> str | None:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.config.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.config.timeout_secs) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            msg = exc.read().decode("utf-8", errors="replace")
            print(f"[HTTP {exc.code}] {msg}")
            return None
        except error.URLError as exc:
            print(f"[Network error] {exc.reason}")
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print("[Protocol error] Could not decode Ollama response as JSON")
            print(raw)
            return None

        message = data.get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            print("[Protocol error] Missing message.content in Ollama response")
            print(json.dumps(data, indent=2))
            return None

        print(f"ai> {content}")
        return content

    def _handle_command(self, text: str) -> bool:
        command, _, rest = text.partition(" ")
        arg = rest.strip()

        if command in {"/q", "/quit", "/exit"}:
            return True
        if command == "/help":
            self._print_help()
            return False
        if command == "/history":
            self._print_history()
            return False
        if command == "/pop":
            self._pop_last_turn()
            return False
        if command == "/system":
            self._set_system_prompt(arg)
            return False
        if command == "/save":
            target = Path(arg) if arg else Path("chat_session.json")
            target.write_text(json.dumps(self.messages, indent=2), encoding="utf-8")
            print(f"Saved chat history to {target}")
            return False

        print("Unknown command. Type /help for available commands.")
        return False

    def _print_history(self) -> None:
        if not self.messages:
            print("(no history)")
            return
        for i, m in enumerate(self.messages, start=1):
            print(f"{i:>2}. {m['role']}: {m['content']}")

    def _pop_last_turn(self) -> None:
        if len(self.messages) < 2:
            print("Not enough history to remove a full turn.")
            return
        self.messages.pop()
        self.messages.pop()
        print("Removed last user+assistant turn.")

    def _set_system_prompt(self, prompt: str) -> None:
        if not prompt:
            print("Usage: /system <new system prompt>")
            return
        for i, m in enumerate(self.messages):
            if m["role"] == "system":
                self.messages[i] = {"role": "system", "content": prompt}
                print("Updated system prompt.")
                return
        self.messages.insert(0, {"role": "system", "content": prompt})
        print("Added system prompt.")

    def _append_transcript(self, user_text: str, ai_text: str) -> None:
        line = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user": user_text,
            "assistant": ai_text,
        }
        with self.config.transcript_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    @staticmethod
    def _print_banner() -> None:
        print("Ollama CLI Chat (with payload interceptor)")
        print("Type /help for commands, /quit to exit.\n")

    @staticmethod
    def _print_help() -> None:
        print(
            textwrap.dedent(
                """
                Commands:
                  /help               Show this help text
                  /history            Print current conversation history
                  /pop                Remove the latest user+assistant pair
                  /system <prompt>    Set or update system prompt
                  /save [path]        Save message list as JSON
                  /quit               Exit
                """
            ).strip()
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI chat client for Ollama with pluggable prompt interception"
    )
    parser.add_argument("--endpoint", default=os.getenv("OLLAMA_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT_SECS)),
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--interceptor",
        type=Path,
        default=Path(os.getenv("OLLAMA_INTERCEPTOR", "interceptor_example.py")),
        help="Python file defining intercept_payload(payload)->payload",
    )
    parser.add_argument("--system", default=os.getenv("OLLAMA_SYSTEM_PROMPT"))
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path(os.getenv("OLLAMA_TRANSCRIPT", "chat_transcript.jsonl")),
        help="JSONL file where chat turns are appended",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Disable transcript writing",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Set stream=false (default false currently for simpler interception)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SessionConfig(
        endpoint=args.endpoint,
        model=args.model,
        timeout_secs=args.timeout,
        interceptor_path=args.interceptor,
        transcript_path=None if args.no_transcript else args.transcript,
        stream=False if args.no_stream else False,
    )

    cli = OllamaChatCLI(config=config, system_prompt=args.system)
    cli.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

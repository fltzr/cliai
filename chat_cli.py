#!/usr/bin/env python3
"""Production-grade Ollama CLI with configurable interception pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import readline  # noqa: F401
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib import error, request


DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1"
DEFAULT_TIMEOUT = 90
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF_SECS = 1.5

JSONDict = dict[str, Any]
Message = dict[str, str]


class ChatError(Exception):
    """Base error for user-facing failures."""


class InterceptorError(ChatError):
    """Raised when an interceptor cannot be loaded or executed."""


class ProtocolError(ChatError):
    """Raised when remote protocol expectations are violated."""


@dataclass(slots=True)
class RuntimeConfig:
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    timeout: int = DEFAULT_TIMEOUT
    stream: bool = True
    retries: int = DEFAULT_RETRIES
    retry_backoff_secs: float = DEFAULT_BACKOFF_SECS
    transcript_path: Path | None = Path("chat_transcript.jsonl")
    interceptors: list[Path] = field(default_factory=list)
    verbose: bool = False


@dataclass(slots=True)
class InterceptorContext:
    turn_index: int
    started_at: datetime
    endpoint: str


PreSendInterceptor = Callable[[JSONDict, InterceptorContext], JSONDict]
PostReceiveInterceptor = Callable[[JSONDict, InterceptorContext], JSONDict]


class InterceptorPipeline:
    """Loads and executes interceptor plugins from Python files.

    A plugin file can define either or both:
      - pre_send(payload, context) -> payload
      - post_receive(response_json, context) -> response_json
    """

    def __init__(self, files: Iterable[Path]):
        self.pre_send_hooks: list[PreSendInterceptor] = []
        self.post_receive_hooks: list[PostReceiveInterceptor] = []
        for file in files:
            self._load_one(file)

    def _load_one(self, file: Path) -> None:
        if not file.exists():
            raise InterceptorError(f"Interceptor file does not exist: {file}")

        spec = importlib.util.spec_from_file_location(f"interceptor_{file.stem}", file)
        if spec is None or spec.loader is None:
            raise InterceptorError(f"Could not load interceptor module from {file}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # backwards compatibility with previous quick prototype name
        intercept_payload = getattr(module, "intercept_payload", None)
        pre_send = getattr(module, "pre_send", None)
        post_receive = getattr(module, "post_receive", None)

        if callable(intercept_payload) and not callable(pre_send):
            pre_send = lambda payload, _ctx: intercept_payload(payload)  # noqa: E731

        if callable(pre_send):
            self.pre_send_hooks.append(pre_send)
        if callable(post_receive):
            self.post_receive_hooks.append(post_receive)

        if not callable(pre_send) and not callable(post_receive):
            raise InterceptorError(
                f"{file} must define pre_send(...) and/or post_receive(...), "
                "or legacy intercept_payload(payload)."
            )

    def run_pre_send(self, payload: JSONDict, context: InterceptorContext) -> JSONDict:
        out = payload
        for hook in self.pre_send_hooks:
            out = hook(out, context)
            self._validate_payload(out)
        return out

    def run_post_receive(self, response_json: JSONDict, context: InterceptorContext) -> JSONDict:
        out = response_json
        for hook in self.post_receive_hooks:
            out = hook(out, context)
            if not isinstance(out, dict):
                raise InterceptorError("post_receive must return a JSON object (dict)")
        return out

    @staticmethod
    def _validate_payload(payload: JSONDict) -> None:
        if not isinstance(payload, dict):
            raise InterceptorError("pre_send interceptor must return dict payload")
        if not isinstance(payload.get("model"), str) or not payload["model"].strip():
            raise InterceptorError("payload.model must be a non-empty string")
        if not isinstance(payload.get("messages"), list):
            raise InterceptorError("payload.messages must be a list")


class OllamaClient:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def chat(self, payload: JSONDict) -> str:
        if payload.get("stream"):
            return self._chat_stream(payload)
        return self._chat_single(payload)

    def _chat_single(self, payload: JSONDict) -> str:
        data = self._request_json(payload)
        msg = data.get("message", {})
        content = msg.get("content")
        if not isinstance(content, str):
            raise ProtocolError("Ollama response missing message.content")
        return content

    def _chat_stream(self, payload: JSONDict) -> str:
        req = self._build_request(payload)
        chunks: list[str] = []
        with request.urlopen(req, timeout=self.config.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProtocolError(f"Invalid streamed JSON line: {line}") from exc

                msg = evt.get("message", {})
                piece = msg.get("content", "")
                if isinstance(piece, str) and piece:
                    print(piece, end="", flush=True)
                    chunks.append(piece)
                if evt.get("done") is True:
                    break
        print()
        return "".join(chunks)

    def _request_json(self, payload: JSONDict) -> JSONDict:
        req = self._build_request(payload)
        with request.urlopen(req, timeout=self.config.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError("Invalid JSON from Ollama") from exc

    def _build_request(self, payload: JSONDict) -> request.Request:
        return request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )


class ChatSession:
    def __init__(self, config: RuntimeConfig, pipeline: InterceptorPipeline):
        self.config = config
        self.pipeline = pipeline
        self.client = OllamaClient(config)
        self.messages: list[Message] = []
        self.stop_event = threading.Event()

    def run(self, system_prompt: str | None = None) -> int:
        self._install_signal_handlers()
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        print("Ollama Pro CLI ready. Type /help for commands.")
        turn = 0
        while not self.stop_event.is_set():
            try:
                user = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user:
                continue
            if user.startswith("/"):
                if self._handle_command(user):
                    break
                continue

            turn += 1
            self.messages.append({"role": "user", "content": user})
            context = InterceptorContext(
                turn_index=turn,
                started_at=datetime.now(timezone.utc),
                endpoint=self.config.endpoint,
            )
            payload: JSONDict = {
                "model": self.config.model,
                "messages": list(self.messages),
                "stream": self.config.stream,
            }

            try:
                payload = self.pipeline.run_pre_send(payload, context)
                reply = self._request_with_retries(payload)
                response_json = {"message": {"role": "assistant", "content": reply}}
                response_json = self.pipeline.run_post_receive(response_json, context)
                final = response_json["message"]["content"]
            except ChatError as exc:
                logging.error("%s", exc)
                self.messages.pop()  # drop failed user turn
                continue

            if not self.config.stream:
                print(f"ai> {final}")

            self.messages.append({"role": "assistant", "content": final})
            self._append_transcript(user, final)

        return 0

    def _request_with_retries(self, payload: JSONDict) -> str:
        attempts = self.config.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self.client.chat(payload)
            except (error.HTTPError, error.URLError, TimeoutError, ProtocolError) as exc:
                should_retry = attempt < attempts
                if isinstance(exc, error.HTTPError) and 400 <= exc.code < 500 and exc.code != 429:
                    should_retry = False

                if not should_retry:
                    if isinstance(exc, error.HTTPError):
                        body = exc.read().decode("utf-8", errors="replace")
                        raise ChatError(f"HTTP {exc.code}: {body}")
                    raise ChatError(f"Request failed: {exc}")

                backoff = self.config.retry_backoff_secs * attempt
                logging.warning("request failed (%s), retrying in %.1fs", exc, backoff)
                time.sleep(backoff)

        raise ChatError("Retries exhausted")

    def _append_transcript(self, user: str, assistant: str) -> None:
        if not self.config.transcript_path:
            return
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user,
            "assistant": assistant,
        }
        self.config.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _handle_command(self, text: str) -> bool:
        cmd, _, arg = text.partition(" ")
        arg = arg.strip()

        if cmd in {"/q", "/quit", "/exit"}:
            return True
        if cmd == "/help":
            print(
                "Commands: /help /history /system <prompt> /pop /save [file] "
                "/model <name> /quit"
            )
            return False
        if cmd == "/history":
            for i, msg in enumerate(self.messages, start=1):
                print(f"{i:>3} {msg['role']}: {msg['content']}")
            return False
        if cmd == "/system":
            if not arg:
                print("Usage: /system <prompt>")
            else:
                self._set_system(arg)
            return False
        if cmd == "/model":
            if not arg:
                print(f"Current model: {self.config.model}")
            else:
                self.config.model = arg
                print(f"Model set: {arg}")
            return False
        if cmd == "/pop":
            if len(self.messages) >= 2:
                self.messages.pop()
                self.messages.pop()
                print("Last turn removed")
            else:
                print("No full turn to remove")
            return False
        if cmd == "/save":
            path = Path(arg) if arg else Path("chat_session.json")
            path.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Saved session: {path}")
            return False

        print("Unknown command. /help for commands.")
        return False

    def _set_system(self, prompt: str) -> None:
        for i, msg in enumerate(self.messages):
            if msg["role"] == "system":
                self.messages[i] = {"role": "system", "content": prompt}
                print("System prompt updated")
                return
        self.messages.insert(0, {"role": "system", "content": prompt})
        print("System prompt added")

    def _install_signal_handlers(self) -> None:
        def _handle(_sig: int, _frame: Any) -> None:
            self.stop_event.set()

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production Ollama CLI with interception")
    parser.add_argument("--endpoint", default=os.getenv("OLLAMA_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_TIMEOUT))))
    parser.add_argument("--retries", type=int, default=int(os.getenv("OLLAMA_RETRIES", str(DEFAULT_RETRIES))))
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=float(os.getenv("OLLAMA_RETRY_BACKOFF", str(DEFAULT_BACKOFF_SECS))),
    )
    parser.add_argument(
        "--interceptor",
        action="append",
        default=[],
        help="Path to interceptor file (repeatable)",
    )
    parser.add_argument("--system", default=os.getenv("OLLAMA_SYSTEM_PROMPT"))
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--transcript", default=os.getenv("OLLAMA_TRANSCRIPT", "chat_transcript.jsonl"))
    parser.add_argument("--no-transcript", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> RuntimeConfig:
    transcript_path = None if args.no_transcript else Path(args.transcript)
    stream = False if args.no_stream else bool(args.stream)
    interceptors = [Path(p) for p in args.interceptor]

    return RuntimeConfig(
        endpoint=args.endpoint,
        model=args.model,
        timeout=args.timeout,
        stream=stream,
        retries=max(0, args.retries),
        retry_backoff_secs=max(0.1, args.retry_backoff),
        transcript_path=transcript_path,
        interceptors=interceptors,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    logging.basicConfig(
        level=logging.DEBUG if config.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        pipeline = InterceptorPipeline(config.interceptors)
        session = ChatSession(config, pipeline)
        return session.run(system_prompt=args.system)
    except ChatError as exc:
        logging.error("fatal: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())

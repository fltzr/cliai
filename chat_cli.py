#!/usr/bin/env python3
"""Ollama Pro CLI built with Click and interceptor plugins."""

from __future__ import annotations

import importlib.util
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib import error, request

import click

DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.1"

JSONDict = dict[str, Any]
Message = dict[str, str]


class ChatError(Exception):
    pass


class InterceptorError(ChatError):
    pass


class ProtocolError(ChatError):
    pass


@dataclass(slots=True)
class RuntimeConfig:
    endpoint: str
    model: str
    timeout: int
    retries: int
    retry_backoff_secs: float
    stream: bool
    transcript_path: Path | None
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

        legacy = getattr(module, "intercept_payload", None)
        pre_send = getattr(module, "pre_send", None)
        post_receive = getattr(module, "post_receive", None)

        if callable(legacy) and not callable(pre_send):
            pre_send = lambda payload, _ctx: legacy(payload)  # noqa: E731

        if callable(pre_send):
            self.pre_send_hooks.append(pre_send)
        if callable(post_receive):
            self.post_receive_hooks.append(post_receive)

        if not callable(pre_send) and not callable(post_receive):
            raise InterceptorError(
                f"{file} must define pre_send/post_receive or legacy intercept_payload"
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
                raise InterceptorError("post_receive must return dict")
        return out

    @staticmethod
    def _validate_payload(payload: JSONDict) -> None:
        if not isinstance(payload, dict):
            raise InterceptorError("interceptor must return dict")
        if not isinstance(payload.get("model"), str):
            raise InterceptorError("payload.model must be string")
        if not isinstance(payload.get("messages"), list):
            raise InterceptorError("payload.messages must be list")


class OllamaClient:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def chat(self, payload: JSONDict) -> str:
        return self._chat_stream(payload) if payload.get("stream") else self._chat_single(payload)

    def _chat_single(self, payload: JSONDict) -> str:
        req = self._request(payload)
        with request.urlopen(req, timeout=self.config.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ProtocolError("Ollama response missing message.content")
        return content

    def _chat_stream(self, payload: JSONDict) -> str:
        req = self._request(payload)
        chunks: list[str] = []
        click.secho("ai> ", fg="cyan", nl=False, bold=True)
        with request.urlopen(req, timeout=self.config.timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                evt = json.loads(line)
                piece = evt.get("message", {}).get("content", "")
                if piece:
                    click.echo(piece, nl=False)
                    chunks.append(piece)
                if evt.get("done"):
                    break
        click.echo()
        return "".join(chunks)

    def _request(self, payload: JSONDict) -> request.Request:
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
        self.stopping = False

    def run(self, system_prompt: str | None) -> int:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        click.secho("Ollama Pro CLI (Click edition)", fg="green", bold=True)
        click.echo("Commands: /help /history /system /model /pop /save /quit")

        turn_index = 0
        while not self.stopping:
            try:
                user_text = click.prompt("you", prompt_suffix="> ").strip()
            except (EOFError, KeyboardInterrupt):
                click.echo("\nExiting.")
                break
            if not user_text:
                continue
            if user_text.startswith("/"):
                if self._handle_command(user_text):
                    break
                continue

            self.messages.append({"role": "user", "content": user_text})
            turn_index += 1
            ctx = InterceptorContext(turn_index, datetime.now(timezone.utc), self.config.endpoint)
            payload: JSONDict = {
                "model": self.config.model,
                "messages": list(self.messages),
                "stream": self.config.stream,
            }

            try:
                payload = self.pipeline.run_pre_send(payload, ctx)
                reply = self._request_with_retries(payload)
                response = self.pipeline.run_post_receive(
                    {"message": {"role": "assistant", "content": reply}}, ctx
                )
                final = response["message"]["content"]
            except (ChatError, error.URLError, error.HTTPError, json.JSONDecodeError) as exc:
                click.secho(f"error: {exc}", fg="red")
                self.messages.pop()
                continue

            if not self.config.stream:
                click.secho(f"ai> {final}", fg="cyan", bold=True)

            self.messages.append({"role": "assistant", "content": final})
            self._append_transcript(user_text, final)

        return 0

    def _request_with_retries(self, payload: JSONDict) -> str:
        for attempt in range(1, self.config.retries + 2):
            try:
                return self.client.chat(payload)
            except (error.URLError, error.HTTPError, TimeoutError, ProtocolError) as exc:
                is_retryable_http = isinstance(exc, error.HTTPError) and (exc.code >= 500 or exc.code == 429)
                retryable = isinstance(exc, (error.URLError, TimeoutError, ProtocolError)) or is_retryable_http
                if attempt > self.config.retries or not retryable:
                    if isinstance(exc, error.HTTPError):
                        body = exc.read().decode("utf-8", errors="replace")
                        raise ChatError(f"HTTP {exc.code}: {body}")
                    raise ChatError(str(exc))
                delay = self.config.retry_backoff_secs * attempt
                click.secho(f"retry {attempt}/{self.config.retries} in {delay:.1f}s...", fg="yellow")
                time.sleep(delay)
        raise ChatError("Retries exhausted")

    def _append_transcript(self, user: str, assistant: str) -> None:
        if not self.config.transcript_path:
            return
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user,
            "assistant": assistant,
            "model": self.config.model,
        }
        self.config.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.transcript_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _handle_command(self, command: str) -> bool:
        cmd, _, arg = command.partition(" ")
        arg = arg.strip()

        if cmd in {"/q", "/quit", "/exit"}:
            return True
        if cmd == "/help":
            click.echo("/help /history /system <p> /model <name> /pop /save [path] /quit")
            return False
        if cmd == "/history":
            for i, msg in enumerate(self.messages, start=1):
                click.echo(f"{i:>3} {msg['role']}: {msg['content']}")
            return False
        if cmd == "/system":
            if not arg:
                click.echo("Usage: /system <prompt>")
            else:
                self._set_system_prompt(arg)
            return False
        if cmd == "/model":
            if arg:
                self.config.model = arg
                click.secho(f"Model set to {arg}", fg="green")
            else:
                click.secho(f"Current model: {self.config.model}", fg="green")
            return False
        if cmd == "/pop":
            if len(self.messages) >= 2:
                self.messages.pop()
                self.messages.pop()
                click.echo("Removed last turn")
            else:
                click.echo("No complete turn to remove")
            return False
        if cmd == "/save":
            path = Path(arg) if arg else Path("chat_session.json")
            path.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False), encoding="utf-8")
            click.secho(f"Saved session to {path}", fg="green")
            return False

        click.echo("Unknown command. Use /help")
        return False

    def _set_system_prompt(self, prompt: str) -> None:
        for i, msg in enumerate(self.messages):
            if msg["role"] == "system":
                self.messages[i] = {"role": "system", "content": prompt}
                click.echo("Updated system prompt")
                return
        self.messages.insert(0, {"role": "system", "content": prompt})
        click.echo("Added system prompt")

    def _signal_handler(self, _sig: int, _frame: Any) -> None:
        self.stopping = True


@click.group()
def app() -> None:
    """Production Ollama CLI built with Click."""


@app.command()
@click.option("--endpoint", default=DEFAULT_ENDPOINT, envvar="OLLAMA_ENDPOINT", show_default=True)
@click.option("--model", default=DEFAULT_MODEL, envvar="OLLAMA_MODEL", show_default=True)
@click.option("--timeout", default=90, type=int, envvar="OLLAMA_TIMEOUT", show_default=True)
@click.option("--retries", default=2, type=int, envvar="OLLAMA_RETRIES", show_default=True)
@click.option("--retry-backoff", default=1.5, type=float, envvar="OLLAMA_RETRY_BACKOFF", show_default=True)
@click.option("--interceptor", type=click.Path(path_type=Path), multiple=True)
@click.option("--system", default=None, envvar="OLLAMA_SYSTEM_PROMPT")
@click.option("--stream/--no-stream", default=True)
@click.option("--transcript", default="chat_transcript.jsonl", envvar="OLLAMA_TRANSCRIPT", type=click.Path(path_type=Path))
@click.option("--no-transcript", is_flag=True, default=False)
@click.option("-v", "--verbose", is_flag=True, default=False)
def chat(
    endpoint: str,
    model: str,
    timeout: int,
    retries: int,
    retry_backoff: float,
    interceptor: tuple[Path, ...],
    system: str | None,
    stream: bool,
    transcript: Path,
    no_transcript: bool,
    verbose: bool,
) -> None:
    """Start an interactive chat session."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    config = RuntimeConfig(
        endpoint=endpoint,
        model=model,
        timeout=max(1, timeout),
        retries=max(0, retries),
        retry_backoff_secs=max(0.1, retry_backoff),
        stream=stream,
        transcript_path=None if no_transcript else transcript,
        interceptors=list(interceptor),
        verbose=verbose,
    )

    try:
        pipeline = InterceptorPipeline(config.interceptors)
        session = ChatSession(config, pipeline)
        raise SystemExit(session.run(system_prompt=system))
    except ChatError as exc:
        click.secho(f"fatal: {exc}", fg="red", bold=True)
        raise SystemExit(2)


def main() -> int:
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())

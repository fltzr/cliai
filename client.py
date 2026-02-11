"""
OpenAI-compatible streaming API client for CLIAI.

Handles SSE streaming, retries with exponential backoff,
timeouts, and graceful error handling for any OpenAI-compatible endpoint.
"""

import json
import time
from dataclasses import dataclass
from typing import Generator, Optional

import httpx

from config import Config
from network import validate_endpoint, BlockedHostError


# Retryable HTTP status codes
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
BASE_BACKOFF = 1.0  # seconds


@dataclass
class ChatChunk:
    """A single chunk from the streaming response."""

    delta_content: str = ""
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict] = None


class APIError(Exception):
    """Raised for non-retryable API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        self.status_code = status_code
        super().__init__(message)


class ChatClient:
    """OpenAI-compatible chat completions client with streaming support."""

    def __init__(self, config: Config):
        self.config = config
        self._base_url = config.endpoint.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if config.api_key:
            self._headers["Authorization"] = f"Bearer {config.api_key}"

        self._timeout = httpx.Timeout(
            connect=10.0,
            read=120.0,
            write=10.0,
            pool=10.0,
        )

        # Network security
        self._allowed_hosts = config.allowed_hosts
        self._enforce_allowlist = config.enforce_allowlist

    def _build_payload(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Build the request payload."""
        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": stream if stream is not None else self.config.stream,
        }

        temp = temperature if temperature is not None else self.config.temperature
        if temp is not None:
            payload["temperature"] = temp

        tokens = max_tokens or self.config.max_tokens
        if tokens is not None:
            payload["max_tokens"] = tokens

        # Include stream_options for usage in streaming mode
        if payload["stream"]:
            payload["stream_options"] = {"include_usage": True}

        return payload

    def _parse_sse_line(self, line: str) -> Optional[ChatChunk]:
        """Parse a single SSE line into a ChatChunk."""
        line = line.strip()
        if not line or line.startswith(":"):
            return None

        if not line.startswith("data: "):
            return None

        data = line[6:]  # Strip "data: " prefix

        if data == "[DONE]":
            return ChatChunk(finish_reason="stop")

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None

        # Handle error responses embedded in SSE
        if "error" in parsed:
            error_msg = parsed["error"]
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", str(error_msg))
            return ChatChunk(error=str(error_msg))

        # Extract delta content from choices
        choices = parsed.get("choices", [])
        if choices:
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            finish = choice.get("finish_reason")
            return ChatChunk(delta_content=content, finish_reason=finish)

        # Extract usage info (often in the final message)
        usage = parsed.get("usage")
        if usage:
            return ChatChunk(usage=usage)

        return None

    def stream_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[ChatChunk, None, None]:
        """
        Send a chat completion request and yield streaming chunks.

        Implements retry with exponential backoff for transient errors.

        Yields:
            ChatChunk objects with delta content, finish reasons, errors, or usage.
        """
        payload = self._build_payload(
            messages, model=model, stream=True,
            temperature=temperature, max_tokens=max_tokens,
        )
        url = f"{self._base_url}/chat/completions"

        # Network security gate
        try:
            validate_endpoint(url, self._allowed_hosts, self._enforce_allowlist)
        except BlockedHostError as e:
            yield ChatChunk(error=str(e))
            return

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    with client.stream(
                        "POST", url, json=payload, headers=self._headers
                    ) as response:
                        # Check for HTTP errors
                        if response.status_code != 200:
                            status = response.status_code

                            # Read body for error details
                            body = ""
                            try:
                                body = response.read().decode("utf-8", errors="replace")
                            except Exception:
                                pass

                            error_message = _extract_error_message(body, status)

                            if status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                                backoff = _calculate_backoff(attempt, response)
                                last_error = error_message
                                time.sleep(backoff)
                                continue

                            yield ChatChunk(error=error_message)
                            return

                        # Parse SSE stream
                        buffer = ""
                        for raw_bytes in response.iter_bytes():
                            buffer += raw_bytes.decode("utf-8", errors="replace")
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                chunk = self._parse_sse_line(line)
                                if chunk:
                                    if chunk.error:
                                        yield chunk
                                        return
                                    yield chunk
                                    if chunk.finish_reason == "stop":
                                        return

                        # Process any remaining data in buffer
                        if buffer.strip():
                            chunk = self._parse_sse_line(buffer)
                            if chunk:
                                yield chunk

                        return  # Success — don't retry

            except httpx.ConnectError:
                last_error = (
                    f"Connection refused: {url}\n"
                    "Is the server running? Check your --endpoint setting."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_calculate_backoff(attempt))
                    continue
                yield ChatChunk(error=last_error)
                return

            except httpx.ConnectTimeout:
                last_error = (
                    f"Connection timed out: {url}\n"
                    "The server may be down or unreachable."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_calculate_backoff(attempt))
                    continue
                yield ChatChunk(error=last_error)
                return

            except httpx.ReadTimeout:
                last_error = (
                    "Read timeout: The server took too long to respond.\n"
                    "The model may be loading or the request is too large."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_calculate_backoff(attempt))
                    continue
                yield ChatChunk(error=last_error)
                return

            except httpx.HTTPError as e:
                last_error = f"HTTP error: {e}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_calculate_backoff(attempt))
                    continue
                yield ChatChunk(error=last_error)
                return

            except KeyboardInterrupt:
                yield ChatChunk(finish_reason="interrupted")
                return

        # Exhausted retries
        yield ChatChunk(
            error=f"Failed after {MAX_RETRIES} retries. Last error: {last_error}"
        )

    def send_chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatChunk:
        """
        Send a non-streaming chat completion request.

        Returns:
            A single ChatChunk with the full response.
        """
        payload = self._build_payload(
            messages, model=model, stream=False,
            temperature=temperature, max_tokens=max_tokens,
        )
        url = f"{self._base_url}/chat/completions"

        # Network security gate
        try:
            validate_endpoint(url, self._allowed_hosts, self._enforce_allowlist)
        except BlockedHostError as e:
            return ChatChunk(error=str(e))

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(url, json=payload, headers=self._headers)

                    if response.status_code != 200:
                        status = response.status_code
                        error_message = _extract_error_message(
                            response.text, status
                        )

                        if status in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                            backoff = _calculate_backoff(attempt, response)
                            last_error = error_message
                            time.sleep(backoff)
                            continue

                        return ChatChunk(error=error_message)

                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        return ChatChunk(error="Empty response: no choices returned.")

                    content = choices[0].get("message", {}).get("content", "")
                    finish = choices[0].get("finish_reason", "stop")
                    usage = data.get("usage")

                    return ChatChunk(
                        delta_content=content,
                        finish_reason=finish,
                        usage=usage,
                    )

            except httpx.ConnectError:
                last_error = f"Connection refused: {url}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_calculate_backoff(attempt))
                    continue
                return ChatChunk(error=last_error)

            except httpx.TimeoutException:
                last_error = "Request timed out."
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_calculate_backoff(attempt))
                    continue
                return ChatChunk(error=last_error)

            except httpx.HTTPError as e:
                last_error = f"HTTP error: {e}"
                if attempt < MAX_RETRIES - 1:
                    time.sleep(_calculate_backoff(attempt))
                    continue
                return ChatChunk(error=last_error)

        return ChatChunk(
            error=f"Failed after {MAX_RETRIES} retries. Last error: {last_error}"
        )


def _extract_error_message(body: str, status_code: int) -> str:
    """Extract a human-readable error message from an HTTP error response."""
    # Common status code meanings
    status_hints = {
        401: "Authentication failed — check your API key.",
        403: "Access forbidden — your API key may not have permission for this model.",
        404: "Endpoint not found — check your --endpoint URL and model name.",
        422: "Invalid request — the server rejected the payload.",
        429: "Rate limited — too many requests. Will retry.",
    }

    hint = status_hints.get(status_code, "")

    # Try to parse JSON error body
    try:
        error_data = json.loads(body)
        if isinstance(error_data, dict):
            err = error_data.get("error", {})
            if isinstance(err, dict):
                msg = err.get("message", "")
            else:
                msg = str(err)
            if msg:
                return f"[{status_code}] {msg}" + (f"\n💡 {hint}" if hint else "")
    except (json.JSONDecodeError, TypeError):
        pass

    # If body is short, use it as error
    if body and len(body) < 500:
        return f"[{status_code}] {body.strip()}" + (f"\n💡 {hint}" if hint else "")

    return f"[{status_code}] {hint or 'Server error.'}"


def _calculate_backoff(
    attempt: int, response: Optional[httpx.Response] = None
) -> float:
    """Calculate backoff time, respecting Retry-After header."""
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    return BASE_BACKOFF * (2**attempt)

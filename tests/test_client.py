"""Tests for client.py — API client SSE parsing and error handling."""

import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import ChatClient, ChatChunk, _extract_error_message, _calculate_backoff
from config import Config


class TestSSEParsing:
    """Test Server-Sent Events line parsing."""

    def setup_method(self):
        self.client = ChatClient(Config(endpoint="http://test:8080/v1"))

    def test_parse_content_chunk(self):
        data = {
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]
        }
        line = f"data: {json.dumps(data)}"
        chunk = self.client._parse_sse_line(line)
        assert chunk is not None
        assert chunk.delta_content == "Hello"
        assert chunk.finish_reason is None

    def test_parse_done_signal(self):
        chunk = self.client._parse_sse_line("data: [DONE]")
        assert chunk is not None
        assert chunk.finish_reason == "stop"

    def test_parse_finish_reason(self):
        data = {
            "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]
        }
        line = f"data: {json.dumps(data)}"
        chunk = self.client._parse_sse_line(line)
        assert chunk is not None
        assert chunk.finish_reason == "stop"

    def test_parse_error_response(self):
        data = {"error": {"message": "Model not found"}}
        line = f"data: {json.dumps(data)}"
        chunk = self.client._parse_sse_line(line)
        assert chunk is not None
        assert chunk.error == "Model not found"

    def test_parse_usage_chunk(self):
        data = {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
        line = f"data: {json.dumps(data)}"
        chunk = self.client._parse_sse_line(line)
        assert chunk is not None
        assert chunk.usage["total_tokens"] == 30

    def test_parse_empty_line(self):
        assert self.client._parse_sse_line("") is None

    def test_parse_comment_line(self):
        assert self.client._parse_sse_line(": keep-alive") is None

    def test_parse_invalid_json(self):
        assert self.client._parse_sse_line("data: {invalid}") is None

    def test_parse_non_data_line(self):
        assert self.client._parse_sse_line("event: message") is None


class TestPayloadBuilding:
    """Test request payload construction."""

    def test_default_payload(self):
        client = ChatClient(Config(model="gpt-4", temperature=0.5))
        messages = [{"role": "user", "content": "Hi"}]
        payload = client._build_payload(messages)
        assert payload["model"] == "gpt-4"
        assert payload["messages"] == messages
        assert payload["stream"] is True
        assert payload["temperature"] == 0.5

    def test_override_model(self):
        client = ChatClient(Config(model="gpt-4"))
        payload = client._build_payload([], model="gpt-3.5-turbo")
        assert payload["model"] == "gpt-3.5-turbo"

    def test_non_stream_payload(self):
        client = ChatClient(Config())
        payload = client._build_payload([], stream=False)
        assert payload["stream"] is False
        assert "stream_options" not in payload

    def test_max_tokens_in_payload(self):
        client = ChatClient(Config(max_tokens=512))
        payload = client._build_payload([])
        assert payload["max_tokens"] == 512

    def test_stream_options_included(self):
        client = ChatClient(Config())
        payload = client._build_payload([], stream=True)
        assert payload["stream_options"] == {"include_usage": True}


class TestErrorExtraction:
    """Test error message parsing from HTTP responses."""

    def test_json_error_body(self):
        body = json.dumps({"error": {"message": "Invalid API key"}})
        msg = _extract_error_message(body, 401)
        assert "Invalid API key" in msg
        assert "401" in msg

    def test_plain_text_error(self):
        msg = _extract_error_message("Bad Request", 400)
        assert "Bad Request" in msg

    def test_status_hint_included(self):
        msg = _extract_error_message("", 429)
        assert "Rate limited" in msg

    def test_unknown_status(self):
        msg = _extract_error_message("", 418)
        assert "418" in msg


class TestBackoff:
    """Test exponential backoff calculation."""

    def test_first_attempt(self):
        delay = _calculate_backoff(0)
        assert delay == 1.0

    def test_second_attempt(self):
        delay = _calculate_backoff(1)
        assert delay == 2.0

    def test_third_attempt(self):
        delay = _calculate_backoff(2)
        assert delay == 4.0

    def test_retry_after_header(self):
        mock_response = mock.MagicMock()
        mock_response.headers = {"retry-after": "5"}
        delay = _calculate_backoff(0, response=mock_response)
        assert delay == 5.0


class TestClientHeaders:
    """Test that client sets correct headers."""

    def test_headers_with_api_key(self):
        client = ChatClient(Config(api_key="sk-test123"))
        assert client._headers["Authorization"] == "Bearer sk-test123"
        assert client._headers["Content-Type"] == "application/json"

    def test_headers_without_api_key(self):
        client = ChatClient(Config(api_key=""))
        assert "Authorization" not in client._headers


class TestNetworkSecurity:
    """Test host allowlist enforcement in ChatClient."""

    def test_stream_chat_blocked_host(self):
        """stream_chat should yield an error chunk when host is blocked."""
        config = Config(
            endpoint="https://evil.com/v1",
            allowed_hosts=["api.openai.com"],
            enforce_allowlist=True,
        )
        client = ChatClient(config)
        messages = [{"role": "user", "content": "hi"}]

        chunks = list(client.stream_chat(messages))
        assert len(chunks) == 1
        assert chunks[0].error is not None
        assert "evil.com" in chunks[0].error

    def test_send_chat_blocked_host(self):
        """send_chat should return an error chunk when host is blocked."""
        config = Config(
            endpoint="https://evil.com/v1",
            allowed_hosts=["api.openai.com"],
            enforce_allowlist=True,
        )
        client = ChatClient(config)
        messages = [{"role": "user", "content": "hi"}]

        result = client.send_chat(messages)
        assert result.error is not None
        assert "evil.com" in result.error

    def test_no_enforcement_allows_any_host(self):
        """When enforce_allowlist=False, no blocking occurs (request may still fail for other reasons)."""
        config = Config(
            endpoint="https://anything.com/v1",
            allowed_hosts=[],
            enforce_allowlist=False,
        )
        client = ChatClient(config)
        # Just verify the validation gate doesn't block — the actual HTTP
        # will fail with ConnectError, which is fine for this test
        assert client._enforce_allowlist is False

    def test_allowed_host_not_blocked(self):
        """When host is in allowlist, no error chunk should be yielded at the gate."""
        config = Config(
            endpoint="https://api.openai.com/v1",
            allowed_hosts=["api.openai.com"],
            enforce_allowlist=True,
        )
        client = ChatClient(config)
        # The validation gate itself shouldn't block; the subsequent HTTP call
        # will fail (no real server), but that's a different error
        messages = [{"role": "user", "content": "hi"}]
        chunks = list(client.stream_chat(messages))
        # Should get an error, but NOT a BlockedHostError
        for chunk in chunks:
            if chunk.error:
                assert "not in the allowed hosts" not in chunk.error


"""Tests for network.py — host allowlisting and endpoint validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from network import (
    BlockedHostError,
    extract_host,
    is_host_allowed,
    validate_endpoint,
)


# ── extract_host ───────────────────────────────────────────────


class TestExtractHost:
    """Test URL host extraction."""

    def test_standard_url(self):
        assert extract_host("https://api.openai.com/v1") == "api.openai.com"

    def test_url_with_port(self):
        assert extract_host("http://localhost:11434/v1") == "localhost"

    def test_ipv4(self):
        assert extract_host("http://192.168.1.100:8080/v1") == "192.168.1.100"

    def test_ipv6_bracketed(self):
        assert extract_host("http://[::1]:8080/v1") == "::1"

    def test_url_with_path(self):
        assert extract_host("https://api.groq.com/openai/v1/chat/completions") == "api.groq.com"

    def test_no_path(self):
        assert extract_host("https://api.openai.com") == "api.openai.com"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError):
            extract_host("")

    def test_bare_string_raises(self):
        with pytest.raises(ValueError):
            extract_host("not-a-url")


# ── is_host_allowed ───────────────────────────────────────────


class TestIsHostAllowed:
    """Test host matching against allowlists."""

    def test_exact_match(self):
        assert is_host_allowed("api.openai.com", ["api.openai.com"]) is True

    def test_exact_match_case_insensitive(self):
        assert is_host_allowed("API.OpenAI.com", ["api.openai.com"]) is True

    def test_not_in_list(self):
        assert is_host_allowed("evil.com", ["api.openai.com"]) is False

    def test_wildcard_subdomain(self):
        assert is_host_allowed("us-east.api.openai.com", ["*.api.openai.com"]) is True

    def test_wildcard_base_match(self):
        """*.example.com should also match example.com itself (just the base)."""
        # The base domain after stripping *. is "api.openai.com"
        assert is_host_allowed("api.openai.com", ["*.api.openai.com"]) is True

    def test_wildcard_no_match(self):
        assert is_host_allowed("evil.com", ["*.openai.com"]) is False

    def test_localhost(self):
        assert is_host_allowed("localhost", ["localhost"]) is True

    def test_ip_exact(self):
        assert is_host_allowed("192.168.1.100", ["192.168.1.100"]) is True

    def test_cidr_match(self):
        assert is_host_allowed("192.168.1.100", ["192.168.1.0/24"]) is True

    def test_cidr_no_match(self):
        assert is_host_allowed("10.0.0.1", ["192.168.1.0/24"]) is False

    def test_cidr_with_hostname_skips(self):
        """CIDR matching should gracefully skip non-IP hosts."""
        assert is_host_allowed("api.openai.com", ["192.168.1.0/24"]) is False

    def test_empty_allowlist(self):
        assert is_host_allowed("anything.com", []) is False

    def test_multiple_patterns(self):
        hosts = ["localhost", "api.openai.com", "*.groq.com"]
        assert is_host_allowed("api.openai.com", hosts) is True
        assert is_host_allowed("us.api.groq.com", hosts) is True
        assert is_host_allowed("localhost", hosts) is True
        assert is_host_allowed("evil.com", hosts) is False

    def test_empty_strings_ignored(self):
        assert is_host_allowed("localhost", ["", "  ", "localhost"]) is True

    def test_ipv6_cidr(self):
        assert is_host_allowed("::1", ["::1/128"]) is True


# ── validate_endpoint ─────────────────────────────────────────


class TestValidateEndpoint:
    """Test the main validation gate."""

    def test_allowed_host_passes(self):
        """Should not raise when host is allowed."""
        validate_endpoint(
            "https://api.openai.com/v1/chat/completions",
            ["api.openai.com"],
            enforce=True,
        )

    def test_blocked_host_raises(self):
        with pytest.raises(BlockedHostError) as exc_info:
            validate_endpoint(
                "https://evil.com/v1/chat/completions",
                ["api.openai.com"],
                enforce=True,
            )
        assert "evil.com" in str(exc_info.value)

    def test_not_enforced_always_passes(self):
        """When enforce=False, even unlisted hosts pass."""
        validate_endpoint(
            "https://evil.com/v1/chat/completions",
            ["api.openai.com"],
            enforce=False,
        )

    def test_not_enforced_empty_list_passes(self):
        validate_endpoint(
            "https://anything.com/v1",
            [],
            enforce=False,
        )

    def test_enforced_empty_list_blocks(self):
        """Empty allowlist + enforce = block everything."""
        with pytest.raises(BlockedHostError):
            validate_endpoint(
                "https://api.openai.com/v1",
                [],
                enforce=True,
            )

    def test_wildcard_in_validation(self):
        validate_endpoint(
            "https://us-east.api.groq.com/openai/v1",
            ["*.api.groq.com"],
            enforce=True,
        )


# ── BlockedHostError ──────────────────────────────────────────


class TestBlockedHostError:
    """Test error message formatting."""

    def test_error_message_contains_host(self):
        err = BlockedHostError("evil.com", "https://evil.com/v1")
        assert "evil.com" in str(err)

    def test_error_message_contains_endpoint(self):
        err = BlockedHostError("evil.com", "https://evil.com/v1")
        assert "https://evil.com/v1" in str(err)

    def test_error_attributes(self):
        err = BlockedHostError("evil.com", "https://evil.com/v1")
        assert err.host == "evil.com"
        assert err.endpoint == "https://evil.com/v1"

"""
Network security module for CLIAI.

Validates outbound connections against a configurable host allowlist.
Supports exact matches, wildcard subdomains, and CIDR notation.
"""

import ipaddress
import re
from urllib.parse import urlparse


class BlockedHostError(Exception):
    """Raised when a connection to a non-allowed host is attempted."""

    def __init__(self, host: str, endpoint: str):
        self.host = host
        self.endpoint = endpoint
        super().__init__(
            f"Blocked: host '{host}' is not in the allowed hosts list.\n"
            f"Endpoint: {endpoint}\n"
            f"Configure 'security.allowed_hosts' in your config file "
            f"or set CLIAI_ALLOWED_HOSTS to permit this host."
        )


def extract_host(url: str) -> str:
    """
    Extract the hostname from an endpoint URL.

    Handles standard URLs, IPv4, IPv6 (bracketed), and ports.

    Args:
        url: Full endpoint URL (e.g. https://api.openai.com/v1).

    Returns:
        The hostname or IP address without port or path.

    Raises:
        ValueError: If the URL cannot be parsed.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise ValueError(f"Cannot extract host from URL: {url}")
    return host


def is_host_allowed(host: str, allowed_hosts: list[str]) -> bool:
    """
    Check if a host is permitted by the allowlist.

    Supports:
        - Exact match: "api.openai.com"
        - Wildcard subdomains: "*.openai.com" matches "api.openai.com"
        - CIDR ranges: "192.168.1.0/24" matches "192.168.1.100"
        - Bare IPs: "127.0.0.1"

    Args:
        host: The hostname or IP to check.
        allowed_hosts: List of allowed host patterns.

    Returns:
        True if the host is allowed, False otherwise.
    """
    if not allowed_hosts:
        return False

    host_lower = host.lower()

    for pattern in allowed_hosts:
        pattern = pattern.strip()
        if not pattern:
            continue

        pattern_lower = pattern.lower()

        # Exact match
        if host_lower == pattern_lower:
            return True

        # Wildcard subdomain match: *.example.com
        if pattern_lower.startswith("*."):
            suffix = pattern_lower[1:]  # ".example.com"
            if host_lower.endswith(suffix) or host_lower == pattern_lower[2:]:
                return True

        # CIDR range match
        if "/" in pattern:
            try:
                network = ipaddress.ip_network(pattern, strict=False)
                host_ip = ipaddress.ip_address(host)
                if host_ip in network:
                    return True
            except (ValueError, TypeError):
                # Not a valid CIDR or host isn't an IP
                pass

    return False


def validate_endpoint(
    url: str,
    allowed_hosts: list[str],
    enforce: bool,
) -> None:
    """
    Validate that an endpoint URL targets an allowed host.

    This is the primary security gate — call before every HTTP request.

    Args:
        url: The full endpoint URL to validate.
        allowed_hosts: List of allowed host patterns.
        enforce: Whether to enforce the allowlist (if False, always passes).

    Raises:
        BlockedHostError: If the host is not allowed and enforcement is on.
        ValueError: If the URL is malformed.
    """
    if not enforce:
        return

    host = extract_host(url)

    if not is_host_allowed(host, allowed_hosts):
        raise BlockedHostError(host, url)

"""
Redaction engine for CLIAI.

Provides reversible masking of sensitive data (IPs, hostnames, UUIDs,
emails, API keys, etc.) using regex patterns and user-defined terms.
Maintains a consistent mapping so the same value always gets the same placeholder.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Redaction:
    """A single redaction: original text → placeholder."""
    original: str
    placeholder: str
    category: str
    auto: bool = True  # True = regex-detected, False = manually added


class Redactor:
    """
    Reversible redaction engine.

    Detects sensitive patterns via regex, allows user-defined terms,
    and maintains a bidirectional mapping for de-masking responses.
    """

    def __init__(self, user_terms: Optional[dict[str, str]] = None):
        """
        Args:
            user_terms: Dict of {sensitive_value: placeholder} from config.
        """
        self._user_terms: dict[str, str] = user_terms or {}
        # Persistent mapping across the session: original → placeholder
        self._mapping: dict[str, str] = {}
        # Reverse mapping: placeholder → original
        self._reverse: dict[str, str] = {}
        # Counters per category for generating unique placeholders
        self._counters: dict[str, int] = {}

    # ── Public API ──────────────────────────────────────────────

    def redact(self, text: str) -> tuple[str, list[Redaction]]:
        """
        Scan text and apply all redactions.

        Returns:
            (redacted_text, list_of_redactions_applied)
        """
        redactions: list[Redaction] = []

        #####
        ## Gather all redactions first before applying:
        #####

        # Phase 0: Re-apply any previously seen mappings
        for original, placeholder in self._mapping.items():
            if original in text:
                redactions.append(Redaction(original, placeholder, "cached"))

        # Phase 1: User-defined terms (highest priority, exact match)
        for term, placeholder in self._user_terms.items():
            if term in text:
                if term not in self._mapping:
                    self._mapping[term] = placeholder
                    self._reverse[placeholder] = term
                    redactions.append(Redaction(term, placeholder, "user_term"))

        # Phase 2: Regex patterns (ordered to avoid overlapping matches)
        for category, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(0)
                # Skip if already mapped
                if value in self._mapping:
                    continue
                # Skip very short matches that are likely false positives
                if len(value) < 4 and category not in ("ipv4",):
                    continue
                placeholder = self._get_or_create_placeholder(value, category)
                redactions.append(Redaction(value, placeholder, category))

        #####
        ## Apply all redactions, longest first to avoid partial replacements
        #####
        redacted_text = text
        for r in sorted(redactions, key=lambda x: len(x.original), reverse=True):
            redacted_text = redacted_text.replace(r.original, r.placeholder)

        #####
        ## Deduplicate redaction list (same original may appear multiple times)
        #####
        seen = set()
        unique_redactions = []
        for r in redactions:
            if r.original not in seen:
                seen.add(r.original)
                unique_redactions.append(r)

        return redacted_text, unique_redactions

    def unredact(self, text: str) -> str:
        """
        Replace all placeholders back with original values.
        Used for de-masking LLM responses.
        """
        result = text
        # Sort by placeholder length (longest first) to avoid partial replacements
        for placeholder, original in sorted(
            self._reverse.items(), key=lambda x: len(x[0]), reverse=True
        ):
            result = result.replace(placeholder, original)
        return result

    def add_manual_redaction(self, original: str, text: str) -> tuple[str, Redaction]:
        """
        Manually add a redaction for text the regex didn't catch.

        Args:
            original: The sensitive text to redact.
            text: The current (possibly already redacted) text.

        Returns:
            (updated_text, redaction_applied)
        """
        placeholder = self._get_or_create_placeholder(original, "manual")
        redaction = Redaction(original, placeholder, "manual", auto=False)
        text = text.replace(original, placeholder)
        return text, redaction

    def remove_redaction(self, redaction: Redaction, text: str) -> str:
        """
        Remove a redaction (unredact a specific item).

        Args:
            redaction: The Redaction to undo.
            text: The current redacted text.

        Returns:
            Updated text with the redaction reversed.
        """
        text = text.replace(redaction.placeholder, redaction.original)
        # Don't remove from persistent mapping — it may be used later
        return text

    def get_mapping_table(self) -> list[tuple[str, str, str]]:
        """
        Get the current mapping table for display.

        Returns:
            List of (original, placeholder, category) tuples.
        """
        result = []
        seen = set()
        for original, placeholder in self._mapping.items():
            if original not in seen:
                seen.add(original)
                result.append((original, placeholder, ""))
        return result

    def get_system_hint(self) -> str:
        """
        Generate a system message instructing the LLM to preserve
        redaction placeholder tokens in its response.
        """
        tokens = ", ".join(sorted(set(self._mapping.values())))
        return (
            "The user's message contains redacted placeholder tokens "
            f"such as: {tokens}. "
            "These tokens represent sensitive data that has been masked. "
            "When referring to these values in your response, use the "
            "EXACT placeholder tokens as given (e.g. [IP_1], [HOST_1]). "
            "Do NOT try to guess or reconstruct the original values."
        )

    # ── Internal ────────────────────────────────────────────────

    def _get_or_create_placeholder(self, value: str, category: str) -> str:
        """Get existing placeholder or create a new one for this value."""
        if value in self._mapping:
            return self._mapping[value]

        # Generate a new placeholder
        tag = _CATEGORY_TAGS.get(category, category.upper())
        count = self._counters.get(tag, 0) + 1
        self._counters[tag] = count
        placeholder = f"[{tag}_{count}]"

        self._mapping[value] = placeholder
        self._reverse[placeholder] = value
        return placeholder


# ── Category Tags ───────────────────────────────────────────────

_CATEGORY_TAGS = {
    "ipv4": "IP",
    "ipv6": "IP",
    "cidr": "CIDR",
    "uuid": "UUID",
    "fqdn": "HOST",
    "url": "URL",
    "email": "EMAIL",
    "aws_arn": "ARN",
    "aws_resource": "AWS",
    "api_key": "KEY",
    "mac_address": "MAC",
    "filepath_user": "PATH",
    "manual": "REDACTED",
    "user_term": "TERM",
    "connection_string": "CONN",
    "private_key": "PRIVKEY",
    "jwt": "JWT",
    "bearer_token": "BEARER",
    "ssh_fingerprint": "SSHFP",
    "cert_fingerprint": "CERTFP",
    "git_remote": "GITREMOTE",
    "user_at_host": "LOGIN",
    "env_secret": "ENV",
    "docker_image": "DOCKER",
    "k8s_pod": "K8SPOD",
    "hex_string": "HEX",
    "ssh_pubkey": "SSHKEY",
    "github_token": "GHTOKEN",
    "gitlab_token": "GLTOKEN",
    "slack_token": "SLACK",
    "stripe_key": "STRIPE",
    "sendgrid_key": "SENDGRID",
    "twilio_key": "TWILIO",
    "openai_key": "OPENAI",
    "pypi_token": "PYPI",
    "npm_token": "NPM",
    "telegram_token": "TELEGRAM",
    "mailchimp_key": "MAILCHIMP",
    "square_key": "SQUARE",
    "discord_token": "DISCORD",
}

# ── Regex Patterns ──────────────────────────────────────────────
# Order matters: more specific patterns first to avoid partial matches.
# Each entry: (category_name, compiled_regex)

_PATTERNS = [
    # SSH public key blobs (known_hosts, authorized_keys, key pastes)
    (
        "ssh_pubkey",
        re.compile(
            r"(?:ssh-(?:rsa|ed25519|dss)|ecdsa-sha2-nistp(?:256|384|521))\s+"
            r"AAAA[A-Za-z0-9+/=]{40,}"
        ),
    ),
    # Private key blocks (PEM-encoded)
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
        ),
    ),
    # JWT tokens (three base64url dot-separated segments)
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
        ),
    ),
    # Bearer token in auth headers
    (
        "bearer_token",
        re.compile(
            r"(?:Authorization:\s*Bearer\s+|Bearer\s+)[A-Za-z0-9._\-]{20,}"
        ),
    ),
    # SSH key fingerprints (SHA256: or MD5: prefixed)
    (
        "ssh_fingerprint",
        re.compile(
            r"(?:SHA256:[A-Za-z0-9+/=]{32,}|MD5:(?:[0-9a-fA-F]{2}:){15}[0-9a-fA-F]{2})"
        ),
    ),
    # Certificate fingerprints (SHA1:/SHA256: with colon-hex)
    (
        "cert_fingerprint",
        re.compile(
            r"(?:SHA1|SHA256|sha1|sha256):(?:[0-9a-fA-F]{2}:){15,31}[0-9a-fA-F]{2}"
        ),
    ),
    # AWS ARNs
    (
        "aws_arn",
        re.compile(r"arn:aws[a-z\-]*:[a-z0-9\-]+:[a-z0-9\-]*:\d{0,12}:[a-zA-Z0-9\-_/:.]+"),
    ),
    # Connection strings (jdbc, mongodb, postgres, mysql, redis, etc.)
    (
        "connection_string",
        re.compile(
            r"(?:jdbc:|mongodb(?:\+srv)?://|postgres(?:ql)?://|mysql://|redis://|amqp://)"
            r"[^\s\"'`]+"
        ),
    ),
    # Git remote URLs (SSH and HTTPS forms)
    (
        "git_remote",
        re.compile(
            r"git@[a-zA-Z0-9.\-]+:[a-zA-Z0-9._\-]+/[a-zA-Z0-9._\-]+(?:\.git)?"
        ),
    ),
    # ── Platform-specific tokens (sourced from detect-secrets) ──
    # GitHub tokens
    (
        "github_token",
        re.compile(
            r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"
        ),
    ),
    # GitLab tokens
    (
        "gitlab_token",
        re.compile(
            r"\b(?:glpat|glptt|gldt|glimt|gloas|glagent)-[A-Za-z0-9_\-]{20,}\b"
        ),
    ),
    # Slack tokens & webhook URLs
    (
        "slack_token",
        re.compile(
            r"(?:xox(?:a|b|p|o|s|r)-(?:\d+-)+[a-z0-9]+"
            r"|https://hooks\.slack\.com/services/T[a-zA-Z0-9_]+/B[a-zA-Z0-9_]+/[a-zA-Z0-9_]+)"
        ),
    ),
    # Stripe keys
    (
        "stripe_key",
        re.compile(r"\b(?:r|s)k_live_[0-9a-zA-Z]{24}\b"),
    ),
    # SendGrid keys
    (
        "sendgrid_key",
        re.compile(r"\bSG\.[a-zA-Z0-9_\-]{22}\.[a-zA-Z0-9_\-]{43}\b"),
    ),
    # Twilio keys
    (
        "twilio_key",
        re.compile(r"\b(?:AC|SK)[a-z0-9]{32}\b"),
    ),
    # OpenAI keys
    (
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9-_]*[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}\b"),
    ),
    # PyPI tokens
    (
        "pypi_token",
        re.compile(r"\bpypi-AgE[A-Za-z0-9-_]{70,}\b"),
    ),
    # npm tokens
    (
        "npm_token",
        re.compile(r"\/\/.+\/:_authToken=\s*(?:npm_.+|[A-Fa-f0-9-]{36})"),
    ),
    # Telegram bot tokens
    (
        "telegram_token",
        re.compile(r"\b\d{8,10}:[0-9A-Za-z_-]{35}\b"),
    ),
    # Mailchimp keys
    (
        "mailchimp_key",
        re.compile(r"\b[0-9a-z]{32}-us[0-9]{1,2}\b"),
    ),
    # Square OAuth tokens
    (
        "square_key",
        re.compile(r"\bsq0csp-[0-9A-Za-z\\\-_]{43}\b"),
    ),
    # Discord bot tokens
    (
        "discord_token",
        re.compile(r"\b[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27,}\b"),
    ),
    # AWS access key IDs
    (
        "api_key",
        re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    ),
    # Generic API key/token patterns (catch-all)
    (
        "api_key",
        re.compile(
            r"(?:sk-[a-zA-Z0-9]{20,}|gsk_[a-zA-Z0-9]{20,}|"
            r"token_[a-zA-Z0-9]{16,})"
        ),
    ),
    # Environment variable secrets (KEY=value patterns for common secret names)
    (
        "env_secret",
        re.compile(
            r"\b(?:PASSWORD|SECRET|TOKEN|API_KEY|PRIVATE_KEY|AUTH|CREDENTIALS"
            r"|DB_PASS|AWS_SECRET_ACCESS_KEY|DATABASE_URL)"
            r"=[^\s\"'`]{1,}"
        ),
    ),
    # Docker image references (registry/image:tag patterns)
    (
        "docker_image",
        re.compile(
            r"\b(?:[a-zA-Z0-9.\-]+\.(?:io|com|net|org|dev|cloud|azurecr|ecr|gcr)"
            r"(?::\d+)?/)"
            r"[a-zA-Z0-9._\-]+(?:/[a-zA-Z0-9._\-]+)*"
            r"(?::[a-zA-Z0-9._\-]+)?(?:@sha256:[0-9a-fA-F]{64})?"
        ),
    ),
    # Kubernetes pod names (deployment-name-replicaset-hash-pod-hash)
    (
        "k8s_pod",
        re.compile(
            r"\b[a-z][a-z0-9\-]+-[a-z0-9]{8,10}-[a-z0-9]{5}\b"
        ),
    ),
    # UUIDs
    (
        "uuid",
        re.compile(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        ),
    ),
    # Email addresses
    (
        "email",
        re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ),
    # user@host patterns (SSH-style, must not match emails — no TLD)
    (
        "user_at_host",
        re.compile(
            r"\b[a-zA-Z][a-zA-Z0-9._\-]*@[a-zA-Z][a-zA-Z0-9.\-]*[a-zA-Z0-9]"
            r"(?=\s|$|[;:,\)\]\}])"
        ),
    ),
    # URLs (with protocol) — must come before FQDN matching
    (
        "url",
        re.compile(r"https?://[^\s\"'`<>\]\)]+"),
    ),
    # MAC addresses
    (
        "mac_address",
        re.compile(r"(?:[0-9a-fA-F]{1,2}[:\-]){5}[0-9a-fA-F]{1,2}"),
    ),
    # CIDR notation
    (
        "cidr",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d\d?)/\d{1,2}\b"
        ),
    ),
    # IPv4 addresses
    (
        "ipv4",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1?\d\d?)\b"
        ),
    ),
    # IPv6 addresses (full, compressed, and with zone IDs)
    (
        "ipv6",
        re.compile(
            # Full form: 8 groups
            r"(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"
            # Compressed with :: (captures both sides)
            r"|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})*"
            # :: at start
            r"|::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}"
            # Trailing :: (link-local etc with no suffix — rare standalone)
            r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"
        ),
    ),
    # AWS resource IDs (i-, sg-, vpc-, subnet-, vol-, snap-, ami-, etc.)
    (
        "aws_resource",
        re.compile(
            r"\b(?:i|sg|vpc|subnet|vol|snap|ami|eni|igw|rtb|acl|nat|eip|"
            r"pcx|cgw|vgw|dopt)-[0-9a-fA-F]{8,17}\b"
        ),
    ),
    # Long hex strings (git SHAs, Docker container IDs — 12+ hex chars)
    (
        "hex_string",
        re.compile(r"\b[0-9a-fA-F]{12,64}\b"),
    ),
    # File paths containing usernames (/home/*, /Users/*, C:\Users\*)
    (
        "filepath_user",
        re.compile(
            r"(?:/(?:home|Users)/[a-zA-Z0-9._\-]+(?:/[^\s\"'`]*)?|"
            r"C:\\Users\\[a-zA-Z0-9._\-]+(?:\\[^\s\"'`]*)?)"
        ),
    ),
    # FQDNs (hostname.domain.tld patterns — at least 2 dots or
    # hostname-like patterns with common infra TLDs)
    (
        "fqdn",
        re.compile(
            r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+"
            r"(?:com|net|org|io|dev|app|aws|cloud|internal|local|"
            r"corp|lan|intra|private|srv|infra|k8s|svc|cluster)\b"
        ),
    ),
]

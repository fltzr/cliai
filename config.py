"""
Configuration system for CLIAI.

Loads settings from YAML config file → environment variables → CLI flags.
Supports named profiles for quick endpoint switching.
"""

import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml
from platformdirs import user_config_dir, user_data_dir


APP_NAME = "cliai"
CONFIG_DIR = Path(user_config_dir(APP_NAME))
DATA_DIR = Path(user_data_dir(APP_NAME))
CONFIG_FILE = CONFIG_DIR / "config.yaml"
SESSIONS_DIR = DATA_DIR / "sessions"

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Resolved configuration for a chat session."""

    endpoint: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "llama3"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    system_prompt: str = ""
    stream: bool = True
    profile_name: str = "default"
    redact_terms: dict = field(default_factory=dict)
    allowed_hosts: list = field(default_factory=list)
    enforce_allowlist: bool = False

    def display_endpoint(self) -> str:
        """Return a display-safe version of the endpoint."""
        return self.endpoint.rstrip("/")

    def display_key(self) -> str:
        """Return a masked version of the API key."""
        if not self.api_key:
            return "(none)"
        if len(self.api_key) <= 8:
            return "****"
        return self.api_key[:4] + "…" + self.api_key[-4:]

    def to_dict(self) -> dict:
        """Serialize to dict (for display/debug)."""
        d = asdict(self)
        d["api_key"] = self.display_key()
        d.pop("redact_terms", None)  # Don't display in info
        return d


DEFAULT_CONFIG_YAML = """\
# CLIAI Configuration
# Define profiles for different endpoints/models.
# Use --profile <name> to select one, or set CLIAI_PROFILE env var.

default_profile: default

profiles:
  default:
    endpoint: "http://localhost:11434/v1"
    api_key: ""
    model: "llama3"
    temperature: 0.7
    # max_tokens: null
    # system_prompt: ""
    stream: true

  # Example: OpenAI
  # openai:
  #   endpoint: "https://api.openai.com/v1"
  #   api_key: "sk-..."
  #   model: "gpt-4o-mini"
  #   temperature: 0.7
  #   stream: true

  # Example: Groq
  # groq:
  #   endpoint: "https://api.groq.com/openai/v1"
  #   api_key: "gsk_..."
  #   model: "llama-3.3-70b-versatile"
  #   temperature: 0.7
  #   stream: true

# Redaction: define sensitive terms to always mask.
# These are applied BEFORE regex patterns.
redact:
  terms: {}
  # Example:
  #   "acme-corp": "[COMPANY]"
  #   "Project Falcon": "[PROJECT]"
  #   "prod-west-2": "[CLUSTER_1]"

# Security: network-level controls for outbound connections.
security:
  enforce_allowlist: false
  allowed_hosts: []
  # Example (uncomment and set enforce_allowlist: true):
  #   - "localhost"
  #   - "api.openai.com"
  #   - "*.groq.com"
  #   - "192.168.1.0/24"
"""


def _load_yaml_config() -> dict:
    """Load the YAML config file. Returns empty dict if not found."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
        return data
    except yaml.YAMLError as e:
        print(f"Warning: Failed to parse {CONFIG_FILE}: {e}", file=sys.stderr)
        return {}


def _get_profile_from_yaml(yaml_data: dict, profile_name: Optional[str]) -> dict:
    """Extract a profile dict from the YAML data."""
    profiles = yaml_data.get("profiles", {})
    if not profiles:
        return {}

    # Determine which profile to use
    name = profile_name or yaml_data.get("default_profile", "default")
    profile = profiles.get(name)

    if profile is None:
        available = ", ".join(profiles.keys())
        print(
            f"Warning: Profile '{name}' not found. Available: {available}",
            file=sys.stderr,
        )
        return {}

    return profile or {}


def _apply_env_vars(config: dict) -> dict:
    """Override config values with CLIAI_* environment variables."""
    env_map = {
        "CLIAI_ENDPOINT": "endpoint",
        "CLIAI_API_KEY": "api_key",
        "CLIAI_MODEL": "model",
        "CLIAI_TEMPERATURE": "temperature",
        "CLIAI_MAX_TOKENS": "max_tokens",
        "CLIAI_SYSTEM_PROMPT": "system_prompt",
        "CLIAI_STREAM": "stream",
    }

    for env_var, config_key in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            # Type coercion
            if config_key == "temperature":
                try:
                    val = float(val)
                except ValueError:
                    continue
            elif config_key == "max_tokens":
                try:
                    val = int(val)
                except ValueError:
                    continue
            elif config_key == "stream":
                val = val.lower() in ("true", "1", "yes")
            config[config_key] = val

    # Comma-separated allowed hosts
    allowed = os.environ.get("CLIAI_ALLOWED_HOSTS")
    if allowed is not None:
        config["allowed_hosts"] = [h.strip() for h in allowed.split(",") if h.strip()]

    # Enforce allowlist toggle
    enforce = os.environ.get("CLIAI_ENFORCE_ALLOWLIST")
    if enforce is not None:
        config["enforce_allowlist"] = enforce.lower() in ("true", "1", "yes")

    return config


def _apply_cli_overrides(config: dict, overrides: dict) -> dict:
    """Apply CLI flag overrides (only non-None values)."""
    for key, val in overrides.items():
        if val is not None:
            config[key] = val
    return config


def load_config(
    profile: Optional[str] = None,
    cli_overrides: Optional[dict] = None,
) -> Config:
    """
    Load configuration with precedence: YAML < env vars < CLI flags.

    Args:
        profile: Profile name to load from YAML config.
        cli_overrides: Dict of CLI flag overrides (non-None values only).

    Returns:
        Resolved Config instance.
    """
    # Start with defaults
    config_dict = asdict(Config())

    # Layer 1: YAML config
    yaml_data = _load_yaml_config()
    yaml_profile = _get_profile_from_yaml(yaml_data, profile)
    config_dict.update({k: v for k, v in yaml_profile.items() if v is not None})

    # Layer 2: Environment variables
    config_dict = _apply_env_vars(config_dict)

    # Layer 3: CLI flags
    if cli_overrides:
        config_dict = _apply_cli_overrides(config_dict, cli_overrides)

    # Track which profile we loaded
    config_dict["profile_name"] = (
        profile
        or os.environ.get("CLIAI_PROFILE")
        or yaml_data.get("default_profile", "default")
    )

    # Extract redact terms from global YAML config (not per-profile)
    redact_config = yaml_data.get("redact", {})
    config_dict["redact_terms"] = redact_config.get("terms", {}) or {}

    # Extract security settings from global YAML config
    security_config = yaml_data.get("security", {})
    if security_config.get("allowed_hosts"):
        config_dict["allowed_hosts"] = security_config["allowed_hosts"]
    if "enforce_allowlist" in security_config:
        config_dict["enforce_allowlist"] = bool(security_config["enforce_allowlist"])

    # Build config, filtering out unknown keys
    valid_keys = {f.name for f in Config.__dataclass_fields__.values()}
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}

    return Config(**filtered)


def create_default_config() -> Path:
    """Create the default config file if it doesn't exist. Returns path."""
    if not CONFIG_FILE.exists():
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(DEFAULT_CONFIG_YAML)
    return CONFIG_FILE


def list_profiles() -> dict[str, dict]:
    """List all available profiles from the config file."""
    yaml_data = _load_yaml_config()
    return yaml_data.get("profiles", {})

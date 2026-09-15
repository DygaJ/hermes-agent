"""``hermes -m <model> --provider bedrock`` must route the -m model, not config's default.

Bedrock's resolver picks the wire per model (OpenAI GPT-5.x -> Mantle Responses, Claude ->
AnthropicBedrock, the rest -> Converse). The CLI startup credential pass used to call
``resolve_runtime_provider`` without ``target_model``, so on a Claude-default install
``-m openai.gpt-5.6-sol --provider bedrock`` was routed for the Claude default and hit Converse,
which rejects bare Mantle-only model ids. The one-shot path (``hermes -z``) and the in-session
``/model`` switch already passed the target model; startup now matches them.
"""

from __future__ import annotations

import pytest

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

MANTLE = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
CONVERSE = "https://bedrock-runtime.us-east-1.amazonaws.com"


class _RuntimeCLI(CLIAgentSetupMixin):
    def __init__(self, *, model: str, provider: str):
        self.model = model
        self.requested_provider = provider
        self.provider = provider
        self.api_key = None
        self.base_url = None
        self.api_mode = "chat_completions"
        self.acp_command = None
        self.acp_args = []
        self.agent = None
        self._fallback_model = []
        self._explicit_api_key = None
        self._explicit_base_url = None
        self._credential_pool = None
        self.service_tier = None
        self.tool_progress_mode = "off"

    def _normalize_model_for_provider(self, _provider: str) -> bool:
        return False


@pytest.fixture
def claude_default_on_bedrock(monkeypatch, tmp_path):
    """config.yaml whose default model is Claude on Bedrock, plus a fake static key pair so the
    resolver's credential check passes without touching botocore's chain."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: us.anthropic.claude-opus-5\n  provider: bedrock\nbedrock:\n  region: us-east-1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-secret")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    return tmp_path


def test_startup_routes_cli_model_to_mantle_not_config_default(claude_default_on_bedrock):
    cli = _RuntimeCLI(model="openai.gpt-5.6-sol", provider="bedrock")
    assert cli._ensure_runtime_credentials() is True
    assert cli.provider == "bedrock"
    assert cli.api_mode == "codex_responses"
    assert cli.base_url == MANTLE


def test_startup_keeps_claude_default_on_anthropic_bedrock_wire(claude_default_on_bedrock):
    cli = _RuntimeCLI(model="", provider="bedrock")
    assert cli._ensure_runtime_credentials() is True
    assert cli.api_mode == "anthropic_messages"
    assert cli.base_url == CONVERSE


def test_readiness_probe_uses_cli_model(claude_default_on_bedrock):
    cli = _RuntimeCLI(model="openai.gpt-5.6-sol", provider="bedrock")
    assert cli._runtime_credentials_ready() is True

"""A Slack message the gateway already admitted, delivered again, is never answered with an error.

After a gateway restart the adapter's in-memory guards (``_dedup``, ``_processed_message_ts``) are
empty, so Slack's ``message_changed`` for an answered message (its thread gaining replies, an
unfurl, a user edit), a Socket Mode redelivery, or a slack-recovery re-fetch rebuilds the message
under its admitted ts with a new local ``timestamp``. Native admission refuses that as a changed
replay (``admission_conflict``), and the chat used to get "Sorry, I encountered an error
(RuntimeStoreError)". The copy must not run, as the old message or as a new one, and must get no
reply. Real Slack adapter, base dispatch, runner and SQLite admission; Slack's API is stubbed."""
import asyncio
import importlib
import sys
from importlib.machinery import PathFinder
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_installed_package(name):
    if PathFinder.find_spec(name) is None:
        return None
    prefix = f"{name}."
    displaced = {
        m: sys.modules.pop(m)
        for m in tuple(sys.modules)
        if (m == name or m.startswith(prefix)) and not isinstance(sys.modules[m], ModuleType)
    }
    try:
        return importlib.import_module(name)
    except ImportError:
        sys.modules.update(displaced)
        return None


_load_installed_package("slack_bolt")
_load_installed_package("slack_sdk")

_slack_mod = importlib.import_module("plugins.platforms.slack.adapter")

TEAM = "T0638SY6NPQ"
BOT = "U0C8QALEB5W"
USER = "U07RKANQZDF"
DM = "D0C85CZK490"
TS = "1793819016.301339"
TEXT = "Show me the GUI at PR #388, compared with the base."


@pytest.fixture(autouse=True)
def _pin_legacy_assistant_threads_api():
    """The mocked client would make the Agent Sessions probe truthy; pin the legacy path."""
    prev = _slack_mod._AGENT_SESSIONS_SUPPORTED
    _slack_mod._AGENT_SESSIONS_SUPPORTED = False
    yield
    _slack_mod._AGENT_SESSIONS_SUPPORTED = prev


def _message(text=TEXT, **extra):
    return {"type": "message", "user": USER, "text": text, "ts": TS, "team": TEAM,
            "client_msg_id": "c1", **extra}


def _changed(message, event_ts):
    """Slack's ``message_changed`` for the message: a new event ts, the message under its own ts."""
    return {"type": "message", "subtype": "message_changed", "hidden": True, "channel": DM,
            "channel_type": "im", "team": TEAM, "ts": event_ts, "event_ts": event_ts, "message": message}


@pytest.mark.asyncio
async def test_admitted_slack_message_delivered_again_after_restart_runs_once_and_gets_no_error(
        tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from hermes_state_runtime import list_session_admissions
    monkeypatch.setenv("SLACK_ALLOWED_USERS", USER)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="slack")
    sent, turns = [], []

    async def send(chat_id, content, **_):
        sent.append(content)
        return SendResult(success=True, message_id=f"1793821326.{len(sent):06d}")

    async def turn(event, *_args, **_kwargs):
        turns.append(event.text)
        execution_result.get()["result"] = {"final_response": "answer", "messages": []}
        return "answer"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)

    def connect():
        """One gateway process's Slack adapter: a restart builds a new one with empty guards."""
        adapter = _slack_mod.SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
        adapter._app = MagicMock()
        adapter._app.client = AsyncMock()
        adapter._app.client.users_info = AsyncMock(return_value={
            "user": {"is_bot": False, "profile": {"display_name": "Dylan"}, "real_name": "Dylan"}})
        adapter._bot_user_id = BOT
        adapter._running = True
        monkeypatch.setattr(adapter, "send", send)
        adapter.gateway_runner, runner.adapters = runner, {Platform.SLACK: adapter}
        runner._wire_adapter_handlers(adapter)
        return adapter

    async def deliver(adapter, event):
        await adapter._handle_slack_message(event, {"team_id": TEAM})
        if adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 10)

    try:
        await deliver(connect(), dict(_message(), channel=DM, channel_type="im", event_ts=TS))
        assert (turns, sent) == ([TEXT], ["answer"])

        # The gateway restarts. The bot's replies grew the message's thread, and Slack sends
        # ``message_changed`` for it with the text unchanged: the host's 2026-11-04 11:42:06 event.
        thread = _message(thread_ts=TS, reply_count=3, reply_users=[BOT], latest_reply="1793821177.880909")
        await deliver(connect(), _changed(thread, "1793821326.000100"))
        # A real edit of the answered message after another restart.
        edit = _message(text=TEXT + " Also check the flip.", thread_ts=TS,
                        edited={"user": USER, "ts": "1793821400.000000"})
        await deliver(connect(), _changed(edit, "1793821400.000200"))

        assert (turns, sent) == ([TEXT], ["answer"]), "a delivered-again message ran or was answered"
        rows = [row for sid in authority.sessions
                for row in list_session_admissions(authority.db, session_id=sid, pending_only=False)]
        assert [(row["request_id"], row["payload"]["text"]) for row in rows] == [(TS, TEXT)]
    finally:
        runner.session_store.close_all_db_handles()
        authority.db.close()


@pytest.mark.asyncio
async def test_a_conflict_without_the_admitted_request_tag_still_reaches_the_chat():
    """Only ``ADMITTED_REQUEST_CONFLICT`` proves the id was admitted. Any other ``admission_conflict``
    (two profile routes that tie, ``session_ingress_context._runtime_profile``) still raises, so the
    chat gets the turn error and gateway.log the traceback."""
    from types import SimpleNamespace

    from gateway.config import Platform
    from gateway.session_ingress import admit_message
    from hermes_state_runtime import RuntimeStoreError

    async def tied_route(_event):
        raise RuntimeStoreError("admission_conflict")

    # A ``source`` lets a mutant that drops the field check reach its log line and return None.
    event = SimpleNamespace(source=SimpleNamespace(platform=Platform.SLACK), message_id=TS)
    with pytest.raises(RuntimeStoreError, match="admission_conflict"):
        await admit_message(SimpleNamespace(admit_native=tied_route), event)

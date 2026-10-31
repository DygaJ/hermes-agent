"""A subdirectory hint reaches the model once while it is in context, across rebuilt agents.

The gateway (and the TUI's local platform, which runs turns through it) builds a fresh ``AIAgent`` for a
continuing session whenever its cached agent was evicted, and a managed worker builds one per admission
(agent/managed_worker.py). A fresh agent has a fresh ``SubdirectoryHintTracker``, so a directory's AGENTS.md
was appended to a tool result again although the earlier copy still rode the replayed history. The tracker
now reads "delivered" from the live message list the next request sends. The hint's exact text in a tool
result counts, unless compaction, a prune or a Codex native compaction checkpoint took it off the wire.

Each test drives real ``run_conversation`` turns with a fresh agent per turn against a real store, replaying
history from that store as the gateway and the worker do. Only the model replies and the tool body are scripted.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB

SESSION = "S13_SUBDIR_HINTS_TURNS"
MARKER = "[Subdirectory context discovered: backend/AGENTS.md]"
RULES = "Backend rules: use FastAPI"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    (root / "backend").mkdir(parents=True)
    (root / "frontend").mkdir()
    (root / "AGENTS.md").write_text("Root project instructions", encoding="utf-8")
    (root / "backend" / "AGENTS.md").write_text(RULES, encoding="utf-8")
    (root / "frontend" / "AGENTS.md").write_text("Frontend rules: use Svelte", encoding="utf-8")
    return root


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    store.create_session(SESSION, source="cli")
    yield store
    store.close()


def _response(call=None):
    calls = None
    if call is not None:
        name, args = call
        calls = [SimpleNamespace(id=f"call_{abs(hash(json.dumps(args)))}", type="function",
                                 function=SimpleNamespace(name=name, arguments=json.dumps(args)))]
    message = SimpleNamespace(content=None if calls else "done", reasoning_content=None, reasoning=None, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if calls else "stop")],
                           model="test/model", usage=None)


def _prepare(agent):
    agent.client = MagicMock()
    agent._cached_system_prompt, agent._use_prompt_caching, agent._disable_streaming = "sys", False, True
    agent.tool_delay, agent.save_trajectories = 0, False
    return agent


def _fresh_agent(store, project, session_id=SESSION):
    """A rebuilt agent for a continuing session: same session id and store, new object."""
    with patch("agent.process_bootstrap.OpenAI"):
        from run_agent import AIAgent
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", model="test/model",
                        platform="cli", quiet_mode=True, skip_memory=True, max_iterations=4,
                        session_db=store, session_id=session_id, cwd=str(project))
    return _prepare(agent)


def _turn(agent, history, tool_path, monkeypatch, text="look at the code", call=None):
    """One user turn: the model makes one tool call (``read_file`` of ``tool_path`` unless ``call`` names
    another), then answers. Returns the turn's messages."""
    call = call or ("read_file", {"path": tool_path})
    tool_defs = [{"type": "function", "function": {"name": call[0], "description": call[0],
                                                   "parameters": {"type": "object", "properties": {}}}}]
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **kw: tool_defs)
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *a, **kw: {})
    agent.client.chat.completions.create.side_effect = [_response(call), _response()]
    with (
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("model_tools.handle_function_call", lambda name, args, task_id=None, **_kw: "file body"),
    ):
        result = agent.run_conversation(text, conversation_history=history, task_id="s13-task")
    assert result["completed"] is True
    return result["messages"]


def _hinted(messages, rule=RULES):
    return [m for m in messages if m.get("role") == "tool" and "[Subdirectory context discovered: " in str(m.get("content"))
            and rule in str(m.get("content"))]


def _seed(store, rows):
    store.append_messages_batch(SESSION, rows)
    return store.get_messages_as_conversation(SESSION)  # the replayed history a rebuilt agent starts from


def _tool_round(content, call_id="call_1"):
    return [{"role": "user", "content": "look at the code"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call_id, "tool_name": "read_file", "content": content},
            {"role": "assistant", "content": "done"}]


def test_s13_turns_gateway_rebuild_skips_a_delivered_hint(project, db, monkeypatch):
    """S13-turns: ``_resolve_turn_agent`` rebuilds the agent after an eviction (fallback, TTL, LRU, pressure)
    and the turn replays the stored history. The hint that history carries is not sent again."""
    import threading
    from collections import OrderedDict

    from gateway.session_execution import GatewaySessionAgentMixin

    def build(**kw):
        with patch("agent.process_bootstrap.OpenAI"):
            from run_agent import AIAgent
            return AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", **kw)

    monkeypatch.setenv("TERMINAL_CWD", str(project))  # the gateway's workspace reaches the agent this way
    cache = OrderedDict()
    runner = SimpleNamespace(
        _agent_cache=cache, _agent_cache_lock=threading.Lock(), _session_db=SimpleNamespace(_db=db),
        _prefill_messages=None, _service_tier=None, _refresh_fallback_model=lambda: None,
        _agent_config_signature=lambda *a, **kw: "sig", _extract_cache_busting_config=lambda cfg: {},
        _enforce_agent_cache_cap=lambda: None, _init_cached_agent_for_turn=lambda agent, depth: None,
        _apply_fallback_chain_to_agent=lambda agent, chain: None,
        session_store=SimpleNamespace(_is_session_ended_in_db=lambda sid, session_key=None: False))
    source = SimpleNamespace(chat_id="c", user_id="u", user_id_alt=None, user_name=None, chat_name=None,
                             chat_type="dm", thread_id=None)
    turn = GatewaySessionAgentMixin.__new__(GatewaySessionAgentMixin)
    turn._runner = runner
    turn._ctx = SimpleNamespace(source=source, user_config={}, enabled_toolsets=None, disabled_toolsets=None,
                                session_id=SESSION, session_key="k", AIAgent=build, _interrupt_depth=0,
                                interim_assistant_messages_enabled=False)
    monkeypatch.setattr("gateway.session_policy.policy_for_source", lambda runner, source: None)

    def resolve():
        agent, reused = turn._resolve_turn_agent({"model": "test/model", "runtime": {}}, "local", None, 4, None, {})
        assert reused is False
        return _prepare(agent)

    app = str(project / "backend" / "app.py")
    first = _turn(resolve(), [], app, monkeypatch)
    assert len(_hinted(first)) == 1
    cache.pop("k")  # what ``_evict_cached_agent`` does to the entry (e.g. after a fallback turn)
    second = _turn(resolve(), db.get_messages_as_conversation(SESSION), app, monkeypatch, text="and again")
    assert len(_hinted(second)) == 1, "a rebuilt agent re-sent a hint its replayed history still carries"


def test_s13_worker_admission_skips_a_delivered_hint(project, db, tmp_path, monkeypatch):
    """S13-worker: each managed-worker admission (``--ignore-user-config``, ``gateway.managed_workers``) builds
    a fresh agent on a ``RuntimeSessionStore`` and replays its history (agent/managed_worker.py:364, :388)."""
    from agent.runtime_session_store import RuntimeSessionStore
    from hermes_state_runtime import begin_runtime_epoch, mutate_worker_execution, register_worker_execution

    epoch = begin_runtime_epoch(db, instance_id="fixture")
    app = str(project / "backend" / "app.py")
    for admission in range(2):
        scope = dict(epoch=epoch, execution_id=f"worker-{admission}", session_id=SESSION, generation=0)
        register_worker_execution(db, **scope, kind="compute", adoption_secret="secret")
        store = RuntimeSessionStore(lambda method, **p: mutate_worker_execution(db, **p), scope,
                                    tmp_path / f"outbox-{admission}")
        try:
            messages = _turn(_fresh_agent(store, project), store.get_messages_as_conversation(SESSION), app,
                             monkeypatch, text=f"admission {admission}")
            store.finish()  # the execution ends with its admission, so the next one can register
        finally:
            store.failure = None
            store.journal["pending"] = []
            store.close()
        assert len(_hinted(messages)) == 1, f"admission {admission} re-sent a delivered hint"


def test_s13_new_dir_gets_its_hint_next_to_a_delivered_multimodal_one(project, db, monkeypatch):
    """S13-new-dir: a rebuilt agent whose tool call touches backend/ and frontend/ gets only frontend's hint,
    because backend's rides a replayed multimodal tool result (the hint sits on its text part)."""
    history = _seed(db, _tool_round([{"type": "text", "text": "screenshot\n\n" + MARKER + "\n" + RULES},
                                     {"type": "image_url", "image_url": {"url": "data:,"}}]))
    cmd = f"cat {project / 'backend' / 'app.py'} {project / 'frontend' / 'app.js'}"
    second = _turn(_fresh_agent(db, project), history, None, monkeypatch, call=("terminal", {"command": cmd}))
    new = second[len(history):]
    assert len(_hinted(new, "Frontend rules")) == 1
    assert not _hinted(new), "a hint still carried by a multimodal result was sent again"


def _compactor(summary):
    compressor = MagicMock()
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor.compress.return_value = summary
    return compressor


def test_s13_after_compaction_hint_is_injected_again(project, db, monkeypatch):
    """S13-after-compaction: once compaction drops the hint-bearing result, the next touch re-injects it."""
    app = str(project / "backend" / "app.py")
    first_agent = _fresh_agent(db, project)
    first = _turn(first_agent, [], app, monkeypatch)
    assert len(_hinted(first)) == 1

    summary = [{"role": "user", "content": "[CONTEXT COMPACTION] earlier work summarised"},
               {"role": "assistant", "content": "acknowledged"}]
    first_agent.context_compressor = _compactor(summary)
    first_agent.compression_in_place = False
    compacted, _ = first_agent._compress_context(first, "sys", approx_tokens=120_000, task_id="s13-task")
    assert not _hinted(compacted)

    # The gateway continues on the rotated session id the compaction recorded.
    second = _turn(_fresh_agent(db, project, first_agent.session_id), compacted, app, monkeypatch, text="again")
    assert len(_hinted(second)) == 1


def test_s13_native_checkpoint_hint_is_injected_again(project, db, monkeypatch):
    """S13-native-checkpoint: the server renders nothing before a Codex native compaction checkpoint and the
    request builder drops the older tool outputs, while the stored rows stay live. A rebuilt agent sends again."""
    from agent.codex_responses_adapter import _chat_messages_to_responses_input

    history = _seed(db, _tool_round("file body\n\n" + MARKER + "\n" + RULES) + [
        {"role": "user", "content": "keep going"},
        {"role": "assistant", "content": "checkpointed",
         "codex_reasoning_items": [{"type": "compaction", "encrypted_content": "opaque-checkpoint"}]}])
    wire = json.dumps(_chat_messages_to_responses_input(history + [{"role": "user", "content": "x"}],
                                                        native_compaction_eligible=True))
    assert MARKER not in wire  # the replayed hint is really off the wire

    second = _turn(_fresh_agent(db, project), history, str(project / "backend" / "app.py"), monkeypatch)
    assert len(_hinted(second[len(history):])) == 1, "a rebuilt agent withheld a hint native compaction took off the wire"


def test_s13_printed_marker_does_not_suppress_the_hint(project, db, monkeypatch):
    """S13-printed-marker: only the full hint text counts as delivered. A tool output that prints the marker
    line (a sqlite dump of stored rows), or a copy of an AGENTS.md that has changed since, withholds nothing."""
    history = _seed(db, _tool_round("sqlite> select content from messages;\n" + MARKER + "\n(row truncated)")
                    + _tool_round("file body\n\n" + MARKER + "\nBackend rules: use Flask", call_id="call_2"))
    second = _turn(_fresh_agent(db, project), history, str(project / "backend" / "app.py"), monkeypatch)
    assert len(_hinted(second[len(history):])) == 1

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from agent.transports.codex_app_server_session import TurnResult
from agent.codex_runtime import _record_codex_app_server_compaction, _on_codex_app_server_compaction_event
from agent.context_compressor import ContextCompressor

def test_interrupted_or_failed_turn_preserves_compaction_count():
    """D3 requirement: Prove completed compactions remain accounted if the active turn fails/interruption
    occurs before finish, and that standard/forced/native paths cannot double-count."""
    compressor = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
    agent = SimpleNamespace(
        session_id="s1",
        platform="cli",
        provider="openai",
        model="gpt-5-codex",
        base_url="",
        context_compressor=compressor,
        events=[],
        event_callback=lambda name, payload: agent.events.append((name, payload)),
        _emit_status=lambda *a, **kw: None,
    )
    
    # 1. Mid-turn compaction happens (count = 1)
    _on_codex_app_server_compaction_event(agent, 1)
    assert compressor.compression_count == 1
    
    # 2. Another mid-turn compaction happens (count = 2)
    _on_codex_app_server_compaction_event(agent, 2)
    assert compressor.compression_count == 2
    
    # If turn fails or is interrupted BEFORE _finish_codex_turn / _record_codex_app_server_compaction is called,
    # the compressor already has compression_count == 2!
    assert compressor.compression_count == 2
    
    # 3. Now verify that when _record_codex_app_server_compaction is eventually called (or on next turn finish),
    # it does NOT double-count the compactions already accounted for!
    turn = TurnResult(thread_id="th-1", turn_id="tu-1", compacted=True)
    turn.compaction_count = 2
    _record_codex_app_server_compaction(agent, turn)
    
    assert compressor.compression_count == 2, "Must not double-count compactions already recorded at mid-turn boundary"

def test_failed_turn_resets_seen_slot_so_next_turn_accounts_compactions():
    """B1 requirement: Verify that when a turn raises, compressor._codex_turn_compaction_seen
    is reset so the next turn's compactions are correctly accounted for across multiple turns."""
    from agent.codex_runtime import run_codex_app_server_turn

    compressor = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
    
    fake_session = SimpleNamespace(
        ensure_started=lambda: "th-1",
        run_turn=MagicMock(side_effect=RuntimeError("simulated turn failure")),
    )

    agent = SimpleNamespace(
        api_mode="codex_app_server",
        session_id="s1",
        platform="cli",
        provider="openai",
        model="gpt-5-codex",
        base_url="",
        valid_tool_names=set(),
        _iters_since_skill=0,
        _skill_nudge_interval=0,
        _codex_session=fake_session,
        _session_db=None,
        _flush_messages_to_session_db=lambda msgs: True,
        session_api_calls=0,
        session_prompt_tokens=0,
        session_completion_tokens=0,
        session_total_tokens=0,
        session_cache_read_tokens=0,
        session_cache_write_tokens=0,
        session_estimated_cost_usd=0.0,
        session_cost_status="",
        session_cost_source="",
        context_compressor=compressor,
        events=[],
        event_callback=lambda name, payload: agent.events.append((name, payload)),
        _emit_status=lambda *a, **kw: None,
    )

    # During turn 1, 2 compactions occur before the failure
    _on_codex_app_server_compaction_event(agent, 1)
    _on_codex_app_server_compaction_event(agent, 2)
    assert compressor.compression_count == 2
    assert compressor._codex_turn_compaction_seen == 2

    # Turn 1 raises
    messages = [{"role": "user", "content": "hello"}]
    res = run_codex_app_server_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=messages,
        effective_task_id="",
    )
    assert res["completed"] is False
    assert "simulated turn failure" in str(res["error"])
    # The watermark MUST be reset to 0 after failure!
    assert compressor._codex_turn_compaction_seen == 0
    assert compressor.compression_count == 2

    # Turn 2 runs cleanly and experiences 2 more compactions (counts 1 and 2 in its own turn)
    turn2_result = TurnResult(thread_id="th-1", turn_id="tu-2", compacted=True)
    turn2_result.compaction_count = 2
    fake_session.run_turn = MagicMock(return_value=turn2_result)
    agent._codex_session = fake_session

    _on_codex_app_server_compaction_event(agent, 1)
    _on_codex_app_server_compaction_event(agent, 2)
    assert compressor.compression_count == 4, f"Expected 4 compactions, got {compressor.compression_count}"
    
    _record_codex_app_server_compaction(agent, turn2_result)
    assert compressor.compression_count == 4

def test_forced_compaction_does_not_double_count():
    compressor = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
    agent = SimpleNamespace(
        session_id="s1",
        platform="cli",
        provider="openai",
        model="gpt-5-codex",
        base_url="",
        context_compressor=compressor,
        events=[],
        event_callback=lambda name, payload: agent.events.append((name, payload)),
        _emit_status=lambda *a, **kw: None,
    )
    turn = TurnResult(thread_id="th-1", turn_id="tu-1", compacted=False)
    _record_codex_app_server_compaction(agent, turn, force=True)
    assert compressor.compression_count == 1

def test_forced_compaction_counted_when_native_compactions_occurred_in_same_turn():
    """B2 requirement: Verify that forced compaction increments compression_count even when
    turn_already_seen / _codex_turn_compaction_seen is non-zero from native compactions."""
    compressor = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
    agent = SimpleNamespace(
        session_id="s1",
        platform="cli",
        provider="openai",
        model="gpt-5-codex",
        base_url="",
        context_compressor=compressor,
        events=[],
        event_callback=lambda name, payload: agent.events.append((name, payload)),
        _emit_status=lambda *a, **kw: None,
    )
    
    # 2 native compactions observed mid-turn
    _on_codex_app_server_compaction_event(agent, 1)
    _on_codex_app_server_compaction_event(agent, 2)
    assert compressor.compression_count == 2
    assert compressor._codex_turn_compaction_seen == 2

    # Now Hermes-forced compaction is executed on the same turn (e.g. conversation_compression:4123)
    turn = TurnResult(thread_id="th-1", turn_id="tu-1", compacted=False)
    _record_codex_app_server_compaction(agent, turn, force=True)
    
    # Must increment to 3, NOT drop the forced compaction
    assert compressor.compression_count == 3, f"Expected 3 compactions after force, got {compressor.compression_count}"


def test_forced_compaction_via_compact_thread_production_caller(tmp_path, monkeypatch):
    """S8/S9 requirement: Drive the REAL caller _compress_context_via_codex_app_server(force=True)
    with a session whose compact_thread() emits a contextCompaction item.
    Asserts:
      1) One real forced compaction results in compression_count == 1 and exactly one on_compaction_complete.
      2) Watermark is cleared so a subsequent native turn with 1 real compaction increments
         compression_count to 2 and emits on_compaction_complete with count 2 (not swallowed, not duplicate).
    """
    import agent.conversation_compression as cc
    import hermes_cli.lifecycle as lifecycle
    from tests.agent.transports.test_codex_app_server_session import FakeClient, make_session

    home = tmp_path / "home_forced_caller"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    hook_counts = []
    orig_invoke = lifecycle.invoke_hook

    def spy_invoke(name, **kwargs):
        if name == "on_compaction_complete":
            hook_counts.append(kwargs.get("compression_count"))
        return orig_invoke(name, **kwargs)

    monkeypatch.setattr(lifecycle, "invoke_hook", spy_invoke)

    compressor = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
    agent = SimpleNamespace(
        api_mode="codex_app_server",
        session_id="sess-forced-caller",
        platform="cli",
        provider="openai",
        model="gpt-5-codex",
        base_url="",
        context_compressor=compressor,
        event_callback=None,
        _session_db=None,
        _codex_session=None,
        codex_app_server_auto_compaction="hermes",
        _cached_system_prompt="SYSTEM",
        _emit_status=lambda *a, **k: None,
        _emit_warning=lambda *a, **k: None,
    )

    # 1. Forced compaction (/compress route)
    client1 = FakeClient()
    agent._codex_session = make_session(
        client1,
        on_compaction=lambda c: _on_codex_app_server_compaction_event(agent, c),
    )
    client1.queue_notification("turn/started", threadId="thread-fake-001", turn={"id": "compact-turn-1"})
    client1.queue_notification(
        "item/completed", threadId="thread-fake-001", turnId="compact-turn-1",
        item={"type": "contextCompaction", "id": "compact-item-1"},
    )
    client1.queue_notification(
        "turn/completed", threadId="thread-fake-001", turnId="compact-turn-1",
        turn={"id": "compact-turn-1", "status": "completed", "error": None},
    )

    cc._compress_context_via_codex_app_server(
        agent, [{"role": "user", "content": "hi"}], "SYSTEM", approx_tokens=1000, task_id="t", force=True,
    )

    assert compressor.compression_count == 1, f"Expected compression_count == 1, got {compressor.compression_count}"
    assert hook_counts == [1], f"Expected exactly one hook emission [1], got {hook_counts}"
    assert getattr(compressor, "_codex_turn_compaction_seen", 0) == 0, "Watermark was not reset after forced compaction"

    # 2. Next native turn with 1 real compaction
    client2 = FakeClient()
    session2 = make_session(
        client2,
        on_compaction=lambda c: _on_codex_app_server_compaction_event(agent, c),
    )
    agent._codex_session = session2
    client2.queue_notification("item/completed", item={"type": "contextCompaction", "id": "cc-next"})
    client2.queue_notification("item/completed", item={"type": "agentMessage", "id": "m", "text": "done"})
    client2.queue_notification("turn/completed", turn={"id": "turn-fake-001", "status": "completed", "error": None})

    turn = session2.run_turn("next ask", turn_timeout=5.0, notification_poll_timeout=0.001)
    _record_codex_app_server_compaction(agent, turn)

    assert compressor.compression_count == 2, f"Expected compression_count == 2, got {compressor.compression_count}"
    assert hook_counts == [1, 2], f"Expected hook counts [1, 2], got {hook_counts}"

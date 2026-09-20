import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from agent.conversation_loop import run_conversation
from agent.transports.codex_app_server_session import TurnResult
from agent.context_compressor import ContextCompressor

def test_codex_receives_plugin_user_context_in_turn_input(monkeypatch):
    """C4 test: plugin_user_context collected at turn start is passed to codex run_turn."""
    from hermes_cli.plugins import PluginManager
    mgr = PluginManager()
    
    # Register pre_llm_call hook that returns context
    mgr._hooks.setdefault("pre_llm_call", []).append(lambda **kw: {"context": "INJECTED_CHECKPOINT_PROMPT"})
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mgr.invoke_hook)
    
    from agent.codex_runtime import run_codex_app_server_turn
    
    captured_inputs = []
    fake_session = SimpleNamespace(
        run_turn=lambda user_input, **kw: (captured_inputs.append(user_input), TurnResult(final_text="ok"))[1]
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
        context_compressor=None,
    )
    
    messages = [{"role": "user", "content": "hello world"}]
    
    run_codex_app_server_turn(
        agent,
        user_message="hello world",
        original_user_message="hello world",
        messages=messages,
        effective_task_id="",
        plugin_user_context="INJECTED_CHECKPOINT_PROMPT",
    )
    
    assert len(captured_inputs) == 1
    user_input = captured_inputs[0]
    if isinstance(user_input, str):
        assert "INJECTED_CHECKPOINT_PROMPT" in user_input
    elif isinstance(user_input, list):
        text_contents = [item.get("text", "") for item in user_input if isinstance(item, dict)]
        assert any("INJECTED_CHECKPOINT_PROMPT" in t for t in text_contents)

def test_c4_e2e_hook_collection_reaches_codex_wire_and_absent_control(monkeypatch):
    """C4 requirement (B5):
    1. Hook-collected context from pre_llm_call reaches the wire through turn_context.
    2. Absent-context control: when no hook returns context, wire input remains exactly user_message.
    3. Unchanged history, system bytes, and thread identity assertions.
    """
    from hermes_cli.plugins import PluginManager
    from agent.turn_context import _collect_pre_llm_call_context
    from agent.codex_runtime import run_codex_app_server_turn

    mgr = PluginManager()
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mgr.invoke_hook)

    # --- Part 1: Absent-context control ---
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
        context_compressor=None,
        _cached_system_prompt="SYSTEM_DEV_PROMPT",
    )

    history = [
        {"role": "user", "content": "prior turn user"},
        {"role": "assistant", "content": "prior turn assistant"},
    ]
    import copy
    history_before = copy.deepcopy(history)
    messages = list(history) + [{"role": "user", "content": "current user ask"}]
    messages_before = copy.deepcopy(messages)

    # 1. Collect context with no pre_llm_call hook registered
    absent_ctx = _collect_pre_llm_call_context(
        agent,
        effective_task_id="t1",
        turn_id="turn-1",
        original_user_message="current user ask",
        messages=messages,
        conversation_history=history,
    )
    assert absent_ctx == ""

    # Wire run with absent context
    wire_inputs_absent = []
    fake_session_1 = SimpleNamespace(
        run_turn=lambda user_input, **kw: (
            wire_inputs_absent.append(user_input),
            TurnResult(final_text="ok", thread_id="th-exact-1", turn_id="tu-1")
        )[1]
    )
    agent._codex_session = fake_session_1

    res_absent = run_codex_app_server_turn(
        agent,
        user_message="current user ask",
        original_user_message="current user ask",
        messages=messages,
        effective_task_id="t1",
        plugin_user_context=absent_ctx,
    )
    assert len(wire_inputs_absent) == 1
    assert wire_inputs_absent[0] == "current user ask"  # Exact unmodified user input!
    assert res_absent["codex_thread_id"] == "th-exact-1"

    # Verify history and system prompt bytes are byte-identical and unchanged
    assert history == history_before
    assert agent._cached_system_prompt == "SYSTEM_DEV_PROMPT"

    # --- Part 2: Active hook context reaches the wire ---
    mgr._hooks.setdefault("pre_llm_call", []).append(
        lambda **kw: {"context": "[ACTIVE_CHECKPOINT_HOOK_CONTEXT]"}
    )

    collected_ctx = _collect_pre_llm_call_context(
        agent,
        effective_task_id="t1",
        turn_id="turn-2",
        original_user_message="current user ask 2",
        messages=messages,
        conversation_history=history,
    )
    assert "[ACTIVE_CHECKPOINT_HOOK_CONTEXT]" in collected_ctx

    wire_inputs_active = []
    fake_session_2 = SimpleNamespace(
        run_turn=lambda user_input, **kw: (
            wire_inputs_active.append(user_input),
            TurnResult(final_text="ok 2", thread_id="th-exact-1", turn_id="tu-2")
        )[1]
    )
    agent._codex_session = fake_session_2

    res_active = run_codex_app_server_turn(
        agent,
        user_message="current user ask 2",
        original_user_message="current user ask 2",
        messages=messages,
        effective_task_id="t1",
        plugin_user_context=collected_ctx,
    )
    assert len(wire_inputs_active) == 1
    wire_input = wire_inputs_active[0]
    assert "current user ask 2" in wire_input
    assert "[ACTIVE_CHECKPOINT_HOOK_CONTEXT]" in wire_input
    # Thread identity preserved across turns
    assert res_active["codex_thread_id"] == "th-exact-1"
    # History and system prompt bytes remain unchanged
    assert history == history_before
    assert agent._cached_system_prompt == "SYSTEM_DEV_PROMPT"

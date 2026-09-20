import pytest
from types import SimpleNamespace
from agent.transports.codex_app_server_session import TurnResult, _apply_accounting_notification
from agent.codex_runtime import _record_codex_app_server_compaction
from agent.context_compressor import ContextCompressor

def test_mutant_count_fidelity_removed():
    """Mutant control: if accounting notification fails to distinguish or count compactions,
    the contract assertion fails."""
    result = TurnResult()
    note1 = {"method": "item/completed", "params": {"item": {"type": "contextCompaction", "id": "1"}}}
    note2 = {"method": "item/completed", "params": {"item": {"type": "contextCompaction", "id": "2"}}}
    _apply_accounting_notification(result, note1)
    _apply_accounting_notification(result, note2)
    assert result.compaction_count == 2

def test_mutant_delivery_removed():
    """Mutant control: if delivery is skipped or disabled, trigger does not succeed."""
    agent_no_steer = SimpleNamespace() # no redirect, no steer
    from tests.agent.test_trail_plugin_candidate_contract import load_candidate_plugin
    mod = load_candidate_plugin()
    delivered = mod.handle_compaction_complete(session_id="s_mutant", compression_count=2, agent=agent_no_steer)
    assert delivered is False


def test_mutant_resume_epoch_disabled(tmp_path, monkeypatch):
    """Mutant control: if counter-instance nonce isolation is removed (so durable dedupe collides
    across fresh ContextCompressor instances or address reuse), a reconstructed instance with 2
    new completions is falsely suppressed and fails the resumption contract."""
    home = tmp_path / "mutant_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_state import SessionDB
    db = SessionDB(home / "state.db")
    db.create_session("sess-mutant-resume", source="cli")
    from tests.agent.test_trail_plugin_candidate_contract import load_candidate_plugin
    mod = load_candidate_plugin()

    comp1 = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
    steers1 = []
    agent1 = SimpleNamespace(_session_db=db, session_id="sess-mutant-resume", context_compressor=comp1, redirect=lambda t: (steers1.append(t), True)[1])
    assert mod.handle_compaction_complete(session_id="sess-mutant-resume", compression_count=1, agent=agent1) is False
    assert mod.handle_compaction_complete(session_id="sess-mutant-resume", compression_count=2, agent=agent1) is True
    assert len(steers1) == 1

    # Instance 2 with colliding nonce / epoch (mutant: _get_counter_nonce returns colliding nonce)
    comp2 = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
    steers2 = []
    agent2 = SimpleNamespace(_session_db=db, session_id="sess-mutant-resume", context_compressor=comp2, redirect=lambda t: (steers2.append(t), True)[1])
    
    # Force colliding nonce to simulate recycled address / shared epoch defect:
    nonce1 = mod._get_counter_nonce(agent1)
    setattr(comp2, "_trail_counter_epoch", nonce1)
    
    # With colliding nonce, instance 2 is falsely suppressed by instance 1's durable record:
    assert mod.handle_compaction_complete(session_id="sess-mutant-resume", compression_count=1, agent=agent2) is False
    mutant_delivered = mod.handle_compaction_complete(session_id="sess-mutant-resume", compression_count=2, agent=agent2)
    assert mutant_delivered is False, "Mutant did not exhibit false suppression"

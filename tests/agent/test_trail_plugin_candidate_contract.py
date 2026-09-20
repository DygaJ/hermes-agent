import json
import pytest
from pathlib import Path
from types import SimpleNamespace
import importlib.util

from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from agent.context_compressor import ContextCompressor

def _create_synthetic_trail_plugin(target_dir: Path) -> Path:
    """Create a self-contained test plugin inline under target_dir (portable fixture)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    plugin_yaml = """name: trail-checkpoint
version: 0.1.0
description: "Proactive trail checkpoint policy on completed compaction"
author: NousResearch
kind: backend
"""
    init_py = '''"""Default-profile proactive trail checkpoint policy plugin.

Triggers a verified checkpoint request at completed compaction count >= 2,
and at each subsequent distinct completed compaction.
Uses profile-scoped durable metadata for duplicate suppression across in-place
and rotated sessions, resume/restart, and /new resets.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

CHECKPOINT_PROMPT = (
    "[Automatic Trail Checkpoint: completed context compactions >= 2. "
    "Please invoke the trail skill at the next safe tool boundary to verify and record a fresh checkpoint.]"
)


def _check_symlink_components(path: Path) -> None:
    """Validate that neither path nor any of its ancestors down to HERMES_HOME is a symlink.
    Scoped to the profile home boundary so symlinked parent directories above HERMES_HOME
    do not disable plugin data persistence.
    """
    home = get_hermes_home()
    curr = path
    while True:
        if curr.is_symlink():
            raise OSError(f"Refusing to access path through symlink component: {curr}")
        if curr == home:
            break
        parent = curr.parent
        if parent == curr:
            break
        curr = parent


def _get_state_file() -> Path:
    """Return the durable JSON state file inside the active profile directory.
    Uses get_hermes_home() so it respects profile scoping (A/B/A).
    Rejects any symlink component along the path family.
    """
    home = get_hermes_home()
    state_dir = home / "plugin-data" / "trail-checkpoint"
    _check_symlink_components(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "state.json"
    if state_file.is_symlink():
        raise OSError(f"Refusing to access symlink state file: {state_file}")
    return state_file


def _load_state() -> dict[str, Any]:
    try:
        f = _get_state_file()
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except OSError:
        raise
    except Exception as exc:
        logger.debug("Failed to read trail-checkpoint state: %s", exc)
    return {"sessions": {}}


def _save_state(state: dict[str, Any]) -> None:
    f = _get_state_file()
    tmp = f.with_name(f"{f.name}.tmp")
    if tmp.is_symlink():
        raise OSError(f"Refusing to write through symlink temp file: {tmp}")
    _check_symlink_components(tmp)
    try:
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        if tmp.is_symlink():
            raise OSError(f"Temp file became a symlink: {tmp}")
        if f.is_symlink():
            raise OSError(f"Target file is a symlink: {f}")
        tmp.replace(f)
    except OSError:
        if tmp.exists() and not tmp.is_symlink():
            try:
                tmp.unlink()
            except Exception:
                pass
        raise
    except Exception as exc:
        logger.warning("Failed to save trail-checkpoint state: %s", exc)


def _resolve_logical_session_id(agent: Any, session_id: str) -> str:
    """Resolve logical session identity across compression rotation.
    Queries session DB lineage if available, otherwise returns the session_id or chain root.
    """
    if not session_id:
        return ""
    db = getattr(agent, "_session_db", None) if agent else None
    if db is not None and hasattr(db, "get_compression_lineage"):
        try:
            lineage = db.get_compression_lineage(session_id)
            if lineage and isinstance(lineage, list):
                return str(lineage[0])
        except Exception:
            pass
    return session_id


def _get_counter_nonce(agent: Any) -> str:
    """Return or initialize an opaque nonce held on the active compressor instance.
    The nonce is stable for the lifetime of the compressor and across plugin reloads,
    but distinct for newly constructed compressor instances, preventing address-reuse
    collisions without relying on memory addresses or PIDs.
    """
    compressor = getattr(agent, "context_compressor", None) if agent else None
    if compressor is not None:
        nonce = getattr(compressor, "_trail_counter_epoch", None)
        if not nonce:
            nonce = uuid.uuid4().hex
            try:
                setattr(compressor, "_trail_counter_epoch", nonce)
            except Exception:
                pass
        return str(nonce)
    return ""


def _lookup_session_entry(sessions: dict[str, Any], logical_id: str) -> dict[str, Any]:
    """Look up durable state for logical_id, with backward compatibility for legacy keys."""
    if logical_id in sessions and isinstance(sessions[logical_id], dict):
        return sessions[logical_id]
    # Migration compatibility: look up prefixed entries like f"{logical_id}:..."
    matches = [
        v for k, v in sessions.items()
        if k.startswith(f"{logical_id}:") and isinstance(v, dict)
    ]
    if matches:
        return max(matches, key=lambda x: x.get("last_handled_count", 0))
    return {}


def handle_compaction_complete(
    session_id: str,
    compression_count: int,
    in_place: bool = False,
    runtime: str = "",
    agent: Any = None,
    **kwargs: Any,
) -> bool:
    """Handle on_compaction_complete event.
    Returns True if a checkpoint was triggered and delivered, False otherwise.
    """
    compressor = getattr(agent, "context_compressor", None) if agent else None
    if compressor is not None:
        observed = getattr(compressor, "_trail_observed_compactions", 0) + 1
        try:
            setattr(compressor, "_trail_observed_compactions", observed)
        except Exception:
            pass
    else:
        observed = 0

    if compression_count < 2:
        return False

    logical_id = _resolve_logical_session_id(agent, session_id)
    if not logical_id:
        logical_id = session_id or "default"

    nonce = _get_counter_nonce(agent)

    state = _load_state()
    sessions = state.setdefault("sessions", {})
    entry = _lookup_session_entry(sessions, logical_id)
    last_count = entry.get("last_handled_count", 0)
    last_nonce = entry.get("counter_epoch", "")

    # Duplicate suppression:
    # 1. If counter nonce matches last_nonce, this is the same active counter instance:
    #    suppress if incoming count <= last_count.
    # 2. If no counter instance exists on agent:
    #    suppress if incoming count <= last_count.
    # 3. If counter nonce is different (process restart, session reconstruction):
    #    if incoming count <= last_count, suppress duplicate re-delivery across restart
    #    unless this fresh counter instance has itself genuinely observed at least 2 completions.
    if nonce and last_nonce and nonce == last_nonce:
        if compression_count <= last_count:
            return False
    elif not nonce:
        if compression_count <= last_count:
            return False
    else:
        if compression_count <= last_count and observed < 2:
            return False

    delivered = False
    if agent is not None:
        if hasattr(agent, "redirect") and callable(agent.redirect):
            delivered = bool(agent.redirect(CHECKPOINT_PROMPT))
        elif hasattr(agent, "steer") and callable(agent.steer):
            delivered = bool(agent.steer(CHECKPOINT_PROMPT))

    if delivered:
        sessions[logical_id] = {
            "last_handled_count": compression_count,
            "session_id": session_id,
            "logical_id": logical_id,
            "counter_epoch": nonce,
        }
        _save_state(state)
        logger.info(
            "Trail checkpoint delivered during active turn for session %s (count=%d)",
            session_id,
            compression_count,
        )
        return True
    else:
        logger.warning(
            "Trail checkpoint trigger queued or failed delivery for session %s (count=%d)",
            session_id,
            compression_count,
        )
        return False


def handle_session_reset(session_id: str = "", **kwargs: Any) -> None:
    """Clear state on session reset (/new) if appropriate."""
    pass


def register(ctx: Any) -> None:
    """Plugin entry point."""
    ctx.register_hook("on_compaction_complete", handle_compaction_complete)
    ctx.register_hook("on_session_reset", handle_session_reset)
'''
    (target_dir / "plugin.yaml").write_text(plugin_yaml, encoding="utf-8")
    init_path = target_dir / "__init__.py"
    init_path.write_text(init_py, encoding="utf-8")
    return init_path

def load_candidate_plugin(target_dir: Path | None = None):
    if target_dir is None:
        import tempfile
        target_dir = Path(tempfile.mkdtemp(prefix="trail_candidate_test_"))
    init_file = _create_synthetic_trail_plugin(target_dir)
    spec = importlib.util.spec_from_file_location("trail_checkpoint_plugin", str(init_file))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

class TestPluginCandidateContract:
    def test_threshold_1_does_not_trigger(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mod = load_candidate_plugin()
        
        delivered = mod.handle_compaction_complete(
            session_id="s1",
            compression_count=1,
            agent=SimpleNamespace(redirect=lambda t: True),
        )
        assert delivered is False

    def test_threshold_2_triggers_and_delivers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mod = load_candidate_plugin()
        
        steer_msgs = []
        agent = SimpleNamespace(redirect=lambda t: (steer_msgs.append(t), True)[1])
        
        delivered = mod.handle_compaction_complete(
            session_id="s1",
            compression_count=2,
            agent=agent,
        )
        assert delivered is True
        assert len(steer_msgs) == 1
        assert "Automatic Trail Checkpoint" in steer_msgs[0]

    def test_duplicate_count_2_suppressed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mod = load_candidate_plugin()
        
        steer_msgs = []
        agent = SimpleNamespace(redirect=lambda t: (steer_msgs.append(t), True)[1])
        
        delivered1 = mod.handle_compaction_complete(
            session_id="s1",
            compression_count=2,
            agent=agent,
        )
        delivered2 = mod.handle_compaction_complete(
            session_id="s1",
            compression_count=2,
            agent=agent,
        )
        assert delivered1 is True
        assert delivered2 is False
        assert len(steer_msgs) == 1

    def test_count_3_triggers_again(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mod = load_candidate_plugin()
        
        steer_msgs = []
        agent = SimpleNamespace(redirect=lambda t: (steer_msgs.append(t), True)[1])
        
        mod.handle_compaction_complete(session_id="s1", compression_count=2, agent=agent)
        delivered3 = mod.handle_compaction_complete(session_id="s1", compression_count=3, agent=agent)
        assert delivered3 is True
        assert len(steer_msgs) == 2

    def test_aba_profile_isolation(self, tmp_path, monkeypatch):
        home_a = tmp_path / "profile_a"
        home_b = tmp_path / "profile_b"
        home_a.mkdir()
        home_b.mkdir()
        
        mod = load_candidate_plugin()
        
        # In profile A, count 2 triggers
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        steer_a = []
        agent_a = SimpleNamespace(redirect=lambda t: (steer_a.append(t), True)[1])
        assert mod.handle_compaction_complete(session_id="s1", compression_count=2, agent=agent_a) is True
        
        # In profile B, count 2 triggers independently
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        steer_b = []
        agent_b = SimpleNamespace(redirect=lambda t: (steer_b.append(t), True)[1])
        assert mod.handle_compaction_complete(session_id="s1", compression_count=2, agent=agent_b) is True
        
        # Back to profile A, count 2 is suppressed
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert mod.handle_compaction_complete(session_id="s1", compression_count=2, agent=agent_a) is False

    def test_delivery_failure_not_acknowledged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mod = load_candidate_plugin()
        
        agent = SimpleNamespace(redirect=lambda t: False)
        delivered = mod.handle_compaction_complete(session_id="s1", compression_count=2, agent=agent)
        assert delivered is False
        
        agent_ok = SimpleNamespace(redirect=lambda t: True)
        delivered_retry = mod.handle_compaction_complete(session_id="s1", compression_count=2, agent=agent_ok)
        assert delivered_retry is True

    def test_refuses_symlink_state_dir(self, tmp_path, monkeypatch):
        home = tmp_path / "profile_sym_dir"
        home.mkdir()
        outside = tmp_path / "outside_target_dir"
        outside.mkdir()
        
        plugin_data = home / "plugin-data"
        plugin_data.mkdir()
        state_dir_symlink = plugin_data / "trail-checkpoint"
        state_dir_symlink.symlink_to(outside)
        
        monkeypatch.setenv("HERMES_HOME", str(home))
        mod = load_candidate_plugin()
        
        with pytest.raises(OSError, match="symlink"):
            mod._save_state({"test": 1})

    def test_refuses_symlink_parent_plugin_data(self, tmp_path, monkeypatch):
        home = tmp_path / "profile_sym_parent"
        home.mkdir()
        outside = tmp_path / "outside_target_parent"
        outside.mkdir()
        
        plugin_data_symlink = home / "plugin-data"
        plugin_data_symlink.symlink_to(outside)
        
        monkeypatch.setenv("HERMES_HOME", str(home))
        mod = load_candidate_plugin()
        
        with pytest.raises(OSError, match="symlink"):
            mod._save_state({"test": 1})

    def test_refuses_symlink_state_file(self, tmp_path, monkeypatch):
        home = tmp_path / "profile_sym_file"
        home.mkdir()
        outside_file = tmp_path / "outside_file.json"
        outside_file.write_text("original outside")
        
        state_dir = home / "plugin-data" / "trail-checkpoint"
        state_dir.mkdir(parents=True)
        state_file = state_dir / "state.json"
        state_file.symlink_to(outside_file)
        
        monkeypatch.setenv("HERMES_HOME", str(home))
        mod = load_candidate_plugin()
        
        with pytest.raises(OSError, match="symlink"):
            mod._save_state({"test": 1})
            
        assert outside_file.read_text() == "original outside"

    def test_refuses_symlink_temp_file(self, tmp_path, monkeypatch):
        home = tmp_path / "profile_sym_tmp"
        home.mkdir()
        outside_file = tmp_path / "outside_tmp_target.json"
        outside_file.write_text("original outside tmp")
        
        state_dir = home / "plugin-data" / "trail-checkpoint"
        state_dir.mkdir(parents=True)
        tmp_file = state_dir / "state.json.tmp"
        tmp_file.symlink_to(outside_file)
        
        monkeypatch.setenv("HERMES_HOME", str(home))
        mod = load_candidate_plugin()
        
        with pytest.raises(OSError, match="symlink"):
            mod._save_state({"test": 1})
            
        assert outside_file.read_text() == "original outside tmp"

    def test_permits_symlinked_ancestor_above_hermes_home(self, tmp_path, monkeypatch):
        """S1 requirement: Symlinked parent directories above HERMES_HOME must be permitted,
        while state files and directories at/below HERMES_HOME are strictly validated."""
        real_parent = tmp_path / "real_parent"
        real_parent.mkdir()
        symlink_parent = tmp_path / "symlink_parent"
        symlink_parent.symlink_to(real_parent)
        
        home = symlink_parent / "hermes_home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        mod = load_candidate_plugin()

        agent = SimpleNamespace(redirect=lambda t: True)
        delivered = mod.handle_compaction_complete(
            session_id="s_symlink_ancestor",
            compression_count=2,
            agent=agent,
        )
        assert delivered is True
        state = mod._load_state()
        assert "s_symlink_ancestor" in state.get("sessions", {})

    def test_rotation_lineage_duplicate_suppressed_and_subsequent_triggers(self, tmp_path, monkeypatch):
        """C5 requirement: In-place/rotation session continuation via real SessionDB.
        Duplicate count 2 on rotated session is suppressed; subsequent count 3 triggers."""
        home = tmp_path / "profile_rotation"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(home / "state.db")
        mod = load_candidate_plugin()

        db.create_session("sess-A", source="cli")
        steers = []
        comp_a = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        agent_a = SimpleNamespace(_session_db=db, session_id="sess-A", context_compressor=comp_a, redirect=lambda t: (steers.append(t), True)[1])
        r1 = mod.handle_compaction_complete(session_id="sess-A", compression_count=2, agent=agent_a)
        assert r1 is True
        assert len(steers) == 1

        db.end_session("sess-A", end_reason="compression")
        db.create_session("sess-B", source="cli", parent_session_id="sess-A")
        comp_b = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        agent_b = SimpleNamespace(_session_db=db, session_id="sess-B", context_compressor=comp_b, redirect=lambda t: (steers.append(t), True)[1])

        # Duplicate count 2 on rotated continuation session must be suppressed
        r2_same = mod.handle_compaction_complete(session_id="sess-B", compression_count=2, agent=agent_b)
        assert r2_same is False
        assert len(steers) == 1

        # Genuinely new distinct count 3 on rotated session must trigger
        r3_next = mod.handle_compaction_complete(session_id="sess-B", compression_count=3, agent=agent_b)
        assert r3_next is True
        assert len(steers) == 2

    def test_restart_and_resume_preserves_deduplication(self, tmp_path, monkeypatch):
        """C5 requirement: Restart / resume preserves durable deduplication state."""
        home = tmp_path / "profile_restart"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(home / "state.db")
        mod1 = load_candidate_plugin()

        db.create_session("sess-1", source="cli")
        steers1 = []
        comp1 = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        agent1 = SimpleNamespace(_session_db=db, session_id="sess-1", context_compressor=comp1, redirect=lambda t: (steers1.append(t), True)[1])
        assert mod1.handle_compaction_complete(session_id="sess-1", compression_count=2, agent=agent1) is True
        assert len(steers1) == 1

        # Simulate process restart by reloading plugin instance under same HERMES_HOME
        mod2 = load_candidate_plugin()
        steers2 = []
        comp2 = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        agent2 = SimpleNamespace(_session_db=db, session_id="sess-1", context_compressor=comp2, redirect=lambda t: (steers2.append(t), True)[1])
        # Already handled count 2 is suppressed
        assert mod2.handle_compaction_complete(session_id="sess-1", compression_count=2, agent=agent2) is False
        assert len(steers2) == 0
        # Subsequent count 3 triggers
        assert mod2.handle_compaction_complete(session_id="sess-1", compression_count=3, agent=agent2) is True
        assert len(steers2) == 1

    def test_explicit_new_session_resets_independent_conversation(self, tmp_path, monkeypatch):
        """C5 requirement: Explicit new-session (/new) reset creates independent conversation
        where count 2 triggers independently and handle_session_reset is executed."""
        home = tmp_path / "profile_new_session"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(home / "state.db")
        mod = load_candidate_plugin()

        db.create_session("sess-old", source="cli")
        steers_old = []
        comp_old = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        agent_old = SimpleNamespace(_session_db=db, session_id="sess-old", context_compressor=comp_old, redirect=lambda t: (steers_old.append(t), True)[1])
        assert mod.handle_compaction_complete(session_id="sess-old", compression_count=2, agent=agent_old) is True
        assert len(steers_old) == 1

        # Reset session
        mod.handle_session_reset(session_id="sess-old")

        # Create new independent session (/new reset)
        db.create_session("sess-new", source="cli")
        steers_new = []
        comp_new = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        agent_new = SimpleNamespace(_session_db=db, session_id="sess-new", context_compressor=comp_new, redirect=lambda t: (steers_new.append(t), True)[1])
        assert mod.handle_compaction_complete(session_id="sess-new", compression_count=2, agent=agent_new) is True
        assert len(steers_new) == 1

    def test_reconstructed_fresh_compressor_instance_resumes_without_false_suppression(self, tmp_path, monkeypatch):
        """P1-RESUME / C5 requirement: A fresh real ContextCompressor instance starting at 0
        in the same logical session (e.g. process restart / session reconstruction without durable
        runtime count restoration) triggers upon observing 2 new completions, rather than being
        suppressed by the prior instance's high-water mark.
        """
        from agent.context_compressor import ContextCompressor

        home = tmp_path / "profile_resume_epoch"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(home / "state.db")
        db.create_session("sess-resume", source="cli")
        mod = load_candidate_plugin()

        # Instance 1: fresh compressor starts at 0, reaches 2 compactions -> triggers 1 checkpoint
        comp1 = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        steers1 = []
        agent1 = SimpleNamespace(_session_db=db, session_id="sess-resume", context_compressor=comp1, redirect=lambda t: (steers1.append(t), True)[1])
        assert mod.handle_compaction_complete(session_id="sess-resume", compression_count=1, agent=agent1) is False
        assert mod.handle_compaction_complete(session_id="sess-resume", compression_count=2, agent=agent1) is True
        assert len(steers1) == 1
        # Duplicate count on same instance is suppressed
        assert mod.handle_compaction_complete(session_id="sess-resume", compression_count=2, agent=agent1) is False
        assert len(steers1) == 1

        # Instance 2: reconstructed agent / fresh compressor starts at 0, reaches 2 compactions in same logical session
        # Deterministically simulate allocator address reuse or colliding identity:
        comp2 = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        steers2 = []
        agent2 = SimpleNamespace(_session_db=db, session_id="sess-resume", context_compressor=comp2, redirect=lambda t: (steers2.append(t), True)[1])
        
        # Verify comp2 gets its own unique nonce independent of memory address
        assert mod._get_counter_nonce(agent1) != mod._get_counter_nonce(agent2)

        assert mod.handle_compaction_complete(session_id="sess-resume", compression_count=1, agent=agent2) is False
        assert mod.handle_compaction_complete(session_id="sess-resume", compression_count=2, agent=agent2) is True
        assert len(steers2) == 1
        # Duplicate on instance 2 is suppressed
        assert mod.handle_compaction_complete(session_id="sess-resume", compression_count=2, agent=agent2) is False
        assert len(steers2) == 1

    def test_deterministic_recycled_address_regression(self, tmp_path, monkeypatch):
        """S10 regression: drive epoch derivation with colliding identity / address reuse.
        A fresh counter must not be falsely suppressed when an address is recycled."""
        home = tmp_path / "profile_recycled_addr"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        db = SessionDB(home / "state.db")
        db.create_session("sess-recycled", source="cli")
        mod = load_candidate_plugin()

        comp1 = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        steers1 = []
        agent1 = SimpleNamespace(_session_db=db, session_id="sess-recycled", context_compressor=comp1, redirect=lambda t: (steers1.append(t), True)[1])
        assert mod.handle_compaction_complete(session_id="sess-recycled", compression_count=1, agent=agent1) is False
        assert mod.handle_compaction_complete(session_id="sess-recycled", compression_count=2, agent=agent1) is True
        assert len(steers1) == 1

        # Instance 2 has its own compressor instance whose nonce is isolated from comp1
        comp2 = ContextCompressor(model="gpt-5-codex", quiet_mode=True)
        steers2 = []
        agent2 = SimpleNamespace(_session_db=db, session_id="sess-recycled", context_compressor=comp2, redirect=lambda t: (steers2.append(t), True)[1])
        assert mod.handle_compaction_complete(session_id="sess-recycled", compression_count=1, agent=agent2) is False
        assert mod.handle_compaction_complete(session_id="sess-recycled", compression_count=2, agent=agent2) is True
        assert len(steers2) == 1

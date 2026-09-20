import pytest
from pathlib import Path
from hermes_cli.plugins import PluginManager

def _create_synthetic_trail_plugin(target_dir: Path) -> None:
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
    (target_dir / "__init__.py").write_text(init_py, encoding="utf-8")


def test_staged_plugin_loads_and_registers_via_plugin_manager(tmp_path, monkeypatch):
    """C8 requirement: Staged plugin loads and registers in PluginManager from a temp home/plugins dir."""
    home = tmp_path / "profile_plugin"
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True)
    
    target_plugin_dir = plugins_dir / "trail-checkpoint"
    _create_synthetic_trail_plugin(target_plugin_dir)
        
    # Write config.yaml enabling the plugin
    config_file = home / "config.yaml"
    config_file.write_text("plugins:\n  enabled:\n    - trail-checkpoint\n")
        
    monkeypatch.setenv("HERMES_HOME", str(home))
    
    from hermes_cli.plugins import discover_plugins
    discover_plugins(force=True)
    
    from hermes_cli.lifecycle import has_hook
    assert has_hook("on_compaction_complete")


def test_staged_plugin_not_autoloaded_without_consent(tmp_path, monkeypatch):
    """S7 / C8 consent control: A user-installed plugin declaring kind: backend in ~/.hermes/plugins
    must NOT autoload without an explicit plugins.enabled entry (only bundled plugins autoload)."""
    home = tmp_path / "profile_noconsent"
    plugins_dir = home / "plugins"
    plugins_dir.mkdir(parents=True)

    target_plugin_dir = plugins_dir / "trail-checkpoint"
    _create_synthetic_trail_plugin(target_plugin_dir)

    # Config with no plugins.enabled entry
    config_file = home / "config.yaml"
    config_file.write_text("model:\n  default: gpt-5\n")

    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.plugins import discover_plugins
    discover_plugins(force=True)

    from hermes_cli.lifecycle import has_hook
    assert not has_hook("on_compaction_complete"), "User plugin must not autoload without consent"

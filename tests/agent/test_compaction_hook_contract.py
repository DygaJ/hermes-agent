import pytest
from hermes_cli.plugins import VALID_HOOKS, PluginManager
import hermes_cli.plugins as plugins_mod

def test_on_compaction_complete_in_valid_hooks():
    assert "on_compaction_complete" in VALID_HOOKS

def test_on_compaction_complete_fires_via_invoke_hook():
    mgr = PluginManager()
    events = []
    def hook_cb(**kwargs):
        events.append(kwargs)

    mgr._hooks.setdefault("on_compaction_complete", []).append(hook_cb)
    results = mgr.invoke_hook("on_compaction_complete", session_id="s1", compression_count=2, in_place=True, runtime="chat_completions")
    assert len(events) == 1
    assert events[0]["session_id"] == "s1"
    assert events[0]["compression_count"] == 2
    assert events[0]["runtime"] == "chat_completions"

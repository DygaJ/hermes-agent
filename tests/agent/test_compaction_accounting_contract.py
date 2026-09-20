import logging
import pytest
from types import SimpleNamespace
from agent.transports.codex_app_server_session import TurnResult, _apply_accounting_notification
from agent.codex_runtime import _record_codex_app_server_compaction
from agent.context_compressor import ContextCompressor

class TestCodexCompactionAccountingContract:
    def test_two_completed_compactions_in_one_turn_increments_count_by_two(self):
        """C1 characterization test: two distinct contextCompaction completions in one turn
        must result in compaction_count == 2 and compression_count incrementing by 2."""
        result = TurnResult(thread_id="thread-1", turn_id="turn-1")
        
        note1 = {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {"type": "contextCompaction", "id": "compact-item-1"},
            },
        }
        note2 = {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {"type": "contextCompaction", "id": "compact-item-2"},
            },
        }
        
        _apply_accounting_notification(result, note1)
        _apply_accounting_notification(result, note2)
        
        assert getattr(result, "compaction_count", 0) == 2
        assert result.compacted is True
        
        agent = SimpleNamespace(
            session_id="session-1",
            platform="cli",
            provider="openai",
            model="gpt-5-codex",
            base_url="",
            context_compressor=ContextCompressor(model="gpt-5-codex", quiet_mode=True),
            events=[],
            event_callback=lambda name, payload: agent.events.append((name, payload)),
            _emit_status=lambda *a, **kw: None,
        )
        
        assert _record_codex_app_server_compaction(agent, result) is True
        assert agent.context_compressor.compression_count == 2

    def test_started_only_does_not_increment_count(self):
        result = TurnResult(thread_id="thread-1", turn_id="turn-1")
        note_started = {
            "method": "item/started",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {"type": "contextCompaction", "id": "compact-item-1"},
            },
        }
        _apply_accounting_notification(result, note_started)
        assert result.compaction_count == 0
        assert result.compacted is False

    def test_duplicate_item_started_and_completed_does_not_double_count(self):
        result = TurnResult(thread_id="thread-1", turn_id="turn-1")
        note_started = {
            "method": "item/started",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {"type": "contextCompaction", "id": "compact-item-1"},
            },
        }
        note_completed1 = {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {"type": "contextCompaction", "id": "compact-item-1"},
            },
        }
        note_completed2 = {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {"type": "contextCompaction", "id": "compact-item-1"},
            },
        }
        _apply_accounting_notification(result, note_started)
        _apply_accounting_notification(result, note_completed1)
        _apply_accounting_notification(result, note_completed2)
        assert result.compaction_count == 1
        assert result.compacted is True

    def test_legacy_thread_compacted_supported(self):
        result = TurnResult(thread_id="thread-1", turn_id="turn-1")
        note = {
            "method": "thread/compacted",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
            },
        }
        _apply_accounting_notification(result, note)
        assert result.compaction_count == 1
        assert result.compacted is True

    def test_two_legacy_thread_compacted_in_one_turn_count_as_two(self):
        """R1 requirement: Two distinct completed compactions within ONE turn must count as two,
        including when emitted via legacy thread/compacted notifications."""
        result = TurnResult(thread_id="thread-1", turn_id="turn-1")
        note1 = {
            "method": "thread/compacted",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
            },
        }
        note2 = {
            "method": "thread/compacted",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
            },
        }
        _apply_accounting_notification(result, note1)
        _apply_accounting_notification(result, note2)
        assert result.compaction_count == 2
        assert result.compacted is True

    def test_malformed_idless_item_does_not_invent_correlation_id_or_count(self, caplog):
        """S4/S6: An idless or malformed item must not be promoted to a distinct completion,
        invent a synthetic unique correlation ID, or echo untrusted payload into logs."""
        result = TurnResult(thread_id="thread-1", turn_id="turn-1")
        secret = "sk-liv-canary-secret-token-12345"
        user_transcript = "private user medical transcription text canary"
        note_idless = {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {
                    "type": "contextCompaction",
                    "auth": f"Bearer {secret}",
                    "transcript": user_transcript,
                },
            },
        }
        note_empty = {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {
                    "type": "contextCompaction",
                    "id": "   ",
                    "token": secret,
                },
            },
        }
        note_non_str = {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {
                    "type": "contextCompaction",
                    "id": 12345,
                    "token": secret,
                },
            },
        }
        with caplog.at_level(logging.WARNING, logger="agent.transports.codex_app_server_session"):
            _apply_accounting_notification(result, note_idless)
            _apply_accounting_notification(result, note_idless)  # Retransmit check
            _apply_accounting_notification(result, note_empty)
            _apply_accounting_notification(result, note_non_str)
        assert result.compaction_count == 0
        assert result.compacted is False

        # S6 canary non-echo assertion
        logged_text = "\n".join(r.getMessage() for r in caplog.records)
        assert secret not in logged_text, "Credentials must not be echoed in refusal logs"
        assert user_transcript not in logged_text, "Transcripts must not be echoed in refusal logs"
        for r in caplog.records:
            assert len(r.getMessage()) <= 120, "Diagnostic log record length must be bounded"

    def test_mirrored_producer_pairs_in_one_turn_count_accurately(self):
        """R1 / S5: Mirrored producer streams (e.g. 0.100.0 dual emission where each completion emits
        both contextCompaction and legacy thread/compacted) do not double-count."""
        # 1 compaction as both representations in producer order (item/completed then legacy)
        result1 = TurnResult(thread_id="thread-1", turn_id="turn-1")
        note_item1 = {
            "method": "item/completed",
            "params": {"threadId": "thread-1", "turnId": "turn-1", "item": {"type": "contextCompaction", "id": "c1"}},
        }
        note_legacy = {"method": "thread/compacted", "params": {"threadId": "thread-1", "turnId": "turn-1"}}
        _apply_accounting_notification(result1, note_item1)
        _apply_accounting_notification(result1, note_legacy)
        assert result1.compaction_count == 1

        # 2 compactions, both mirrored in canonical order (item1, legacy1, item2, legacy2)
        result2 = TurnResult(thread_id="thread-1", turn_id="turn-1")
        note_item2 = {
            "method": "item/completed",
            "params": {"threadId": "thread-1", "turnId": "turn-1", "item": {"type": "contextCompaction", "id": "c2"}},
        }
        _apply_accounting_notification(result2, note_item1)
        _apply_accounting_notification(result2, note_legacy)
        _apply_accounting_notification(result2, note_item2)
        _apply_accounting_notification(result2, note_legacy)
        assert result2.compaction_count == 2

        # 2 compactions, grouped order (item1, item2, legacy1, legacy2)
        result3 = TurnResult(thread_id="thread-1", turn_id="turn-1")
        _apply_accounting_notification(result3, note_item1)
        _apply_accounting_notification(result3, note_item2)
        _apply_accounting_notification(result3, note_legacy)
        _apply_accounting_notification(result3, note_legacy)
        assert result3.compaction_count == 2

        # Legacy-first pair order (legacy1, item1, legacy2, item2)
        result4 = TurnResult(thread_id="thread-1", turn_id="turn-1")
        _apply_accounting_notification(result4, note_legacy)
        _apply_accounting_notification(result4, note_item1)
        _apply_accounting_notification(result4, note_legacy)
        _apply_accounting_notification(result4, note_item2)
        assert result4.compaction_count == 2

    def test_failed_or_other_items_do_not_count(self):
        result = TurnResult(thread_id="thread-1", turn_id="turn-1")
        note_msg = {
            "method": "item/completed",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "item": {"type": "agentMessage", "id": "msg-1"},
            },
        }
        _apply_accounting_notification(result, note_msg)
        assert result.compaction_count == 0
        assert result.compacted is False

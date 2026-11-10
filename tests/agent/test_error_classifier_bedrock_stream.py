"""Bedrock Converse reports an outage that starts mid-stream as a botocore ``EventStreamError``:
no HTTP status, the error code only in ``exc.response["Error"]["Code"]``. That code has to reach
the classifier, or the turn ends after its retries instead of waiting on the outage ladder."""
import pytest

from agent.error_classifier import TRANSIENT_REASONS, FailoverReason, classify_api_error
from agent.turn_retry_state import TurnRetryState
from tests.agent.test_turn_recovery_autorecover import _Agent, _settle

botocore_exceptions = pytest.importorskip("botocore.exceptions")

_UNAVAILABLE_TEXT = "Bedrock is unable to process your request."


def _stream_error(code, message=_UNAVAILABLE_TEXT):
    return botocore_exceptions.EventStreamError({"Error": {"Code": code, "Message": message}}, "ConverseStream")


def _bedrock_agent():
    agent = _Agent()
    agent.provider = "bedrock"
    return agent


def _classify(error):
    return classify_api_error(error, provider="bedrock", model="global.anthropic.claude-fable-5-2")


@pytest.mark.parametrize("code", [
    "serviceUnavailableException", "ServiceUnavailableException", "SERVICEUNAVAILABLEEXCEPTION",
])
def test_mid_stream_service_unavailable_is_a_transient_overload(code):
    result = _classify(_stream_error(code))
    assert (result.reason, result.retryable) == (FailoverReason.overloaded, True)
    assert result.reason in TRANSIENT_REASONS
    assert result.should_rotate_credential is False


def test_verdict_comes_from_the_structured_code_not_the_message():
    # Same code, a message that names nothing: still an overload.
    assert _classify(_stream_error("serviceUnavailableException", "x")).reason == FailoverReason.overloaded
    # The full botocore sentence without the structured code decides nothing.
    text_only = Exception(str(_stream_error("serviceUnavailableException")))
    assert "serviceUnavailableException" in str(text_only)
    assert _classify(text_only).reason == FailoverReason.unknown
    # Another code under the outage wording is not an overload either.
    assert _classify(_stream_error("validationException")).reason == FailoverReason.unknown


def test_code_carried_only_by_the_cause_is_classified_the_same():
    try:
        try:
            raise _stream_error("serviceUnavailableException")
        except botocore_exceptions.EventStreamError as cause:
            raise RuntimeError("stream worker failed") from cause
    except RuntimeError as wrapped:
        result = _classify(wrapped)
    assert (result.reason, result.retryable) == (FailoverReason.overloaded, True)


def test_the_code_means_overload_for_bedrock_only():
    error = _stream_error("serviceUnavailableException")
    assert classify_api_error(error, provider="openai").reason == FailoverReason.unknown


@pytest.mark.parametrize("code, message, reason", [
    ("throttlingException", "Too many requests, please wait before trying again.", FailoverReason.rate_limit),
    ("validationException", "The input is malformed.", FailoverReason.unknown),
    ("someFutureException", "Something new happened.", FailoverReason.unknown),
])
def test_other_mid_stream_codes_keep_their_verdict(code, message, reason):
    assert _classify(_stream_error(code, message)).reason == reason


def test_pre_stream_503_stays_overloaded():
    error = botocore_exceptions.ClientError(
        {"Error": {"Code": "ServiceUnavailableException", "Message": _UNAVAILABLE_TEXT},
         "ResponseMetadata": {"HTTPStatusCode": 503}}, "ConverseStream")
    result = _classify(error)
    assert (result.reason, result.status_code) == (FailoverReason.overloaded, 503)


def test_exhausted_retries_on_mid_stream_unavailable_enter_the_outage_ladder():
    """Retries spent, no fallback left, nothing delivered: the turn waits one visible cycle and
    re-enters the retry loop from zero instead of ending."""
    agent = _bedrock_agent()
    verdict, retry = _settle(agent, _stream_error("serviceUnavailableException"))
    assert (verdict.action, verdict.retry_count, retry.auto_recovery_cycles_used) == ("continue", 0, 1)
    assert agent.statuses == agent.waits and len(agent.statuses) == 1
    assert "(cycle 1/5)" in agent.statuses[0]

    # The ladder stays bounded for this error as for any other.
    spent = TurnRetryState(primary_recovery_attempted=True, auto_recovery_cycles_used=5)
    verdict, _ = _settle(_bedrock_agent(), _stream_error("serviceUnavailableException"), retry=spent)
    assert verdict.action == "return" and verdict.result == {"failed": True}

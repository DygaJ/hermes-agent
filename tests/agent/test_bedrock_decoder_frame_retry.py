"""AnthropicBedrock reports an outage that starts mid-stream as a plain ``ValueError`` from the
SDK's event-stream decoder, with the code only in the frame's ``:exception-type`` header. That code
has to reach the classifier, or the turn treats the outage as a local bug and leaves the provider
after one attempt. The Converse shape of the same fault is in test_error_classifier_bedrock_stream.

The agent fixture and the frame builder follow NousResearch/hermes-agent PR #138609."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import retry_utils
from agent.api_error_summary import provider_stream_frame_code
from agent.error_classifier import TRANSIENT_REASONS, FailoverReason, classify_api_error
from run_agent import AIAgent

_UNAVAILABLE_TEXT = "Bedrock is unable to process your request."


def _frame(exception_type, message=_UNAVAILABLE_TEXT):
    # The text the SDK's decoder raises: the repr of botocore's ``to_response_dict()``.
    response_dict = {
        "status_code": 400,
        "headers": {":exception-type": exception_type, ":content-type": "application/json",
                    ":message-type": "exception"},
        "body": f'{{"message":"{message}"}}'.encode(),
    }
    return ValueError(f"Bad response code, expected 200: {response_dict}")


def _classify(error, provider="bedrock"):
    return classify_api_error(error, provider=provider, model="global.anthropic.claude-fable-5-2")


@pytest.mark.parametrize("code", ["serviceUnavailableException", "ServiceUnavailableException"])
def test_decoder_frame_service_unavailable_is_a_transient_overload(code):
    result = _classify(_frame(code))
    assert (result.reason, result.retryable) == (FailoverReason.overloaded, True)
    assert result.reason in TRANSIENT_REASONS


@pytest.mark.parametrize("alias", ["aws", "aws-bedrock", "amazon-bedrock", "amazon"])
def test_a_bedrock_alias_classifies_like_bedrock(alias):
    assert _classify(_frame("serviceUnavailableException"), provider=alias).reason == FailoverReason.overloaded


def test_the_code_means_overload_for_bedrock_only():
    assert _classify(_frame("serviceUnavailableException"), provider="anthropic").reason == FailoverReason.unknown


@pytest.mark.parametrize("code, reason", [
    ("throttlingException", FailoverReason.rate_limit),
    ("validationException", FailoverReason.unknown),
    ("internalServerException", FailoverReason.unknown),
    ("modelStreamErrorException", FailoverReason.unknown),
    ("modelTimeoutException", FailoverReason.unknown),
])
def test_other_decoder_frame_codes_keep_their_verdict(code, reason):
    # Only the code seen live is mapped. internalServerException also answers requests that fail
    # the same way on every retry, so it stays out until a live case shows otherwise.
    assert _classify(_frame(code, "x")).reason == reason


def test_only_the_exception_type_header_is_read():
    named_in_body = _frame("validationException", "Retry only on serviceUnavailableException.")
    assert provider_stream_frame_code(named_in_body) == "validationException"
    assert _classify(named_in_body).reason == FailoverReason.unknown
    assert provider_stream_frame_code(ValueError("serviceUnavailableException")) is None
    assert provider_stream_frame_code(RuntimeError(str(_frame("serviceUnavailableException")))) is None


@pytest.fixture
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance.client = MagicMock()
    instance._cached_system_prompt = "You are helpful."
    instance._use_prompt_caching = False
    instance.tool_delay = 0
    instance.compression_enabled = False
    instance.save_trajectories = False
    instance._fallback_chain = []
    instance._api_max_retries = 3
    instance.provider = "bedrock"
    return instance


def _response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _run(agent, side_effect):
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=side_effect) as api_call,
        patch.object(agent, "_try_activate_fallback", return_value=False) as fallback,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_dump_api_request_debug"),
        patch.object(retry_utils, "jittered_backoff", return_value=0.0),
    ):
        return agent.run_conversation("hello"), api_call, fallback


def test_service_unavailable_frame_is_retried_on_the_same_provider(agent):
    error = _frame("serviceUnavailableException")
    result, api_call, fallback = _run(agent, [error, error, _response("Recovered")])
    assert api_call.call_count == 3
    assert fallback.call_count == 0  # the provider keeps its whole retry budget before any switch
    assert result["final_response"] == "Recovered"


def test_validation_frame_still_ends_as_a_client_error(agent):
    result, api_call, _ = _run(agent, _frame("validationException", "The input is malformed."))
    assert api_call.call_count == 1
    assert result["failed"] is True

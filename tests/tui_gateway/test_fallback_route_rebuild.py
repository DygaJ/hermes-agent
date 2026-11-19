"""The billing record must not override a TUI session's primary route."""
import json
from types import SimpleNamespace

import pytest

from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from hermes_constants import get_hermes_home
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session, seed_usage

PRIMARY = ('claude-fable-5-2', 'anthropic')
FALLBACK = ('global.anthropic.claude-fable-5-2', 'bedrock')
PICK = ('claude-sonnet-5', 'anthropic')
CHAIN = [{'model': FALLBACK[0], 'provider': FALLBACK[1]}, {'model': 'gpt-6.2-astra', 'provider': 'openai-codex'}]
SHAPES = ['nested', 'provider-only', 'billing-only', 'pick', 'switch']
pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')


def profile(home, chain):
    home.mkdir(parents=True, exist_ok=True)
    config = {'model': {'default': PRIMARY[0], 'provider': PRIMARY[1], 'api_mode': 'anthropic_messages',
                        'base_url': 'https://api.anthropic.com'}, 'fallback_providers': chain}
    (home / 'config.yaml').write_text(json.dumps(config))
    return config


def account_and_sync(store, sid, shape, switched):
    seed_usage(store, sid, model=FALLBACK[0], billing_provider=FALLBACK[1], input_tokens=4, api_call_count=1)
    row = store.sessions.get(sid)
    assert (row.model, row.billing_provider) == FALLBACK
    if shape == 'billing-only':
        assert not row.model_config
    if shape in ('nested', 'provider-only'):
        assert row.model_config == {'provider': PRIMARY[1]}
    model, provider = switched if shape == 'switch' else FALLBACK
    serving = SimpleNamespace(model=model, provider=provider, api_mode='anthropic_messages',
                              _fallback_activated=shape != 'switch')
    if shape in ('nested', 'pick', 'switch'):
        GatewayTurnPrepareMixin._sync_session_model_from_agent(SimpleNamespace(persistence=store), sid, serving)
        row = store.sessions.get(sid)
        runtime = row.model_config['gateway_runtime']
        assert (row.model, runtime['model'], runtime['provider'], runtime['fallback_active']) == (
            model, model, provider, shape != 'switch')
    return serving


def saved_row(shape, chain):
    home = get_hermes_home()
    profile(home, CHAIN)
    with open_persistence(profile_scope(home=home)) as db:
        config = {} if shape == 'billing-only' else {'provider': PRIMARY[1]}
        if shape == 'pick':
            config['model'] = PICK[0]
        seed_session(db, 'chat', 'tui', model=config.get('model', PRIMARY[0]), model_config=config)
        account_and_sync(db, 'chat', shape, FALLBACK)
        if chain != CHAIN:
            profile(home, chain)
        return session_row(db, 'chat'), {'pick': PICK, 'switch': FALLBACK}.get(shape, PRIMARY)


@pytest.fixture(params=SHAPES)
def saved_route(request):
    return saved_row(request.param, CHAIN)


def assert_tui_build(monkeypatch, row, expected, chain):
    import run_agent
    from tui_gateway.server import _make_agent, _stored_session_runtime_overrides, _profile_build_scope, _load_cfg
    from tests.gateway.test_fallback_route_rebuild import anthropic_request

    monkeypatch.setattr(run_agent, 'AIAgent', SimpleNamespace)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'offline-test')
    monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'offline-test')
    monkeypatch.setenv('AWS_REGION', 'us-east-1')
    with _profile_build_scope(get_hermes_home()):
        assert _load_cfg()['fallback_providers'] == chain
        built = _make_agent('chat', 'chat', **_stored_session_runtime_overrides(row))
    actual = (built.model, built.provider, built.fallback_model or [], anthropic_request(built))
    assert actual == (*expected, chain, expected[0] if expected[1] == 'anthropic' else None), actual


def test_tui_resume_primary(monkeypatch, saved_route):
    assert_tui_build(monkeypatch, *saved_route, CHAIN)


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['provider-only', 'billing-only', 'nested', 'pick', 'switch'])
async def test_tui_resume_preserves_receipt(monkeypatch, shape):
    from tests.gateway.test_fallback_route_rebuild import cold_turn

    async def resume(owner, sid, row):
        response = await owner.dispatch({'id': 'resume', 'method': 'session.resume', 'params': {'session_id': sid}})
        assert 'result' in response, response

    await cold_turn(monkeypatch, shape, resume)


def test_tui_resume_primary_when_chain_no_longer_lists_the_fallback(monkeypatch):
    assert_tui_build(monkeypatch, *saved_row('provider-only', []), [])

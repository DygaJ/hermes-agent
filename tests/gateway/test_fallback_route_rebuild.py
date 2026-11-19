"""A cold local gateway turn must retain its primary after accounting and route sync."""
import json
from types import SimpleNamespace

import pytest

from gateway.session_execution import GatewaySessionAgentMixin
from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from tests.gateway.test_session_follow_profile_config import _owner
from tests.fixtures.storage_read import session_row
from tests.tui_gateway.test_fallback_route_rebuild import account_and_sync, profile, PICK, PRIMARY, CHAIN

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['nested', 'provider-only', 'billing-only', 'pick', 'switch'])
async def test_cold_local_constructor_after_fallback_sync(monkeypatch, shape):
    await cold_turn(monkeypatch, shape)


async def cold_turn(monkeypatch, shape, resume=None):
    from gateway import run
    from gateway.session_local_recovery import restore_local_session
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    config = profile(home, CHAIN)
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'offline-test')
    store, runner, authority, owner = _owner(home)
    try:
        response = await owner.dispatch({'id': '1', 'method': 'session.create', 'params': {
            'request_id': '1', 'source': 'tui', 'cwd': str(home), 'toolsets': []}})
        assert 'result' in response, response
        sid = response['result']['session_id']
        db = runner.persistence = authority.persistence
        expected = PRIMARY[0]
        if shape in ('pick', 'switch'):
            expected = PICK[0]
            runner._resolve_session_agent_runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime.__get__(runner)
            snap = await owner.dispatch(dict(id='attach', method='session.resume', params={'session_id': sid}))
            result = await owner.dispatch(dict(id='pick', method='session.mutate', params=dict(
                session_id=sid, operation='model', request_id='pick', payload={'model': expected, 'provider': 'anthropic'},
                expected_revision=snap['result']['revision'], expected_generation=snap['result']['execution_generation'])))
            assert 'result' in result, result
        original = db.bindings.receipt(sid).policy
        before = launch_route(original)
        assert before == (expected, expected, 'anthropic')
        if shape != 'billing-only':
            db.sessions.patch_model_config(sid, {'provider': 'anthropic'})
        serving = account_and_sync(db, sid, shape, PICK)
        row = session_row(db, sid)
        source = authority.sessions[sid].source
        key = store._generate_session_key(source)
        runner._agent_cache = {key: (serving,)}
        ctx = SimpleNamespace(session_key=key, session_id=sid, source=source, AIAgent=SimpleNamespace,
                              enabled_toolsets=[], disabled_toolsets=[], user_config=config)
        turn = GatewaySessionAgentMixin()
        turn._ctx, turn._runner = ctx, runner
        assert turn._pop_cached_agent_for_eviction() is serving
        adapter = runner._adapter_for_source(source)
        adapter.forget(source, key)
        authority.sessions.pop(sid)
        if resume:
            runner._resolve_session_agent_runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime.__get__(runner)
            await resume(owner, sid, row)
        restore_local_session(authority, sid)

        ctx.source = source = authority.sessions[sid].source
        model, runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, source=source)
        runner._service_tier = None
        runner._refresh_fallback_model = run.GatewayRunner._refresh_fallback_model.__get__(runner)
        route = GatewayTurnPrepareMixin._resolve_turn_agent_config(runner, 'notice', model, runtime)
        built = turn._build_fresh_agent(route, 'tui', '', 10, None, {}, True)
        after = launch_route(db.bindings.receipt(sid).policy)
        observed = dict(constructor=(built.model, built.provider), chain=built.fallback_model,
                        anthropic_request=anthropic_request(built), launch=after)
        assert observed == dict(constructor=(expected, 'anthropic'), chain=CHAIN,
                                anthropic_request=expected, launch=before), observed
    finally:
        await owner.close()
        store.close_all_db_handles()


def launch_route(policy):
    config = json.loads(policy['config_json'])['model']
    return policy['model'], config['default'], config['provider']


def anthropic_request(built):
    if built.provider != 'anthropic':
        return None
    from agent.anthropic_adapter import build_anthropic_kwargs, create_anthropic_message
    kwargs = build_anthropic_kwargs(built.model, [{'role': 'user', 'content': 'test'}], None, 32, None)
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **request: request['model']))
    return create_anthropic_message(client, kwargs)

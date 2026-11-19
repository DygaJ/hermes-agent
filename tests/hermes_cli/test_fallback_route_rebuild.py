"""CLI --resume must not pin a turn-scoped fallback."""
import asyncio
from types import SimpleNamespace

import pytest

from hermes_cli.cli_model_switch_mixin import stored_session_route
from tests.tui_gateway import test_fallback_route_rebuild as tui_shapes

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')
PRIMARY = tui_shapes.PRIMARY
saved_route = tui_shapes.saved_route  # the shared fixture, registered in this module too


def test_cli_resume_primary(saved_route):
    row, expected = saved_route
    route = stored_session_route(row, current_model=PRIMARY[0], current_provider=PRIMARY[1])
    assert (route[:2] if route else PRIMARY) == expected


async def resume_cli(owner, sid, row):
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    from hermes_cli.gateway_agent import GatewayAgent
    from hermes_cli.gateway_connection import SessionState

    state = SessionState(sid)
    loop = asyncio.get_running_loop()

    def call(method, *, timeout=None, abandon=None, **params):
        request = dict(id='resume-test', method=method, params=params)
        result = asyncio.run_coroutine_threadsafe(owner.dispatch(request), loop).result(timeout=30)
        assert 'result' in result, result
        return result['result']

    connection = SimpleNamespace(state=lambda _: state, call=call, epoch=1)
    agent = await asyncio.to_thread(GatewayAgent, session_id=sid, connection=connection, model=PRIMARY[0], provider=PRIMARY[1])
    cli = SimpleNamespace(agent=agent, model=PRIMARY[0], provider=PRIMARY[1], base_url='', api_mode='', api_key='')
    await asyncio.to_thread(CLIModelSwitchMixin._restore_session_model, cli, row, quiet=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['provider-only', 'billing-only', 'nested', 'pick', 'switch'])
async def test_cli_resume_preserves_receipt(monkeypatch, shape):
    from tests.gateway.test_fallback_route_rebuild import cold_turn
    await cold_turn(monkeypatch, shape, resume_cli)

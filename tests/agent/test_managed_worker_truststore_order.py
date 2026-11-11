"""Regression for #126808: a managed worker installs platform trust before runtime imports.

``python -m agent.managed_worker`` never passes through ``hermes_cli.main``, so nothing
installed the platform trust store at its start. A worker whose first model call went to
Bedrock imported botocore first and got the lazy installation afterwards. botocore keeps
the ``ssl.SSLContext`` it finds at its own import, so every TLS context it built after
that recursed until the stack ran out, and the worker died on its next connection.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("truststore")
pytest.importorskip("botocore")

CHECKOUT_ROOT = Path(__file__).resolve().parents[2]

# The worker's own start: main() says hello, then fails on the empty stdin where the
# bootstrap frame should be and returns 1. The lines after it are what a Bedrock-backed
# turn does next: import the SDK, reach the lazy trust installation, open a connection.
CHILD = """
    import io
    import sys

    from agent import managed_worker

    sys.stdin = io.TextIOWrapper(io.BytesIO(b""))
    assert managed_worker.main() == 1

    from botocore.httpsession import create_urllib3_context

    from agent.ssl_verify import install_truststore

    install_truststore()
    create_urllib3_context()
"""


def test_worker_start_installs_platform_trust_before_the_provider_sdk_loads(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CHECKOUT_ROOT)
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(CHILD)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env=env,
        check=False,
    )
    assert "RecursionError" not in child.stderr, child.stderr[-2000:]
    assert child.returncode == 0, child.stdout + child.stderr[-2000:]

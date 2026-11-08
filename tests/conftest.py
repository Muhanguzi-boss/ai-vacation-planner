"""Test-session isolation from a developer's Datadog configuration.

``app.core.telemetry.telemetry`` builds its sinks from ``Settings``, which reads the real
``.env``. Without this file, ``DD_METRICS_ENABLED=true`` in a developer's ``.env`` (or shell)
would make ordinary tests send metrics to a local Datadog Agent and on to a real account.

Process environment variables take precedence over ``.env`` in pydantic-settings, so forcing
the variable here, before any ``app`` module is imported, keeps the shared telemetry on the
JSON log sink only. Production behaviour and ``.env`` are untouched. Datadog tests are not
affected: they build ``DogStatsDSink`` instances directly (with fake or loopback sockets) or
patch ``settings`` explicitly.
"""

import os

os.environ["DD_METRICS_ENABLED"] = "false"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _shared_telemetry_never_sends_to_datadog():
    """Fail any test that leaves a real DogStatsD sink on the shared telemetry object."""
    from app.core.telemetry import DogStatsDSink, telemetry

    yield
    leaked = [sink for sink in (telemetry._sinks or []) if isinstance(sink, DogStatsDSink)]
    if leaked:
        telemetry._sinks = None  # rebuilt from the (Datadog-disabled) settings on next use
        pytest.fail("Shared telemetry was left with a DogStatsD sink; tests must not send metrics.")

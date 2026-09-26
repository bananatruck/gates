"""The suite runs offline, and a test that tries to connect fails loudly.

D61 lets ``rig/`` hold live tools beside the model-free loops, on one
condition: the suite drives every one of them with a fake and never opens a
socket. This fixture is that condition made structural. A test that genuinely
needs the network is marked ``live`` and skipped unless asked for.
"""

from __future__ import annotations

import socket

import pytest


class OfflineError(RuntimeError):
    """A test reached for the network (D61)."""


def _refuse(*args, **kwargs):
    raise OfflineError("the test suite is offline (D61); mark a live test with @pytest.mark.live")


@pytest.fixture(autouse=True)
def _offline(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)

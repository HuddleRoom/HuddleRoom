"""Process-group probes tolerate transient zombie visibility without hiding signal failures."""
import signal
from types import SimpleNamespace

import pytest

from huddleroom.services.cli_streaming import terminate_process_group


@pytest.mark.asyncio
async def test_termination_retries_transient_probe_permission_error(monkeypatch):
    signals = []
    probes = iter([PermissionError("zombie group"), ProcessLookupError("reaped")])

    def killpg(pid, sig):
        signals.append(sig)
        if sig == 0:
            raise next(probes)

    monkeypatch.setattr("huddleroom.services.cli_streaming.os.killpg", killpg)
    await terminate_process_group(SimpleNamespace(pid=123, returncode=-15), grace_seconds=0.01)
    assert signals == [signal.SIGTERM, 0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("grace_seconds", [0, 0.01])
async def test_termination_preserves_persistent_probe_permission_error(monkeypatch, grace_seconds):
    signals = []

    def killpg(pid, sig):
        signals.append(sig)
        if sig == 0:
            raise PermissionError("denied")

    monkeypatch.setattr("huddleroom.services.cli_streaming.os.killpg", killpg)
    with pytest.raises(PermissionError, match="denied"):
        await terminate_process_group(SimpleNamespace(pid=123, returncode=-15), grace_seconds=grace_seconds)
    assert signals[0] == signal.SIGTERM
    assert signal.SIGKILL not in signals
    assert len(signals) <= 3


@pytest.mark.asyncio
async def test_termination_escalates_live_group_to_kill(monkeypatch):
    signals = []

    def killpg(pid, sig):
        signals.append(sig)
        if sig == 0 and signal.SIGKILL in signals:
            raise ProcessLookupError("killed")

    monkeypatch.setattr("huddleroom.services.cli_streaming.os.killpg", killpg)
    await terminate_process_group(SimpleNamespace(pid=123, returncode=-9), grace_seconds=0)
    assert signals == [signal.SIGTERM, 0, signal.SIGKILL, 0]


@pytest.mark.asyncio
async def test_probe_permission_error_at_deadline_never_escalates(monkeypatch):
    signals = []
    times = iter([0, 0.005, 0.02])

    def killpg(pid, sig):
        signals.append(sig)
        if sig == 0:
            raise PermissionError("still denied")

    async def sleep(delay):
        pass

    monkeypatch.setattr("huddleroom.services.cli_streaming.os.killpg", killpg)
    monkeypatch.setattr("huddleroom.services.cli_streaming.asyncio.get_running_loop",
                        lambda: SimpleNamespace(time=lambda: next(times)))
    monkeypatch.setattr("huddleroom.services.cli_streaming.asyncio.sleep", sleep)
    with pytest.raises(PermissionError, match="still denied"):
        await terminate_process_group(SimpleNamespace(pid=123, returncode=-15), grace_seconds=0.01)
    assert signals == [signal.SIGTERM, 0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("denied_signal", [signal.SIGTERM, signal.SIGKILL])
async def test_termination_does_not_hide_signal_permission_error(monkeypatch, denied_signal):
    def killpg(pid, sig):
        if sig == denied_signal:
            raise PermissionError("signal denied")

    monkeypatch.setattr("huddleroom.services.cli_streaming.os.killpg", killpg)
    with pytest.raises(PermissionError, match="signal denied"):
        await terminate_process_group(SimpleNamespace(pid=123, returncode=-15), grace_seconds=0)

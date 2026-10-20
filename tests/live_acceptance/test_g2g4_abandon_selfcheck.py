"""Self-check: a deliberate restart abandons unsettled calls whose shim process is gone."""
from types import SimpleNamespace

from tests.live_acceptance.test_live_run_g2g4 import G2G4Scenario


class _Shim:
    def __init__(self, live, ordinals, settled, executing=()):
        self._live, self._inv, self._settled, self._exec = live, ordinals, settled, set(executing)

    def live_pids(self):
        return self._live

    def invocations(self):
        return [SimpleNamespace(ordinal=o, exec_at=1.0 if o in self._exec else None) for o in self._inv]

    def settled_ordinals(self):
        return set(self._settled)


def test_abandon_dead_skips_live_shims():
    sc = object.__new__(G2G4Scenario)
    sc.abandoned = {"codex": set(), "claude": set()}
    sc.shims = {"codex": _Shim([], [1, 2, 3], [1]), "claude": _Shim([99], [1], [])}
    sc.abandon_dead()
    assert sc.abandoned == {"codex": {2, 3}, "claude": set()}


def test_abandon_dead_keeps_executing_unsettled_calls():
    sc = object.__new__(G2G4Scenario)
    sc.abandoned = {"codex": set(), "claude": set()}
    sc.shims = {"codex": _Shim([], [1, 2, 3], [1], executing=[3]), "claude": _Shim([], [], [])}
    sc.abandon_dead()
    assert sc.abandoned["codex"] == {2}

#!/usr/bin/env python3
"""Checks for cnfproof.py: the bias/unit normalisation never changes a move, and the exported CNF accepts exactly
the never-losing nets (pinned weights), so an UNSAT proof of that CNF is a statement about nets.

The normalisation tests need only the standard library. The CNF tests need python-sat (pip install -r requirements-sat.txt).

    python3 tictactoe/tests/test_cnfproof.py      PASS/FAIL per test with its time, exits 1 if any failed
"""
from __future__ import annotations

import copy
import os
import random
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cnfproof as C  # noqa: E402
import minsat as M  # noqa: E402
import net as N  # noqa: E402

W = os.path.join(ROOT, "weights")
P = M.Problem("xo", True)


def _parts(net):
    L = net["layers"]
    return L[0]["W"], L[0]["b"], L[1]["W"], L[1]["b"]


def saved_sym_nets():
    out = []
    for f in sorted(os.listdir(W)):
        net = N.load(os.path.join(W, f))
        if net["encoding"] == "trits" and net["symmetry"] and len(net["layers"]) == 2 and net["act"] == "relu":
            out.append((f, net))
    return out


def moves(parts):
    return [P.move(*parts, i) for i in range(len(P.boards))]


def random_net(rng, H, b1r, b2r):
    W1 = [[rng.choice((-1, 0, 0, 1)) for _ in range(9)] for _ in range(H)]
    W2 = [[rng.choice((-1, 0, 1)) for _ in range(H)] for _ in range(9)]
    return W1, [rng.randint(-b1r, b1r) for _ in range(H)], W2, [rng.randint(-b2r, b2r) for _ in range(9)]


def in_ranges(parts):
    W1, b1, W2, b2 = parts
    H = len(W1)
    for j in range(H):
        r = sum(1 for v in W1[j] if v)
        assert -r <= b1[j] <= r
    assert min(b2) == 0 and max(b2) <= 8 * (36 * H + 1)
    keys = [[b1[j]] + [W1[j][i] for i in range(8, -1, -1)] + [W2[k][j] for k in range(9)] for j in range(H)]
    assert all(keys[j] >= keys[j + 1] for j in range(H - 1))
    for j in range(H):  # inactive at some position (the positions step)
        assert any(sum(w * v for w, v in zip(W1[j], x)) + b1[j] <= 0 for x in P.x)


def test_normalise_saved_nets():
    for name, net in saved_sym_nets():
        parts = _parts(net)
        q = C.normalise(*parts, P=P)
        assert moves(q) == moves(parts), name
        in_ranges(q)


def test_normalise_random_wild_biases():
    """b1 far outside [-r, r] and b2 spreads in the thousands; also tiny spreads so ties matter."""
    rng = random.Random(7)
    for t in range(400):
        H = rng.randint(1, 4)
        parts = random_net(rng, H, rng.choice((2, 12, 40)), rng.choice((0, 1, 3, 50, 3000)))
        q = C.normalise(*parts, P=P)
        assert moves(q) == moves(parts), t
        in_ranges(q)


def have_pysat():
    try:
        import pysat  # noqa: F401
        return True
    except ImportError:
        return False


_CNF = {}


def cnf_solver(H, require_used):
    """(map, pysat solver loaded from the exported DIMACS file) — the file a proof is about, not the Python object."""
    key = (H, require_used)
    if key not in _CNF:
        from pysat.formula import CNF
        from pysat.solvers import Solver

        d = tempfile.mkdtemp(dir=None)
        mp, _ = C.build(H, "general", require_used, True, "binary", out=os.path.join(d, f"h{H}"))
        s = Solver(name="cadical195", bootstrap_with=CNF(from_file=os.path.join(d, f"h{H}.cnf")).clauses)
        _CNF[key] = (mp, s)
    return _CNF[key]


def active_somewhere(W1, b1, j):
    return any(sum(w * v for w, v in zip(W1[j], x)) + b1[j] > 0 for x in P.x)


def all_used(parts):
    W1, b1, W2, _ = parts
    return all(any(W1[j]) and any(W2[k][j] for k in range(9)) and b1[j] > -sum(1 for v in W1[j] if v)
               and active_somewhere(W1, b1, j) for j in range(len(W1)))


def pinned(parts, require_used=None):
    q = C.normalise(*parts, P=P)
    ru = all_used(q) if require_used is None else require_used
    mp, s = cnf_solver(len(q[0]), ru)
    ok = s.solve(assumptions=C.pin_units(mp, *q))
    model = s.get_model() if ok else None
    return ok, mp, q, model


def test_cnf_accepts_never_losing_nets():
    if not have_pysat():
        print("  skipped (no python-sat)")
        return
    for name, net in saved_sym_nets():
        if not N.verify(net)["perfect"]:
            continue
        ok, mp, q, model = pinned(_parts(net))
        assert ok, name
        true = {v for v in model if v > 0}
        assert C.decode(mp, true) == tuple(q) or list(C.decode(mp, true)) == list(q), name


def test_cnf_rejects_losing_variants():
    if not have_pysat():
        print("  skipped (no python-sat)")
        return
    for name, net in saved_sym_nets():
        if not N.verify(net)["perfect"]:
            continue
        found = False
        for li, r, c in [(li, r, c) for li, L in enumerate(net["layers"]) for r, row in enumerate(L["W"])
                         for c, v in enumerate(row) if v]:
            bad = copy.deepcopy(net)
            bad["layers"][li]["W"][r][c] = 0
            if not N.verify(bad)["perfect"]:
                parts = _parts(bad)
                ok, *_ = pinned(parts, require_used=False)
                assert not ok, name
                found = True
                break
        assert found, name


def test_cnf_agrees_with_reference_on_random_nets():
    if not have_pysat():
        print("  skipped (no python-sat)")
        return
    rng = random.Random(11)
    for t in range(60):
        H = rng.choice((1, 2, 3))
        parts = random_net(rng, H, 4, 20)
        ok, *_ = pinned(parts, require_used=False)
        assert ok == (not P.failures(*parts, "reach")), t


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__} ({time.time() - t0:.1f} s): {type(e).__name__}: {e}", flush=True)
    sys.exit(1 if bad else 0)

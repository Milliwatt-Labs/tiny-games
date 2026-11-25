#!/usr/bin/env python3
"""Checks for the exact search (minsat.py, satnet.py) and the X-only picks.

The Problem / reference parts need nothing but the standard library. The CNF checks need python-sat (pip install -r requirements-sat.txt); they are skipped without it.

    python3 -m pytest tictactoe/tests/test_minsat.py -q
    python3 tictactoe/tests/test_minsat.py      PASS/FAIL per test with its time, exits 1 if any failed
"""
from __future__ import annotations

import copy
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import game as G  # noqa: E402
import minsat as M  # noqa: E402
import net as N  # noqa: E402

W = os.path.join(ROOT, "weights")
ONE_LAYER_TRITS = [p for p in sorted(glob.glob(os.path.join(W, "*.json")))
                   if (lambda n: n["encoding"] == "trits" and len(n["layers"]) == 2 and n["act"] == "relu")(N.load(p))]


def _parts(net):
    L = net["layers"]
    return L[0]["W"], L[0]["b"], L[1]["W"], L[1]["b"]


def have_pysat():
    try:
        import pysat  # noqa: F401
        return True
    except ImportError:
        return False


def test_problem_counts():
    assert len(M.Problem("xo", True).boards) == 627
    assert len(M.Problem("x", True).boards) == 338
    assert len(M.Problem("xo", False).boards) == 4520
    assert len(M.Problem("x", False).boards) == 2423


def test_reference_matches_net_py():
    """Problem.move is net.choose_move in the net's frame, and `reach` failures are exactly net.verify losses."""
    for path in ONE_LAYER_TRITS:
        net = N.load(path)
        P = M.Problem("xo", net["symmetry"])
        parts = _parts(net)
        for i, b in enumerate(P.boards[::7]):
            i = P.index[b]
            assert P.move(*parts, i) == N._argmax_legal(N.forward_int(net, P.x[i]), b)
        res = N.verify(net)
        fails = P.failures(*parts, "reach")
        assert (not fails) == res["perfect"], path
        Px = M.Problem("x", net["symmetry"])
        assert (not Px.failures(*parts, "reach")) == (res["as_x"]["loss"] == 0), path


def test_reach_detects_a_losing_net():
    net = N.load(os.path.join(W, "smallest-never-loses.json"))
    P = M.Problem("xo", True)
    for li, r, c in [(li, r, c) for li, L in enumerate(net["layers"]) for r, row in enumerate(L["W"])
                     for c, v in enumerate(row) if v]:
        bad = copy.deepcopy(net)
        bad["layers"][li]["W"][r][c] = 0
        if not N.verify(bad)["perfect"]:
            assert P.failures(*_parts(bad), "reach")
            return
    raise AssertionError("no single-weight deletion breaks the 63 net")


def test_xonly_picks():
    """weights/xonly-*.json never lose as X (full game tree, pure-Python reference)."""
    paths = sorted(glob.glob(os.path.join(W, "xonly-*.json")))
    assert paths
    for p in paths:
        net = N.load(p)
        res = N.verify(net, fast=False)
        assert res["as_x"]["loss"] == 0, p
        claim = net.get("verified", {})
        if "as_x" in claim:
            assert claim["as_x"] == res["as_x"], p
        if "size" in net:
            assert N.size(net) == net["size"], p


def test_bias_only_priority_order():
    """No weights: a fixed cell order on the canonical board never loses as X (priority.py found 146 of 9!)."""
    import priority as PR

    PR._init()
    assert PR.never_loses((4, 3, 5, 8, 2, 6, 7, 0, 1), G.X, True)
    assert not PR.never_loses((4, 3, 5, 8, 2, 6, 7, 0, 1), G.O, True)
    assert not PR.never_loses((4, 3, 5, 8, 2, 6, 7, 0, 1), G.X, False)


def _sat_accepts(net, sides, cond, enc, bias="small"):
    import satnet as S

    P = M.Problem(sides, net["symmetry"])
    H = len(net["layers"][0]["b"])
    sn = S.SatNet(P, H, cond, bias, 3, 6, enc=enc)
    for i in range(len(P.boards)):
        sn.add_position(i)
    W1, b1, W2, b2 = _parts(net)
    # the encoding sorts hidden units by key; give the net in that order
    order = sorted(range(H), key=lambda j: _key(W1[j], b1[j], sn.l1), reverse=True)
    W1 = [W1[j] for j in order]
    b1 = [b1[j] for j in order]
    W2 = [[row[j] for j in order] for row in W2]
    return sn.solve(sn.fix_net(W1, b1, W2, b2))


def _key(row, b, l1):
    v = b - l1
    for i in range(8, -1, -1):
        v = v * 4 + {-1: 0, 0: 1, 1: 3}[row[i]]  # (p, not n) as a 2-bit number, cell 8 most significant
    return v


def _first_losing_variant(net, sides):
    """The net with its first nonzero weight zeroed that makes it lose (as X for sides "x", else as X or O)."""
    for li, r, c in [(li, r, c) for li, L in enumerate(net["layers"]) for r, row in enumerate(L["W"])
                     for c, v in enumerate(row) if v]:
        bad = copy.deepcopy(net)
        bad["layers"][li]["W"][r][c] = 0
        res = N.verify(bad)
        if (res["as_x"]["loss"] if sides == "x" else not res["perfect"]):
            return bad
    raise AssertionError("no single-weight deletion makes the net lose")


def test_cnf_accepts_known_nets():
    """The CNF must accept every saved net it can express (binary for the both-sides nets, both encodings for
    the X-only one); a wrong circuit would turn true UNSAT claims into false ones."""
    if not have_pysat():
        print("  (skipped: no python-sat)")
        return
    for name in ("smallest-never-loses-params.json", "smallest-never-loses.json"):
        assert _sat_accepts(N.load(os.path.join(W, name)), "xo", "reach", "binary") is True, name
    net = N.load(os.path.join(W, "xonly-raw-H1.json"))
    for enc in ("binary", "unary"):
        assert _sat_accepts(net, "x", "reach", enc) is True, enc
        assert _sat_accepts(net, "x", "reach", enc, bias="general") is True, enc


def test_cnf_rejects_losing_nets():
    if not have_pysat():
        print("  (skipped: no python-sat)")
        return
    bad = _first_losing_variant(N.load(os.path.join(W, "smallest-never-loses.json")), "xo")
    assert _sat_accepts(bad, "xo", "reach", "binary") is False
    bad = _first_losing_variant(N.load(os.path.join(W, "xonly-raw-H1.json")), "x")
    for enc in ("binary", "unary"):
        assert _sat_accepts(bad, "x", "reach", enc) is False, enc


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]:
        t = time.time()
        try:
            fn()
            print(f"PASS {name} {time.time() - t:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name} {time.time() - t:.1f}s: {e!r}", flush=True)
    sys.exit(1 if failed else 0)

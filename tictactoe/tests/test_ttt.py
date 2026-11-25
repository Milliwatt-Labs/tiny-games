#!/usr/bin/env python3
"""Exhaustive re-check of the saved tic-tac-toe nets with the integer-only reference (no torch, no numpy needed
for the reference path).

    python3 -m pytest tictactoe/tests -q
    python3 tictactoe/tests/test_ttt.py     PASS/FAIL per test with its time, exits 1 if any failed
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import game as G  # noqa: E402
import net as N  # noqa: E402

WEIGHTS = sorted(glob.glob(os.path.join(ROOT, "weights", "*.json")))


def test_solver_counts():
    assert len(G.all_positions()) == 5478
    dec = G.decision_positions()
    assert len(dec) == 4520
    assert G.value(G.START) == 0
    assert len({G.canonical(b)[0] for b in dec}) == 627
    # a symmetry maps optimal moves to optimal moves
    for b in dec[::37]:
        for p in G.SYMS:
            t = G.transform(b, p)
            assert sorted(p[m] for m in G.optimal_moves(t)) == sorted(G.optimal_moves(b))


def test_known_positions():
    # X to win at once on the top row
    b = (1, 1, 0, 2, 2, 0, 0, 0, 0)
    assert G.value(b) == 1 and G.optimal_moves(b) == [2]
    # after X takes a corner, O must take the centre
    assert G.optimal_moves((1, 0, 0, 0, 0, 0, 0, 0, 0)) == [4]


def test_weights_exist():
    names = {os.path.basename(p) for p in WEIGHTS}
    assert "smallest-never-loses.json" in names, names


def _check_one(path):
    net = N.load(path)  # also checks ternary weights and integer biases
    res = N.verify(net, fast=False)  # pure-Python integer reference on every position
    claim = net.get("verified", {})
    if claim.get("perfect", True):
        assert res["perfect"], (path, N.summary(res))
    assert res["as_x"]["loss"] == 0 or not claim.get("perfect", True)
    for k in ("takes_wins", "converts", "finisher", "converter"):
        if claim.get(k):
            assert res[k], (path, k, N.summary(res))
    if claim.get("converter"):
        assert not N.converts_by_play(lambda b: N.choose_move(net, b)), path
    for k in ("optimal", "tree_optimal", "tree_positions", "win_now_taken", "won_kept"):
        if k in claim:
            assert res[k] == claim[k], (path, k, res[k], claim[k])
    if "size" in net:
        assert N.size(net) == net["size"], path
    return res


def test_saved_nets_exhaustive():
    assert WEIGHTS
    for p in WEIGHTS:
        _check_one(p)


def test_finisher_picks():
    """The 2026-11-21 follow-up picks: never lose, and take every immediate win in all 4,520 positions."""
    path = os.path.join(ROOT, "weights", "smallest-finisher.json")
    net = N.load(path)
    assert net["criterion"] == "finisher" and net["verified"]["finisher"]
    res = N.verify(net, fast=False)
    assert res["perfect"] and res["takes_wins"] and res["win_now_positions"] == 2358
    conv = os.path.join(ROOT, "weights", "smallest-converter.json")
    if os.path.exists(conv):
        c = N.load(conv)
        res = N.verify(c, fast=False)
        assert res["converter"] and res["won_positions"] == 2836
        assert not N.converts_by_play(lambda b: N.choose_move(c, b))


def test_game_2_4_6():
    # X holds 4 and 6 (0-indexed), so 2 completes the 2-4-6 diagonal. The 63-weight net plays 5 here (in its own
    # game tree as X); the finisher must win.
    b = (0, 0, 0, 0, 1, 0, 1, 2, 2)
    assert G.winning_moves(b) == [2]
    old = N.load(os.path.join(ROOT, "weights", "smallest-never-loses.json"))
    assert N.choose_move(old, b) == 5
    assert not N.verify(old)["takes_wins"]
    assert N.choose_move(N.load(os.path.join(ROOT, "weights", "smallest-finisher.json")), b) == 2


def test_convert_check_agrees_with_play():
    # keeping the value in every won position <=> winning from every won position against every reply
    for p in WEIGHTS:
        net = N.load(p)
        res = N.verify(net)
        bad = N.converts_by_play(lambda b: N.choose_move(net, b))
        assert res["converts"] == (not bad), p


def test_batched_matches_reference():
    try:
        import numpy  # noqa: F401
    except ImportError:
        return
    boards = G.decision_positions()
    for p in WEIGHTS:
        net = N.load(p)
        assert N.moves_batch(net, boards) == [N.choose_move(net, b) for b in boards], p


def test_losing_net_is_caught():
    # an all-zero net plays the lowest empty cell; it must be reported as losing
    net = {"format": N.FORMAT, "encoding": "planes", "symmetry": False, "act": "relu",
           "layers": [{"W": [[0] * 18 for _ in range(9)], "b": [0] * 9}]}
    res = N.verify(net, fast=False)
    assert not res["perfect"] and res["as_o"]["loss"] > 0


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        t = time.time()
        try:
            fn()
            print(f"PASS {name} {time.time() - t:.2f}s")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name} {time.time() - t:.2f}s: {e!r}")
    sys.exit(1 if failed else 0)

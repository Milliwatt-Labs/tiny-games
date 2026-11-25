"""Bias-only "nets" (no weights, H = 0): the move is the first empty cell in a fixed priority order, on the
canonical board (symmetry) or the raw board. Enumerates all 9! = 362,880 orders and counts the ones that never
lose as X, as O and as both, checked over the full game tree. Pure Python; ~minutes with --jobs.

    python3 priority.py --jobs 8 > results/priority.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool

import game as G

DEC = None
CAN = None


def _init():
    global DEC, CAN
    DEC = G.decision_positions()
    CAN = {b: G.canonical(b) for b in DEC}


def never_loses(order, side, sym):
    rank = {c: r for r, c in enumerate(order)}

    def move(b):
        if sym:
            cb, p = CAN[b]
            return p[min(G.legal(cb), key=rank.__getitem__)]
        return min(G.legal(b), key=rank.__getitem__)

    seen = {}

    def ok(b):  # True iff no line from b (side to play somewhere) ends in a loss for `side`
        if b in seen:
            return seen[b]
        w = G.winner(b)
        if w:
            r = w == side
        elif all(b):
            r = True
        elif G.to_move(b) == side:
            r = ok(G.play(b, move(b)))
        else:
            r = all(ok(G.play(b, m)) for m in G.legal(b))
        seen[b] = r
        return r

    return ok(G.START)


def check(order):
    return tuple(never_loses(order, side, sym) for sym in (True, False) for side in (G.X, G.O))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    orders = list(itertools.permutations(range(9)))
    with Pool(a.jobs, initializer=_init) as pool:
        res = pool.map(check, orders, chunksize=2000)
    names = ["sym_x", "sym_o", "raw_x", "raw_o"]
    out = {"orders": len(orders)}
    for t, n in enumerate(names):
        good = [o for o, r in zip(orders, res) if r[t]]
        out[n] = len(good)
        out[n + "_first"] = list(good[0]) if good else None
    both = [o for o, r in zip(orders, res) if r[0] and r[1]]
    out["sym_xo"] = len(both)
    out["raw_xo"] = sum(1 for r in res if r[2] and r[3])
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

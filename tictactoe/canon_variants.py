"""How much the 146 of priority.py depends on which canonical form is used.

game.canonical takes the lexicographically smallest of the 8 images of the board, with cells in row-major order
from the top left and the absolute coding empty 0 < X 1 < O 2; on a board that is its own image under several
symmetries, the first such permutation in game.SYMS maps the move back. Here the same 9! fixed-order count (never
loses as X, full game tree) is repeated with other choices that "lexicographically smallest image" also allows: the
largest image instead, X and O swapped in the order, and the empty cell ranked above both pieces.

    nice -n 19 python3 canon_variants.py --jobs 6      -> results/canon-variants.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import game as G  # noqa: E402

VARIANTS = {  # name: (symbol rank for empty, X, O; pick the max image instead of the min)
    "paper: min, empty < X < O": ((0, 1, 2), False),
    "max, empty < X < O": ((0, 1, 2), True),
    "min, empty < O < X": ((0, 2, 1), False),
    "min, X < O < empty": ((2, 0, 1), False),
    "min, O < X < empty": ((2, 1, 0), False),
}
CAN = None


def canon(b, rank, use_max):
    best = None
    for p in G.SYMS:
        t = tuple(rank[v] for v in G.transform(b, p))
        if best is None or (t > best[0] if use_max else t < best[0]):
            best = (t, p)
    return best[1]


def _init(rank, use_max):
    global CAN
    CAN = {}
    for b in G.decision_positions():
        p = canon(b, rank, use_max)
        CAN[b] = (G.transform(b, p), p)


def never_loses_x(order):
    rank = {c: r for r, c in enumerate(order)}
    seen = {}

    def ok(b):
        if b in seen:
            return seen[b]
        w = G.winner(b)
        if w:
            r = w == G.X
        elif all(b):
            r = True
        elif G.to_move(b) == G.X:
            cb, p = CAN[b]
            r = ok(G.play(b, p[min(G.legal(cb), key=rank.__getitem__)]))
        else:
            r = all(ok(G.play(b, m)) for m in G.legal(b))
        seen[b] = r
        return r

    return ok(G.START)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    orders = list(itertools.permutations(range(9)))
    out = {}
    for name, (rank, use_max) in VARIANTS.items():
        with Pool(a.jobs, initializer=_init, initargs=(rank, use_max)) as pool:
            res = pool.map(never_loses_x, orders, chunksize=4000)
        out[name] = sum(res)
        print(name, out[name], flush=True)
    with open(os.path.join(HERE, "results", "canon-variants.json"), "w") as f:
        json.dump({"what": "fixed orders (of 362,880) that never lose as X on the canonical board, by canonical form",
                   "counts": out}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()

"""Exhaustively check a saved net, or play against it.

    python3 verify.py weights/smallest-never-loses.json           exit 0 iff the net never loses as X or O
    python3 verify.py weights/smallest-never-loses.json --play    you play X (or --o to play O); cells 1-9
"""
from __future__ import annotations

import argparse
import json
import sys

import game as G
import net as N


def play(net, human):
    b = G.START
    while not G.is_terminal(b):
        print(G.show(b), "\n")
        if G.to_move(b) == human:
            try:
                m = int(input("your move (1-9): ")) - 1
            except (ValueError, EOFError):
                return 1
            if not (0 <= m < 9) or b[m]:
                print("not an empty cell")
                continue
        else:
            m = N.choose_move(net, b)
            print(f"net plays {m + 1}")
        b = G.play(b, m)
    print(G.show(b))
    w = G.winner(b)
    print("draw" if not w else ("you win" if w == human else "net wins"))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--o", action="store_true", help="play O (the net opens)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    net = N.load(a.weights)
    if a.play:
        return play(net, G.O if a.o else G.X)
    res = N.verify(net, fast=False)
    if a.json:
        print(json.dumps({"size": N.size(net), **res}))
    else:
        print(N.size(net))
        print(N.summary(res))
    return 0 if res["perfect"] else 1


if __name__ == "__main__":
    sys.exit(main())

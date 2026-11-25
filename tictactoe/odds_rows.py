#!/usr/bin/env python3
"""LaTeX rows of the exact-probability table (main.tex, between %ODDS-ROWS-BEGIN and %ODDS-ROWS-END), from
tictactoe/results/exact-odds.json. check_numbers.py asserts that the block in main.tex equals rows().

    python3 odds_rows.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "results", "exact-odds.json")
OPPS = [("random", "random", 0.0), ("kid, $p=0.5$", "kid", 0.5), ("kid, $p=0.9$", "kid", 0.9),
        ("mixed, $q=0.25$", "mix", 0.25), ("mixed, $q=0.5$", "mix", 0.5), ("mixed, $q=0.75$", "mix", 0.75),
        ("mixed, $q=0.9$", "mix", 0.9)]
COLS = [("finisher", "X"), ("finisher", "O"), ("satxo", "X"), ("satxo", "O"), ("xonly", "X")]


def load():
    return json.load(open(SRC))


def rows():
    d = load()
    arms = {(a["net"], a["side"], a["opponent"], a["p"]): a for a in d["arms"] if a["group"] == "B"}
    out = []
    for label, kind, p in OPPS:
        cells = []
        for net, side in COLS:
            w = arms[(net, side, kind, p)]["exact"]["wins"]
            cells.append(f"{w['p_ideal']:.4f} ({w['z']:+.1f})".replace("-", "$-$").replace("+", "$+$"))
        out.append(f"{label} & " + " & ".join(cells) + r" \\")
    return out


if __name__ == "__main__":
    print("\n".join(rows()))

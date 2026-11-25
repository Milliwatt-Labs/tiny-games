#!/usr/bin/env python3
"""What a move costs outside the parameter count.

Writes results/cost.json (the paper's preprocessing and size paragraphs read it through check_numbers.py):

1. Preprocessing per move, counted over all 4,520 positions with a move to make, exactly as net.choose_move does it:
   - canonicalisation: the 8 images of the board (7 non-identity permutations of 9 cells), the lexicographically
     smallest kept by sequential comparison with early exit (cell comparisons counted), and one table lookup to map
     the chosen cell back;
   - legal mask: one emptiness test per cell (9);
   - argmax with lowest-index ties: (legal cells - 1) score comparisons, strict ">" so the first maximum wins.
   Also: in how many positions the move is decided by the tie rule (two or more legal cells share the top score),
   and in how many the mask decides (an occupied cell scores higher than every legal cell).
2. Network arithmetic per move: additions for the nonzero weights and biases, ReLU per hidden unit.

    python3 cost.py            -> results/cost.json, and a short table on stdout
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import game as G  # noqa: E402
import net as N  # noqa: E402

NETS = {"85": "smallest-never-loses-h4.json", "104": "smallest-never-loses-h5.json",
        "560": "smallest-never-loses-nosym.json", "123": "smallest-never-loses-nosym-sat.json", "142": "smallest-never-loses-nosym-sat-h7.json",
        "47-x-only": "smallest-x-only.json"}

def lexmin_cost(b):
    """Cell comparisons to find the lexicographically smallest of the 8 images (sequential, early exit), as in
    game.canonical (which takes min over the images)."""
    imgs = [G.transform(b, p) for p in G.SYMS]
    best = imgs[0]
    comps = 0
    for im in imgs[1:]:
        for a, c in zip(im, best):
            comps += 1
            if a != c:
                if a < c:
                    best = im
                break
    assert best == G.canonical(b)[0]
    return comps


def net_costs(net):
    boards = G.decision_positions()
    sym = net.get("symmetry")
    W1, b1 = net["layers"][0]["W"], net["layers"][0]["b"]
    W2, b2 = net["layers"][1]["W"], net["layers"][1]["b"]
    H = len(W1)
    pre_max = score_max = 0
    ties = masked = 0
    min_gap = None
    if net.get("x_only"):
        boards = [b for b in boards if G.to_move(b) == G.X]
    for b in boards:
        cb = G.canonical(b)[0] if sym else b
        x = G.encode(cb, "trits")
        a = [sum(w * v for w, v in zip(r, x)) + bb for r, bb in zip(W1, b1)]
        h = [max(0, v) for v in a]
        z = [sum(w * v for w, v in zip(r, h)) + bb for r, bb in zip(W2, b2)]
        pre_max = max(pre_max, max(abs(v) for v in a))
        legal = G.legal(cb)
        top = max(z[k] for k in legal)
        score_max = max(score_max, max(abs(z[k] - min(b2)) for k in range(9)))
        ties += sum(1 for k in legal if z[k] == top) > 1
        masked += max(z) > top
        rest = sorted((z[k] for k in legal), reverse=True)
        if len(rest) > 1 and rest[0] > rest[1]:
            min_gap = rest[0] - rest[1] if min_gap is None else min(min_gap, rest[0] - rest[1])
    nz1 = sum(1 for r in W1 for v in r if v)
    nz2 = sum(1 for r in W2 for v in r if v)
    nzb1 = sum(1 for v in b1 if v)
    med = sorted(b2)[4]
    b2s = [v - med for v in b2]
    nzb2 = sum(1 for v in b2s if v)
    size = N.size(net)
    return {
        "file": None, "symmetry": bool(sym), "positions": len(boards), "hidden": H,
        "dense_params": size["params"], "nonzero_weights": size["nonzero"], "nonzero_W1": nz1, "nonzero_W2": nz2,
        "nonzero_biases_stored": size["nonzero_biases"], "nonzero_b1": nzb1, "nonzero_b2_after_median_shift": nzb2,
        "adds_per_move": nz1 + nzb1 + nz2 + sum(1 for v in b2 if v), "relu_per_move": H,
        "max_abs_preactivation": pre_max, "max_abs_score_above_min_bias": score_max,
        "min_winning_gap_when_untied": min_gap, "positions_decided_by_tie_rule": ties, "positions_where_mask_decides": masked,
    }


def main():
    boards = G.decision_positions()
    lm = [lexmin_cost(b) for b in boards]
    legal = [len(G.legal(b)) for b in boards]
    pre = {
        "positions": len(boards),
        "canonicalisation": {"images": 8, "permutation_reads": 7 * 9, "cell_comparisons_mean": round(sum(lm) / len(lm), 2),
                             "cell_comparisons_max": max(lm), "cell_comparisons_worst_case": 7 * 9,
                             "map_back_lookups": 1},
        "legal_mask": {"emptiness_tests": 9},
        "argmax_tie_break": {"score_comparisons_mean": round(sum(n - 1 for n in legal) / len(legal), 2),
                             "score_comparisons_max": max(legal) - 1},
    }
    nets = {}
    for k, f in NETS.items():
        c = net_costs(N.load(os.path.join(HERE, "weights", f)))
        c["file"] = f
        nets[k] = c
    out = {"preprocessing_per_move": pre, "nets": nets}
    with open(os.path.join(HERE, "results", "cost.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(json.dumps(pre))
    for k, c in nets.items():
        print(f"{k:10} params {c['dense_params']:4} nonzero {c['nonzero_weights']:4} ties {c['positions_decided_by_tie_rule']}"
              f" mask {c['positions_where_mask_decides']} max|a| {c['max_abs_preactivation']} max|z| "
              f"{c['max_abs_score_above_min_bias']} gap {c['min_winning_gap_when_untied']}")


if __name__ == "__main__":
    main()

"""Exact outcome probabilities of the game-play arms (humanity/humanity.py) by recursion over the game tree.

The networks are deterministic, so for a given opponent model the probability of each outcome from a position s is

    P(s) = P(outcome of s)                                  if s is terminal
    P(s) = P(T(s, net move))                                if the net is to move
    P(s) = sum_a q(a | s) P(T(s, a))                        if the opponent is to move

with q the opponent's move distribution. Two versions of q are computed, both exactly with rationals:

    ideal        the stated model: "uniform" is 1/n, "with probability p" is p.
    implemented  what the GPU kernel and the Python reference player actually draw: a 32-bit uniform u, "with
                 probability p" is u < round(p 2^32), and the i-th of n cells is picked when floor(u n / 2^32) = i,
                 so a cell has probability (ceil((i+1) 2^32 / n) - ceil(i 2^32 / n)) / 2^32.

For each arm the output gives both probabilities of a net win, a draw and a net loss (for group C: X win, draw, O win,
plus the two counts reported in the paper), the GPU count, the expected count n p under the implemented model, and
z = (count - n p) / sqrt(n p (1 - p)).

    python3 exact_odds.py            -> results/exact-odds.json, and a table on stdout
"""
from __future__ import annotations

import json
import math
import os
import sys
from fractions import Fraction as F
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import game as G  # noqa: E402
import net as N  # noqa: E402

NETS = {"finisher": "smallest-finisher.json", "satxo": "smallest-sat-xo.json", "xonly": "smallest-x-only.json"}
TWO32 = 1 << 32
OUTCOMES = ("x", "draw", "o")  # X wins, draw, O wins


def thresh(p):  # same as humanity.thresh
    return min(int(round(p * TWO32)), TWO32 - 1)


def pick_dist(cells, mode):
    cells = sorted(cells)
    n = len(cells)
    if mode == "ideal":
        return {c: F(1, n) for c in cells}
    ceil = lambda a: -((-a) // n)  # noqa: E731  ceil(a / n)
    return {c: F(ceil((i + 1) * TWO32) - ceil(i * TWO32), TWO32) for i, c in enumerate(cells)}


def prob(p, mode):
    return F(p).limit_denominator(1000) if mode == "ideal" else F(thresh(p), TWO32)


def completing_for(b, side):
    out = []
    for m in G.legal(b):
        s = list(b)
        s[m] = side
        if G.winner(tuple(s)) == side:
            out.append(m)
    return out


def opp_dist(kind, p, b, mode):
    """Move distribution of humanity.Reference.opp_move."""
    me = G.to_move(b)
    out = {}

    def add(d, w):
        for m, q in d.items():
            out[m] = out.get(m, 0) + w * q

    if kind == "perfect":
        return pick_dist(G.optimal_moves(b), mode)
    if kind == "random":
        return pick_dist(G.legal(b), mode)
    t = prob(p, mode)
    if kind == "kid":
        rest = F(1)
        w = completing_for(b, me)
        if w:
            add(pick_dist(w, mode), rest * t)
            rest *= 1 - t
        bl = completing_for(b, G.O if me == G.X else G.X)
        if bl:
            add(pick_dist(bl, mode), rest * t)
            rest *= 1 - t
        add(pick_dist(G.legal(b), mode), rest)
        return out
    if kind == "mix":
        add(pick_dist(G.optimal_moves(b), mode), t)
        add(pick_dist(G.legal(b), mode), 1 - t)
        return out
    raise ValueError(kind)


def solve_arm(nets, arm, mode):
    net = nets[arm["net"]]
    side = {"X": G.X, "O": G.O}.get(arm["side"], 0)
    kind, p = arm["opponent"], arm["p"]

    @lru_cache(maxsize=None)
    def rec(b):
        if G.is_terminal(b):
            w = G.winner(b)
            return (F(int(w == G.X)), F(int(w is None or w == G.EMPTY)), F(int(w == G.O)))
        mover = G.to_move(b)
        if side == 0 or side == mover:
            return rec(G.play(b, N.choose_move(net, b)))
        acc = [F(0)] * 3
        for m, q in opp_dist(kind, p, b, mode).items():
            r = rec(G.play(b, m))
            for k in range(3):
                acc[k] += q * r[k]
        return tuple(acc)

    if arm["group"] != "C":
        return rec(G.START), None
    # C: open_k uniformly random opening moves, then the net plays both sides. Also the two reported counts.
    lost, notconv = F(0), F(0)
    tot = [F(0)] * 3

    def opening(b, k, w):
        nonlocal lost, notconv
        if k == 0 or G.is_terminal(b):
            r = rec(b)
            for i in range(3):
                tot[i] += w * r[i]
            v = G.value(b) if not G.is_terminal(b) else None
            if v is not None:
                vx = v if G.to_move(b) == G.X else -v  # value for X
                # lost from a draw or better: the side that lost had value >= 0 after the opening
                lost += w * ((r[2] if vx >= 0 else 0) + (r[0] if vx <= 0 else 0))
                notconv += w * (r[1] if vx != 0 else 0)
            return
        for m, q in pick_dist(G.legal(b), mode).items():
            opening(G.play(b, m), k - 1, w * q)

    opening(G.START, arm["open_k"], F(1))
    return tuple(tot), (lost, notconv)


def main():
    nets = {k: N.load(os.path.join(HERE, "weights", v)) for k, v in NETS.items()}
    hu = json.load(open(os.path.join(HERE, "results", "humanity.json")))
    arms = hu["arms"]
    out = []
    for a in arms:
        row = {k: a[k] for k in ("name", "group", "net", "side", "opponent", "p", "open_k", "games")}
        res = {}
        for mode in ("ideal", "implemented"):
            pr, extra = solve_arm(nets, a, mode)
            res[mode] = (pr, extra)
        n = a["games"]
        pr, extra = res["implemented"]
        ideal, ideal_extra = res["ideal"]
        if a["group"] == "C":
            names = ("x_wins", "draws", "o_wins")
            got = {"x_wins": a.get("x_wins"), "draws": a["draws"], "o_wins": a.get("o_wins")}
            probs = dict(zip(names, pr))
            iprobs = dict(zip(names, ideal))
            for nm, gk in (("lost_from_draw_or_better", 0), ("won_not_converted", 1)):
                probs[nm], iprobs[nm], got[nm] = extra[gk], ideal_extra[gk], a.get(nm)
        else:
            netside = {"X": 0, "O": 2}[a["side"]]
            probs = {"wins": pr[netside], "draws": pr[1], "losses": pr[2 - netside]}
            iprobs = {"wins": ideal[netside], "draws": ideal[1], "losses": ideal[2 - netside]}
            got = {"wins": a["wins"], "draws": a["draws"], "losses": a["losses"]}
        row["exact"] = {}
        for k, q in probs.items():
            e = {"p_ideal": float(iprobs[k]), "p_implemented": float(q), "p_ideal_exact": str(iprobs[k]),
                 "gpu": got[k], "expected": float(n * q)}
            if got[k] is not None:
                sd = math.sqrt(float(n * q * (1 - q)))
                e["z"] = None if sd == 0 else (got[k] - float(n * q)) / sd
                e["exact_match"] = sd == 0 and got[k] == n * q
            row["exact"][k] = e
        out.append(row)
        zs = [f"{k} {v.get('z'):+.2f}" if v.get("z") is not None else f"{k} exact" for k, v in row["exact"].items()
              if "z" in v or v.get("exact_match")]
        print(f"{a['name']:22s} " + "  ".join(f"{k}={v['p_ideal']:.6f}" for k, v in row["exact"].items()) + "   z: "
              + ", ".join(zs), flush=True)
    # one z per B arm (net wins; draws are the complement, losses have probability 0): independent arms
    zb = [r["exact"]["wins"]["z"] for r in out if r["group"] == "B"]
    summary = {"arms": len(out), "B_arms": len(zb), "B_max_abs_z_wins": max(abs(z) for z in zb),
               "B_chi2_wins": sum(z * z for z in zb), "B_chi2_dof": len(zb),
               "B_within_2sd": sum(abs(z) < 2 for z in zb),
               "A_exact": all(r["exact"][k].get("exact_match") for r in out if r["group"] == "A"
                              for k in ("wins", "draws", "losses")),
               "max_abs_ideal_minus_implemented": max(abs(v["p_ideal"] - v["p_implemented"]) for r in out
                                                     for v in r["exact"].values()),
               "zero_probability_losses": all(r["exact"]["losses"]["p_ideal_exact"] == "0"
                                              for r in out if r["group"] != "C")}
    print(json.dumps(summary, indent=1))
    with open(os.path.join(HERE, "results", "exact-odds.json"), "w") as f:
        json.dump({"what": __doc__.split("\n")[0], "summary": summary, "arms": out}, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()

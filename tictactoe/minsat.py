"""Exact search for the smallest 1-hidden-layer ternary ReLU net that never loses at tic-tac-toe (CP-SAT).

The model class is the one in net.py, restricted to one hidden layer and the trits encoding:

    h = relu(W1 x + b1),  z = W2 h + b2,  move = argmax of z over the empty cells, lowest index on ties,

W1 (H x 9) and W2 (9 x H) in {-1, 0, +1}, b1 and b2 integers, x the board relative to the side to move (+1 mover,
-1 opponent, 0 empty), with or without symmetry canonicalisation (the net sees game.canonical(board) and plays in
that frame). For a width H the question "is there such a net that never loses?" goes to OR-tools CP-SAT as a
finite constraint problem. SAT gives a net (checked again with net.verify); INFEASIBLE means no net of that width
exists under the stated bias ranges.

Two never-lose conditions (`--cond`):

    reach   the exact one. reach(p) is a boolean per net-to-move position; the roots (empty board for X, the 9
            replies for O, up to symmetry) are reached; reach(p) and "the net plays s at p" imply reach(q) for every
            net-to-move position q one opponent reply later. At every reached position the net's move must be
            non-losing (keeps the mover's minimax value >= 0). This is equivalent to the full game-tree check in
            net.game_tree: a never-losing net only meets positions of value >= 0, and a losing move there hands the
            opponent a forced win, which the game-tree check would find.
    all     sufficient and stronger: the move is non-losing in every position of the chosen sides (627 canonical
            positions for X and O, 338 for X only), reachable or not. Lost positions (value -1) allow any move.

So reach-INFEASIBLE at width H proves no net of width <= H never loses; all-SAT is a valid never-losing net.

Bias ranges (`--bias`):

    small    b1 in [-B1, B1], b2 in [-B2, B2] (defaults 3 and 6; the saved nets use -1..3 and -4..3).
    general  no loss of generality for integer biases, so INFEASIBLE is a proof for all integer biases. For a hidden
             unit with r nonzero weights, W1 x lies in [-r, r]: b1 <= -r makes the unit always 0 (we keep b1 = -r as
             the "dead unit" padding), and b1 > r makes it linear, so lowering b1 to r and adding (b1 - r) W2[k, j] to
             each output bias changes nothing. Hence -r <= b1 <= r and 0 <= h <= 2r <= 18. Then each output's sum
             W2 h differs between two outputs by at most 2S, S = sum_j 2 r_j <= 18 H; shifting b2 so its minimum is 0
             and clamping every gap between consecutive sorted b2 values to 2S + 1 preserves every comparison, so
             b2 in [0, 8 (2S + 1)] (S taken as 18 H).

Search: counterexample-guided (`--cegar`, default). Solve on a subset of positions; if the net found is not valid
on the whole set, add the positions where it fails (for `reach`, the failing position and the chain of net moves
that led there, so the reach implications stay a relaxation) and solve again, hinting the last net. INFEASIBLE on a
subset is INFEASIBLE on the whole set because every constraint used is one of the full model's. `--full` puts all
positions in at once.

Symmetry breaking: hidden units are interchangeable, so units are sorted by (W1 row, b1) as a base-3 key,
non-strictly (two equal rows are allowed; their W2 columns need not merge into a ternary one).

    python3 minsat.py --hidden 8 --cond reach --sides xo --sym      is there an 8-unit net? writes the net if SAT
    python3 minsat.py --hidden 6 --cond reach --sides x --sym --min-nonzero    fewest nonzero weights at H = 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import game as G
import net as N

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- the game side: positions, allowed moves, reach edges ----------------------------------------------------------
def allowed_moves(b):
    """Non-losing moves: keep the mover's value >= min(value, 0). In a lost position every move is allowed."""
    v = min(G.value(b), 0)
    return [m for m in G.legal(b) if -G.value(G.play(b, m)) >= v]


class Problem:
    """Net-to-move positions in the net's frame (canonical if symmetry), their inputs, allowed moves and edges."""

    def __init__(self, sides="xo", symmetry=True):
        self.sides, self.symmetry = sides, symmetry
        want = {G.X} if sides == "x" else ({G.O} if sides == "o" else {G.X, G.O})
        self.want = want
        self.frame = (lambda b: G.canonical(b)[0]) if symmetry else (lambda b: b)
        boards = sorted({self.frame(b) for b in G.decision_positions() if G.to_move(b) in want})
        self.boards = boards
        self.index = {b: i for i, b in enumerate(boards)}
        self.x = [G.encode(b, "trits") for b in boards]
        self.legal = [G.legal(b) for b in boards]
        self.allowed = [allowed_moves(b) for b in boards]
        roots = set()
        if G.X in want:
            roots.add(self.frame(G.START))
        if G.O in want:
            roots |= {self.frame(G.play(G.START, m)) for m in range(9)}
        self.roots = sorted(self.index[b] for b in roots)
        self.edges = {}  # (i, s) -> sorted indices of net-to-move positions one opponent reply after the net plays s
        for i, b in enumerate(boards):
            for s in self.legal[i]:
                c = G.play(b, s)
                out = set()
                if not G.is_terminal(c):
                    for r in G.legal(c):
                        d = G.play(c, r)
                        if not G.is_terminal(d):
                            out.add(self.index[self.frame(d)])
                self.edges[(i, s)] = sorted(out)

    # the net's move at position i (its own frame), from integer weights
    def move(self, W1, b1, W2, b2, i):
        x = self.x[i]
        h = [max(0, sum(w * v for w, v in zip(row, x)) + bb) for row, bb in zip(W1, b1)]
        z = [sum(w * v for w, v in zip(row, h)) + bb for row, bb in zip(W2, b2)]
        best = None
        for m in self.legal[i]:
            if best is None or z[m] > z[best]:
                best = m
        return best

    def failures(self, W1, b1, W2, b2, cond):
        """Positions to add to the CEGAR subset. all: every position where the move is not allowed. reach: for every
        reached position with a disallowed move, that position and the chain of reached positions leading to it."""
        mv = [self.move(W1, b1, W2, b2, i) for i in range(len(self.boards))]
        if cond == "all":
            return [i for i in range(len(self.boards)) if mv[i] not in self.allowed[i]]
        parent = {r: None for r in self.roots}
        queue = list(self.roots)
        bad = []
        while queue:
            i = queue.pop()
            if mv[i] not in self.allowed[i]:
                bad.append(i)
                continue
            for q in self.edges[(i, mv[i])]:
                if q not in parent:
                    parent[q] = i
                    queue.append(q)
        out = set()
        for i in bad:
            while i is not None and i not in out:
                out.add(i)
                i = parent[i]
        return sorted(out)

    def to_net(self, W1, b1, W2, b2):
        return {"format": N.FORMAT, "encoding": "trits", "symmetry": bool(self.symmetry), "act": "relu",
                "layers": [{"W": W1, "b": b1}, {"W": W2, "b": b2}], "scales": [1.0, 1.0]}


# ---- the CP-SAT model ----------------------------------------------------------------------------------------------
def bias_ranges(mode, H, b1=3, b2=6):
    if mode == "small":
        # argmax ignores a common shift of b2, so b2 in [-B2, B2] is the same class as b2 in [0, 2 B2] with min 0
        return (-b1, b1), (0, 2 * b2), 9 + b1
    S = 18 * H
    return (-9, 9), (0, 8 * (2 * S + 1)), 18


def build_model(P, H, subset, cond, bias="small", b1r=3, b2r=6, min_nonzero=False, max_nonzero=None, hint=None,
                fix_w1=None):
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    (l1, u1), (l2, u2), hmax = bias_ranges(bias, H, b1r, b2r)
    w1p = [[m.NewBoolVar(f"w1p{j}_{i}") for i in range(9)] for j in range(H)]
    w1n = [[m.NewBoolVar(f"w1n{j}_{i}") for i in range(9)] for j in range(H)]
    w2p = [[m.NewBoolVar(f"w2p{k}_{j}") for j in range(H)] for k in range(9)]
    w2n = [[m.NewBoolVar(f"w2n{k}_{j}") for j in range(H)] for k in range(9)]
    for j in range(H):
        for i in range(9):
            m.AddBoolOr([w1p[j][i].Not(), w1n[j][i].Not()])
    for k in range(9):
        for j in range(H):
            m.AddBoolOr([w2p[k][j].Not(), w2n[k][j].Not()])
    b1 = [m.NewIntVar(l1, u1, f"b1_{j}") for j in range(H)]
    b2 = [m.NewIntVar(l2, u2, f"b2_{k}") for k in range(9)]
    b2min = m.NewIntVar(l2, u2, "b2min")
    m.AddMinEquality(b2min, b2)
    m.Add(b2min == l2)
    if bias == "general":
        for j in range(H):
            r = sum(w1p[j]) + sum(w1n[j])
            m.Add(b1[j] <= r)
            m.Add(b1[j] >= -r)
    # hidden-unit symmetry breaking: non-increasing base-3 key of (W1 row, b1)
    keys = []
    for j in range(H):
        key = sum((3 ** i) * (w1p[j][i] - w1n[j][i] + 1) for i in range(9)) + (3 ** 9) * (b1[j] - l1)
        keys.append(key)
    for j in range(H - 1):
        m.Add(keys[j] >= keys[j + 1])
    if fix_w1 is not None:
        for j in range(H):
            for i in range(9):
                v = fix_w1[j][i]
                m.Add(w1p[j][i] == (1 if v == 1 else 0))
                m.Add(w1n[j][i] == (1 if v == -1 else 0))

    reach = {}
    amcache = {}
    for i in subset:
        reach[i] = 1 if cond == "all" else m.NewBoolVar(f"r{i}")
    for i in subset:
        x = P.x[i]
        hs = []
        for j in range(H):
            z = sum(x[c] * (w1p[j][c] - w1n[j][c]) for c in range(9) if x[c]) + b1[j]
            h = m.NewIntVar(0, hmax, f"h{i}_{j}")
            a = m.NewBoolVar(f"a{i}_{j}")
            m.Add(h == z).OnlyEnforceIf(a)
            m.Add(z >= 1).OnlyEnforceIf(a)
            m.Add(h == 0).OnlyEnforceIf(a.Not())
            m.Add(z <= 0).OnlyEnforceIf(a.Not())
            hs.append(h)
        Y = {}
        for k in P.legal[i]:
            ts = []
            for j in range(H):
                t = m.NewIntVar(-hmax, hmax, f"t{i}_{k}_{j}")
                m.Add(t == hs[j]).OnlyEnforceIf(w2p[k][j])
                m.Add(t == -hs[j]).OnlyEnforceIf(w2n[k][j])
                m.Add(t == 0).OnlyEnforceIf([w2p[k][j].Not(), w2n[k][j].Not()])
                ts.append(t)
            # lowest index on ties == strict argmax of 9 y_k + (8 - k)
            Yk = m.NewIntVar(9 * (l2 - H * hmax), 9 * (u2 + H * hmax) + 8, f"Y{i}_{k}")
            m.Add(Yk == 9 * (sum(ts) + b2[k]) + (8 - k))
            Y[k] = Yk
        good = [k for k in P.legal[i] if k in P.allowed[i]]
        bad = [k for k in P.legal[i] if k not in P.allowed[i]]
        r = reach[i]
        if not bad:
            if cond == "reach":
                for s in good:
                    for q in P.edges[(i, s)]:
                        if q in reach:
                            c = _argmax_lit(m, Y, s, good, i, amcache)
                            m.AddBoolOr([r.Not(), c.Not(), reach[q]])
            continue
        lo, hi = 9 * (l2 - H * hmax), 9 * (u2 + H * hmax) + 8
        Mg = m.NewIntVar(lo, hi, f"Mg{i}")
        m.AddMaxEquality(Mg, [Y[k] for k in good])
        Mb = m.NewIntVar(lo, hi, f"Mb{i}")
        m.AddMaxEquality(Mb, [Y[k] for k in bad])
        if cond == "all":
            m.Add(Mg >= Mb + 1)
        else:
            m.Add(Mg >= Mb + 1).OnlyEnforceIf(r)
            cache = {}
            for s in good:
                for q in P.edges[(i, s)]:
                    if q in reach:
                        if s not in cache:
                            c = m.NewBoolVar(f"c{i}_{s}")
                            m.Add(Y[s] <= Mg - 1).OnlyEnforceIf(c.Not())  # s is the max of the good moves => c
                            cache[s] = c
                        m.AddBoolOr([r.Not(), cache[s].Not(), reach[q]])
    if cond == "reach":
        for i in P.roots:
            if i in reach:
                m.Add(reach[i] == 1)
    nz = sum(sum(w1p[j]) + sum(w1n[j]) for j in range(H)) + sum(sum(w2p[k]) + sum(w2n[k]) for k in range(9))
    if max_nonzero is not None:
        m.Add(nz <= max_nonzero)
    if min_nonzero:
        m.Minimize(nz)
    if hint is not None:
        W1, hb1, W2, hb2 = hint
        for j in range(H):
            for i in range(9):
                m.AddHint(w1p[j][i], 1 if W1[j][i] == 1 else 0)
                m.AddHint(w1n[j][i], 1 if W1[j][i] == -1 else 0)
            m.AddHint(b1[j], min(max(hb1[j], l1), u1))
        for k in range(9):
            for j in range(H):
                m.AddHint(w2p[k][j], 1 if W2[k][j] == 1 else 0)
                m.AddHint(w2n[k][j], 1 if W2[k][j] == -1 else 0)
            m.AddHint(b2[k], min(max(hb2[k], l2), u2))
    vars_ = (w1p, w1n, b1, w2p, w2n, b2)
    return m, vars_


def _argmax_lit(m, Y, s, group, i, _cache):
    """A literal forced true when Y[s] is the max over `group` (only used where every legal move is allowed)."""
    key = (i, s)
    if key not in _cache:
        c = m.NewBoolVar(f"c{i}_{s}")
        others = [Y[k] for k in group if k != s]
        if others:
            Mo = m.NewIntVar(-10 ** 6, 10 ** 6, f"Mo{i}_{s}")
            m.AddMaxEquality(Mo, others)
            m.Add(Y[s] <= Mo - 1).OnlyEnforceIf(c.Not())
        else:
            m.Add(c == 1)
        _cache[key] = c
    return _cache[key]


def extract(solver, vars_, H):
    w1p, w1n, b1, w2p, w2n, b2 = vars_
    W1 = [[solver.Value(w1p[j][i]) - solver.Value(w1n[j][i]) for i in range(9)] for j in range(H)]
    W2 = [[solver.Value(w2p[k][j]) - solver.Value(w2n[k][j]) for j in range(H)] for k in range(9)]
    return W1, [solver.Value(v) for v in b1], W2, [solver.Value(v) for v in b2]


def initial_subset(P):
    """Roots plus every position one net move and one reply from a root: the opening is where most nets break."""
    s = set(P.roots)
    for i in list(P.roots):
        for sm in P.legal[i]:
            s |= set(P.edges[(i, sm)])
    return s


def solve(P, H, cond="reach", bias="small", b1r=3, b2r=6, cegar=True, timeout=3600.0, workers=8, min_nonzero=False,
          max_nonzero=None, seed=0, log=print, hint=None):
    """Returns a dict: status SAT / UNSAT / UNKNOWN (+ net, nonzero, optimal flag for --min-nonzero), rounds, seconds."""
    from ortools.sat.python import cp_model

    t0 = time.time()
    subset = set(range(len(P.boards))) if not cegar else initial_subset(P)
    rounds = 0
    best = None
    while True:
        rounds += 1
        left = timeout - (time.time() - t0)
        if left <= 1:
            return {"status": "UNKNOWN", "rounds": rounds, "subset": len(subset), "seconds": round(time.time() - t0, 1),
                    "best": best}
        m, vars_ = build_model(P, H, sorted(subset), cond, bias, b1r, b2r, min_nonzero, max_nonzero, hint)
        sv = cp_model.CpSolver()
        sv.parameters.max_time_in_seconds = left
        sv.parameters.num_search_workers = workers
        sv.parameters.random_seed = seed
        st = sv.Solve(m)
        name = sv.StatusName(st)
        log(f"H={H} {cond} {bias} round {rounds}: subset {len(subset)}/{len(P.boards)} -> {name} "
            f"({sv.WallTime():.1f} s, {sv.NumBranches()} branches, {sv.NumConflicts()} conflicts)")
        if st == cp_model.INFEASIBLE:
            return {"status": "UNSAT", "rounds": rounds, "subset": len(subset), "seconds": round(time.time() - t0, 1)}
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {"status": "UNKNOWN", "rounds": rounds, "subset": len(subset), "seconds": round(time.time() - t0, 1),
                    "best": best}
        W1, b1, W2, b2 = extract(sv, vars_, H)
        hint = (W1, b1, W2, b2)
        fail = P.failures(W1, b1, W2, b2, cond)
        new = [i for i in fail if i not in subset]
        if fail and not new:
            raise AssertionError("model and reference disagree on a subset position")
        if not fail:
            net = P.to_net(W1, b1, W2, b2)
            out = {"status": "SAT", "rounds": rounds, "subset": len(subset), "seconds": round(time.time() - t0, 1),
                   "net": net, "nonzero": N.size(net)["nonzero"]}
            if min_nonzero:
                # optimal on the subset and valid on the whole set => optimal on the whole set
                out["optimal"] = st == cp_model.OPTIMAL
                out["bound"] = int(sv.BestObjectiveBound())
            return out
        subset |= set(new)


def check_net(net, sides):
    """Exhaustive re-check with net.verify (pure-Python reference): never loses as the given sides."""
    res = N.verify(net, fast=False)
    ok = (res["as_x"]["loss"] == 0 if "x" in sides else True) and (res["as_o"]["loss"] == 0 if "o" in sides else True)
    return ok, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--cond", default="reach", choices=["reach", "all"])
    ap.add_argument("--sides", default="xo", choices=["x", "o", "xo"])
    ap.add_argument("--sym", action="store_true", help="symmetry canonicalisation")
    ap.add_argument("--bias", default="small", choices=["small", "general"])
    ap.add_argument("--b1", type=int, default=3)
    ap.add_argument("--b2", type=int, default=6)
    ap.add_argument("--full", action="store_true", help="all positions at once instead of CEGAR")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-nonzero", action="store_true")
    ap.add_argument("--max-nonzero", type=int)
    ap.add_argument("--engine", default="sat", choices=["sat", "cpsat"],
                    help="sat: CNF + CaDiCaL (satnet.py, fast); cpsat: the CP-SAT model here (slow, independent check)")
    ap.add_argument("--solver", default="cadical195", help="pysat solver name for --engine sat")
    ap.add_argument("--enc", default="binary", choices=["binary", "unary"], help="arithmetic encoding (satnet.py)")
    ap.add_argument("--require-used", action="store_true",
                    help="every hidden unit has an input and an output weight (sound once width H-1 is UNSAT)")
    ap.add_argument("--out", help="write the net here if SAT")
    ap.add_argument("--json", help="append the result row here")
    a = ap.parse_args()
    P = Problem(a.sides, a.sym)
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.engine == "sat":
        import satnet

        r = satnet.solve(P, a.hidden, a.cond, a.bias, a.b1, a.b2, a.timeout, a.min_nonzero, a.max_nonzero,
                         a.require_used, a.solver, log, a.full, a.enc)
    else:
        r = solve(P, a.hidden, a.cond, a.bias, a.b1, a.b2, not a.full, a.timeout, a.workers, a.min_nonzero,
                  a.max_nonzero, a.seed, log=log)
    row = {k: v for k, v in vars(a).items() if k not in ("out", "json")}
    row.update({k: v for k, v in r.items() if k not in ("net", "best")})
    row["positions"] = len(P.boards)
    if r["status"] == "SAT":
        ok, res = check_net(r["net"], a.sides)
        row["verified"] = ok
        row["as_x"], row["as_o"] = res["as_x"], res["as_o"]
        row["size"] = N.size(r["net"])
        assert ok, "SAT model fails the exhaustive verifier"
        if a.out:
            N.save(r["net"], a.out, verified={k: res[k] for k in ("as_x", "as_o", "perfect", "optimal")},
                   x_only=a.sides == "x", size=N.size(r["net"]),
                   source={k: row[k] for k in ("engine", "enc", "hidden", "cond", "sides", "sym", "bias", "b1", "b2")})
    print(json.dumps(row), flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

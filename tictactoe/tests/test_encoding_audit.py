#!/usr/bin/env python3
"""Encoding audit for the never-lose CNF (cnfproof.py / satnet.py): the CNF file means what the paper says.

Every test loads the exported DIMACS file into a fresh solver, so what is tested is the file a proof is about, and
compares it with net.py (the reference forward pass and game-tree verifier), not with minsat.py's own reference.

(a) test_decode_sat_models: every SAT model the exact search found (tests/sat-models: the minsat.py runs,
    widths 1, 2, 4, 5 and 6) never loses by net.verify, matches the check stored with it, and the exported CNF at
    its width accepts it. Pinning only its hidden layer and letting the solver choose the output layer gives a new
    model; that model is decoded with cnfproof.decode and must also pass net.verify.
(b) test_move_semantics_*: a CNF with the same circuit but no safety constraint, so its argmax literals can be read
    for any net. For random nets (zero-heavy weights and narrow output-bias spreads, so ties are common) the model
    must make exactly one argmax literal true per position, that cell must be the empty cell with the largest score
    and the lowest index among equal scores (a direct simulation written here), it must equal net.choose_move, and
    the CNF must force it (the other cells are UNSAT under the same pins). Occupied cells have no argmax literal.
(c) test_verdict_*: the full proof CNF (general biases, all 627 positions, the exact reach condition). Random nets,
    normalised: SAT iff net.verify says never loses. Random assignments of the CNF's parameter bits (a never-losing
    net's bits with 1-4 bits flipped, and fully random bits): SAT iff the decoded values are a valid trit/bias
    assignment in the CNF's normal form (ranges, unit order, the positions step) and the decoded net never loses.

(d) test_verdict_bounded_*: the bounded-bias CNF (the mode of the width-2 certificate: b1 in [-3, 3], b2 in [0, 12],
    no inactivity clause). Width 4: the two width-4 nets and every one-weight and one-bias change that stays in the
    box, pinned: SAT iff net.verify says never loses. Width 2: random nets in the box are all rejected, and none
    never loses. Results: results/encoding-audit-bounded.json.

Needs python-sat (pip install -r requirements-sat.txt). About 15 minutes on one core.

    python3 tests/test_encoding_audit.py      PASS/FAIL per test, exit 1 on any FAIL
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cnfproof as C  # noqa: E402
import game as G  # noqa: E402
import minsat as M  # noqa: E402
import net as N  # noqa: E402
import satnet  # noqa: E402

try:
    import pysat  # noqa: F401
except ImportError:  # under pytest without python-sat: skip the module (the solver tests need it)
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("needs python-sat", allow_module_level=True)

SAT_MODELS = os.path.join(HERE, "sat-models")
WEIGHTS = os.path.join(ROOT, "weights")
TMP = tempfile.mkdtemp(dir=None)
STATS = {}


def solver_from(path):
    from pysat.formula import CNF
    from pysat.solvers import Solver

    return Solver(name="cadical195", bootstrap_with=CNF(from_file=path).clauses)


_CACHE = {}


def proof_cnf(H, sides="xo", sym=True):
    """The proof CNF exactly as cnfproof.py build writes it (general biases, strong, no require-used)."""
    key = ("proof", H, sides, sym)
    if key not in _CACHE:
        out = os.path.join(TMP, f"p{H}{sides}{int(sym)}")
        mp, _ = C.build(H, "general", False, True, "binary", sides=sides, sym=sym, out=out)
        _CACHE[key] = (mp, solver_from(out + ".cnf"), M.Problem(sides, sym))
    return _CACHE[key]


def bounded_cnf(H):
    """The bounded-bias proof CNF as cnfproof.py build writes it for the H=2 certificate (bias small: b1 in [-3, 3],
    b2 in [0, 12] after the shift to minimum 0, strong, no require-used, no inactivity clause)."""
    key = ("bounded", H)
    if key not in _CACHE:
        out = os.path.join(TMP, f"b{H}")
        mp, _ = C.build(H, "small", False, True, "binary", out=out)
        _CACHE[key] = (mp, solver_from(out + ".cnf"), M.Problem("xo", True))
    return _CACHE[key]


def bounded_normal(p, mp):
    """normalise without the positions step (the bounded CNF has no inactivity clause); None if outside the box."""
    q = C.normalise(*p)
    W1, b1, W2, b2 = q
    if not all(mp["l1"] <= b <= mp["u1"] for b in b1) or max(b2) > mp["u2"]:
        return None
    return q


class Sub:  # the positions a CNF was built over, for normalise's positions step
    def __init__(self, xs):
        self.x = xs


def moves_cnf(H, sym, positions=None):
    """Same circuit as the proof CNF (satnet.SatNet, general biases, strong), but every legal move is allowed, so the
    CNF is satisfiable for every net in normal form and its argmax literals show the move it encodes."""
    key = ("moves", H, sym)
    if key not in _CACHE:
        P = M.Problem("xo", sym)
        P.allowed = [list(L) for L in P.legal]
        pos = sorted(range(len(P.boards))) if positions is None else sorted(positions)
        out = os.path.join(TMP, f"m{H}{int(sym)}")
        sink = C.Sink(out + ".body")
        S = satnet.SatNet(P, H, "reach", "general", 3, 6, sink=sink, strong=True)
        for i in pos:
            S.add_position(i)
        S.finish_positions()
        sink.close()
        with open(out + ".cnf", "w") as f:
            f.write(f"p cnf {S.C.top} {sink.n}\n")
            with open(out + ".body") as b:
                shutil.copyfileobj(b, f, 1 << 20)
        os.remove(out + ".body")
        mp = {"enc": "binary", "l1": S.l1, "w1": S.w1, "w2": S.w2, "B1": S.B1, "B2": S.B2}
        _CACHE[key] = (mp, solver_from(out + ".cnf"), P, pos, {i: S.argmax[i] for i in pos})
    return _CACHE[key]


def parts(net):
    L = net["layers"]
    return L[0]["W"], L[0]["b"], L[1]["W"], L[1]["b"]


def scores(W1, b1, W2, b2, x):
    h = [max(0, sum(w * v for w, v in zip(row, x)) + bb) for row, bb in zip(W1, b1)]
    return [sum(w * v for w, v in zip(row, h)) + bb for row, bb in zip(W2, b2)]


def direct_move(W1, b1, W2, b2, board):
    """The model's move rule, written out: largest score among empty cells, lowest index on equal scores."""
    z = scores(W1, b1, W2, b2, G.encode(board, "trits"))
    empty = [k for k in range(9) if board[k] == G.EMPTY]
    top = max(z[k] for k in empty)
    return min(k for k in empty if z[k] == top), z, empty, top


def to_net(p, sym):
    return {"format": N.FORMAT, "encoding": "trits", "symmetry": sym, "act": "relu",
            "layers": [{"W": p[0], "b": p[1]}, {"W": p[2], "b": p[3]}], "scales": [1.0, 1.0]}


def random_net(rng, H, b1r, b2r, zero=0.5):
    w = lambda: 0 if rng.random() < zero else rng.choice((-1, 1))  # noqa: E731
    return ([[w() for _ in range(9)] for _ in range(H)], [rng.randint(-b1r, b1r) for _ in range(H)],
            [[w() for _ in range(H)] for _ in range(9)], [rng.randint(-b2r, b2r) for _ in range(9)])


def never_loses(p, sides="xo", sym=True):
    v = N.verify(to_net(p, sym))
    return v["perfect"] if sides == "xo" else v["as_x"]["loss"] == 0


def model_true(s):
    return {v for v in s.get_model() if v > 0}


# ---- (a) every SAT model found ------------------------------------------------------------------------------------
def test_decode_sat_models():
    files = sorted(f for f in os.listdir(SAT_MODELS) if f.endswith(".json"))
    files += ["../../weights/smallest-never-loses-h4.json", "../../weights/smallest-never-loses-h5.json"]
    assert len(files) >= 8
    n = 0
    for f in files:
        net = N.load(os.path.join(SAT_MODELS, f))
        src = net.get("source") if isinstance(net.get("source"), dict) else {}
        sides = src.get("sides", "x" if net.get("x_only") else "xo")
        sym = net["symmetry"]
        v = N.verify(net)
        assert v["as_x"] == net["verified"]["as_x"], f
        if sides == "xo":
            assert v["perfect"] and v["as_o"] == net["verified"]["as_o"], f
        else:
            assert v["as_x"]["loss"] == 0 and v["as_o"]["loss"] > 0, f
        if not sym and sides == "xo":
            continue
        p = parts(net)
        H = len(p[0])
        mp, s, P = proof_cnf(H, sides, sym)
        q = C.normalise(*p, P=P)
        assert [P.move(*q, i) for i in range(len(P.boards))] == [P.move(*p, i) for i in range(len(P.boards))], f
        assert s.solve(assumptions=C.pin_units(mp, *q)), f + ": CNF rejects a never-losing net"
        assert list(C.decode(mp, model_true(s))) == list(q), f + ": decode(model) differs from the pinned net"
        # pin only the hidden layer: the solver picks the output layer; decode that model and verify it
        pins = C.pin_units(mp, *q)
        n_hidden = 2 * 9 * H + sum(len(b) for b in mp["B1"])
        hid = [lit for lit in pins if abs(lit) in {abs(x) for x in hidden_vars(mp)}]
        assert len(hid) == n_hidden
        assert s.solve(assumptions=hid), f
        d = C.decode(mp, model_true(s))
        assert never_loses(d, sides, sym), f + ": a decoded solver model loses"
        n += 1
    STATS["sat_models_decoded"] = n


def hidden_vars(mp):
    return [v for row in mp["w1"] for pn in row for v in pn] + [b for B in mp["B1"] for b in B]


# ---- (b) move semantics, ties and the legal mask ------------------------------------------------------------------
def _move_semantics(H, sym, nets, sample_forced=40, positions=None):
    mp, s, P, pos, am = moves_cnf(H, sym, positions)
    sub = Sub([P.x[i] for i in pos])
    rng = random.Random(5)
    ties = masked = checked = forced = 0
    for p in nets:
        q = C.normalise(*p, P=sub)
        assert s.solve(assumptions=C.pin_units(mp, *q)), "moves CNF rejects a normalised net"
        true = model_true(s)
        net = to_net(q, sym)
        for i in pos:
            board = P.boards[i]
            mv, z, empty, top = direct_move(*q, board)
            assert sorted(am[i]) == empty, "argmax literals are not exactly the empty cells"
            on = sorted(k for k, lit in am[i].items() if (lit in true if lit > 0 else -lit not in true))
            assert on == [mv], (i, on, mv)
            cb, perm = G.canonical(board) if sym else (board, list(range(9)))
            assert cb == board
            assert N.choose_move(net, board) == perm[mv]
            ties += sum(1 for k in empty if z[k] == top) > 1
            masked += max(z) > top
            checked += 1
        for i in rng.sample(pos, min(sample_forced, len(pos))):
            mv = direct_move(*q, P.boards[i])[0]
            for k, lit in am[i].items():
                if k != mv:
                    assert not s.solve(assumptions=C.pin_units(mp, *q) + [lit]), "CNF allows a second move"
                    forced += 1
    return ties, masked, checked, forced


def test_move_semantics_sym():
    rng = random.Random(3)
    nets = [random_net(rng, 2, rng.choice((1, 3, 9)), rng.choice((0, 1, 2, 30)), rng.choice((0.4, 0.7)))
            for _ in range(24)]
    ties, masked, checked, forced = _move_semantics(2, True, nets)
    STATS["moves_sym"] = {"positions_checked": checked, "tied_positions": ties, "mask_decides": masked,
                          "forced_checks": forced}
    assert ties > 1000 and masked > 1000, STATS["moves_sym"]


def test_move_semantics_no_symmetry():
    """No canonicalisation: the CNF's move equals net.choose_move on the raw board (300 random raw positions)."""
    rng = random.Random(4)
    P = M.Problem("xo", False)
    pos = rng.sample(range(len(P.boards)), 300)
    nets = [random_net(rng, 3, 4, rng.choice((0, 1, 5)), 0.6) for _ in range(12)]
    ties, masked, checked, forced = _move_semantics(3, False, nets, 20, pos)
    STATS["moves_raw"] = {"positions_checked": checked, "tied_positions": ties, "mask_decides": masked,
                          "forced_checks": forced}
    assert ties > 100 and masked > 100, STATS["moves_raw"]


# ---- (c) verdicts on the full proof CNF ---------------------------------------------------------------------------
def normal_form(mp, p, P):
    """The CNF's side conditions for decoded values (general biases, strong, no require-used)."""
    W1, b1, W2, b2 = p
    H = len(W1)
    for j in range(H):
        r = sum(1 for v in W1[j] if v)
        if not -r <= b1[j] <= r:
            return False
        if not any(sum(w * v for w, v in zip(W1[j], x)) + b1[j] <= 0 for x in P.x):
            return False
    if min(b2) != 0 or max(b2) > mp["u2"]:
        return False
    keys = [[b1[j]] + [W1[j][i] for i in range(8, -1, -1)] + [W2[k][j] for k in range(9)] for j in range(H)]
    return all(keys[j] >= keys[j + 1] for j in range(H - 1))


def param_vars(mp):
    return ([pn for row in mp["w1"] + mp["w2"] for pn in row],
            [b for B in mp["B1"] + mp["B2"] for b in B])


def assignment_units(mp, trits, bits):
    """Units for raw values: trits as (p, n) booleans (both true allowed, to test rejection), bias bits."""
    pairs, bvars = param_vars(mp)
    out = []
    for (p, n), (vp, vn) in zip(pairs, trits):
        out += [p if vp else -p, n if vn else -n]
    out += [b if v else -b for b, v in zip(bvars, bits)]
    return out


def decode_raw(mp, trits, bits):
    pairs, bvars = param_vars(mp)
    if any(vp and vn for vp, vn in trits):
        return None
    true = {p for (p, _), (vp, _) in zip(pairs, trits) if vp} | {n for (_, n), (_, vn) in zip(pairs, trits) if vn}
    true |= {b for b, v in zip(bvars, bits) if v}
    return C.decode(mp, true)


def raw_of(mp, q):
    pins = set(C.pin_units(mp, *q))
    pairs, bvars = param_vars(mp)
    return [(p in pins, n in pins) for p, n in pairs], [b in pins for b in bvars]


def test_verdict_random_nets_h2():
    mp, s, P = proof_cnf(2)
    rng = random.Random(9)
    n = 0
    for t in range(150):
        p = random_net(rng, 2, rng.choice((2, 9, 40)), rng.choice((0, 2, 40, 3000)), rng.choice((0.3, 0.6)))
        q = C.normalise(*p, P=P)
        assert s.solve(assumptions=C.pin_units(mp, *q)) == never_loses(q), t
        n += 1
    STATS["verdict_random_h2"] = n


def test_verdict_random_assignments_h4():
    """Around the 85-parameter net (width 4): flip 1-4 random parameter bits; plus fully random bits."""
    mp, s, P = proof_cnf(4)
    base = C.normalise(*parts(N.load(os.path.join(WEIGHTS, "smallest-never-loses-h4.json"))), P=P)
    sat_base = C.normalise(*parts(N.load(os.path.join(SAT_MODELS, "xo-sym-H4.json"))), P=P)
    rng = random.Random(13)
    counts = {"sat": 0, "unsat": 0, "invalid_trit": 0, "not_normal": 0, "loses": 0}
    for t in range(300):
        src = base if t % 2 else sat_base
        trits, bits = raw_of(mp, src)
        flat = [("t", i, 0) for i in range(len(trits))] + [("t", i, 1) for i in range(len(trits))] + \
               [("b", i, 0) for i in range(len(bits))]
        if t % 10 == 9:  # fully random
            trits = [(rng.random() < 0.4, rng.random() < 0.4) for _ in trits]
            bits = [rng.random() < 0.5 for _ in bits]
        else:
            trits = [list(x) for x in trits]
            for kind, i, k in rng.sample(flat, rng.randint(1, 4)):
                if kind == "t":
                    trits[i][k] = not trits[i][k]
                else:
                    bits[i] = not bits[i]
            trits = [tuple(x) for x in trits]
        d = decode_raw(mp, trits, bits)
        if d is None:
            want = False
            counts["invalid_trit"] += 1
        elif not normal_form(mp, d, P):
            want = False
            counts["not_normal"] += 1
        else:
            want = never_loses(d)
            counts["loses"] += not want
        got = s.solve(assumptions=assignment_units(mp, trits, bits))
        assert got == want, (t, got, want)
        counts["sat" if got else "unsat"] += 1
    STATS["verdict_assignments_h4"] = counts
    assert counts["sat"] >= 5 and counts["loses"] >= 20 and counts["invalid_trit"] and counts["not_normal"], counts


def test_verdict_perturbed_nets_h4():
    """One-weight and one-bias changes of the width-4 nets, normalised: SAT iff net.verify says never loses."""
    mp, s, P = proof_cnf(4)
    pos = neg = 0
    for f in (os.path.join(WEIGHTS, "smallest-never-loses-h4.json"), os.path.join(SAT_MODELS, "xo-sym-H4.json")):
        W1, b1, W2, b2 = parts(N.load(f))
        variants = []
        for j in range(4):
            for i in range(9):
                for v in (-1, 0, 1):
                    if v != W1[j][i]:
                        A = [r[:] for r in W1]
                        A[j][i] = v
                        variants.append((A, b1, W2, b2))
            for d in (-1, 1):
                b = b1[:]
                b[j] += d
                variants.append((W1, b, W2, b2))
        for k in range(9):
            for d in (-1, 1):
                b = b2[:]
                b[k] += d
                variants.append((W1, b1, W2, b))
        for p in variants:
            q = C.normalise(*p, P=P)
            want = never_loses(q)
            assert s.solve(assumptions=C.pin_units(mp, *q)) == want
            pos += want
            neg += not want
    STATS["verdict_perturbed_h4"] = {"never_loses": pos, "loses": neg}
    assert pos > 10 and neg > 10, STATS["verdict_perturbed_h4"]


def test_verdict_bounded_h4():
    """The bounded-mode CNF (the mode of the H=2 certificate), at width 4: the two width-4 nets and every one-weight
    and one-bias change of them that stays in the bounded box, pinned: SAT iff net.verify says never loses."""
    mp, s, P = bounded_cnf(4)
    pos = neg = out = 0
    for f in (os.path.join(WEIGHTS, "smallest-never-loses-h4.json"), os.path.join(SAT_MODELS, "xo-sym-H4.json")):
        W1, b1, W2, b2 = parts(N.load(f))
        variants = [(W1, b1, W2, b2)]
        for j in range(4):
            for i in range(9):
                for v in (-1, 0, 1):
                    if v != W1[j][i]:
                        A = [r[:] for r in W1]
                        A[j][i] = v
                        variants.append((A, b1, W2, b2))
            for d in (-1, 1):
                b = b1[:]
                b[j] += d
                variants.append((W1, b, W2, b2))
        for k in range(9):
            for j in range(4):
                for v in (-1, 0, 1):
                    if v != W2[k][j]:
                        A = [r[:] for r in W2]
                        A[k][j] = v
                        variants.append((W1, b1, A, b2))
            for d in (-1, 1):
                b = b2[:]
                b[k] += d
                variants.append((W1, b1, W2, b))
        for t, p in enumerate(variants):
            q = bounded_normal(p, mp)
            if q is None:
                out += 1
                continue
            want = never_loses(q)
            if t == 0:
                assert want, f
            assert s.solve(assumptions=C.pin_units(mp, *q)) == want, (f, t)
            pos += want
            neg += not want
    STATS["verdict_bounded_h4"] = {"never_loses": pos, "loses": neg, "outside_box": out}
    assert pos > 10 and neg > 10, STATS["verdict_bounded_h4"]


def test_verdict_bounded_h2():
    """The bounded-mode CNF at width 2, the certified case: random nets in the box, pinned, are all rejected, and
    net.verify agrees that none never loses."""
    mp, s, P = bounded_cnf(2)
    rng = random.Random(21)
    n = 0
    while n < 100:
        p = random_net(rng, 2, 3, 6, rng.choice((0.3, 0.6)))
        q = bounded_normal(p, mp)
        if q is None:
            continue
        want = never_loses(q)
        assert not want
        assert s.solve(assumptions=C.pin_units(mp, *q)) == want
        n += 1
    STATS["verdict_bounded_h2"] = n


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    try:
        import pysat  # noqa: F401
    except ImportError:
        print("SKIP all (no python-sat)")
        sys.exit(0)
    only = set(sys.argv[1:])
    bad = 0
    for t in [t for t in TESTS if not only or t.__name__ in only]:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {t.__name__} ({time.time() - t0:.1f} s): {type(e).__name__}: {e!r}", flush=True)
    print("STATS " + json.dumps(STATS))
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if bad else 0)

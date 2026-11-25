"""Integer-only reference forward pass and exhaustive verification for ternary tic-tac-toe nets.

Weight file (JSON, format "ttt-ternary-mlp-v1"):

    encoding   "planes" (18 inputs in {0,1}) or "trits" (9 inputs in {-1,0,+1}); see game.encode
    symmetry   true: the net sees game.canonical(board) and its move is mapped back through the symmetry
    act        hidden activation on integers: "relu" (max(0, z)) or "step" (1 if z >= 0 else 0)
    layers     [{"W": out x in matrix with entries in {-1,0,+1}, "b": out integers}, ...]
    scales     informational only; every saved net has scales 1 (the model class has no scales)

Forward pass, all integers: h = act(W h + b) for each hidden layer, z = W h + b for the last layer, move =
argmax of z over the empty cells, ties to the lowest cell index (in the net's own frame when symmetry is on).
The model class has no scales: the integer net is the whole model. (A real scale on a hidden layer would not in
general fold into an integer bias, so none is assumed.) The legality mask is part of the output rule, not learned.

Nothing here needs numpy except the optional batched path (`moves_batch`), which the tests check against the
pure-Python reference on every position.
"""
from __future__ import annotations

import json

import game as G

FORMAT = "ttt-ternary-mlp-v1"


# ---- load / save ------------------------------------------------------------------------------------------------
def load(path):
    with open(path) as f:
        net = json.load(f)
    check(net)
    return net


def save(net, path, **extra):
    check(net)
    out = dict(net)
    out.update(extra)
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")


def check(net):
    assert net.get("format") == FORMAT, net.get("format")
    assert net["encoding"] in G.ENCODINGS
    assert net["act"] in ("relu", "step")
    n_in = G.ENCODINGS[net["encoding"]]
    for L in net["layers"]:
        W, b = L["W"], L["b"]
        assert len(W) == len(b) and all(len(r) == n_in for r in W), "layer shape"
        assert all(w in (-1, 0, 1) for r in W for w in r), "weights must be ternary"
        assert all(isinstance(v, int) for v in b), "biases must be integers"
        n_in = len(W)
    assert n_in == 9, "last layer must have 9 outputs"


def size(net):
    """Parameter counts. 'nonzero' = ternary weights that are not 0."""
    w = sum(len(r) for L in net["layers"] for r in L["W"])
    nz = sum(1 for L in net["layers"] for r in L["W"] for v in r if v)
    nb = sum(len(L["b"]) for L in net["layers"])
    nzb = sum(1 for L in net["layers"] for v in L["b"] if v)
    return {"weights": w, "nonzero": nz, "biases": nb, "nonzero_biases": nzb, "params": w + nb,
            "hidden": [len(L["b"]) for L in net["layers"][:-1]]}


# ---- integer reference ------------------------------------------------------------------------------------------
def forward_int(net, x):
    """Integer logits for one input vector (list of ints). Pure Python ints throughout."""
    h = list(x)
    last = len(net["layers"]) - 1
    for li, L in enumerate(net["layers"]):
        z = [sum(w * v for w, v in zip(row, h) if w) + b for row, b in zip(L["W"], L["b"])]
        if li < last:
            z = [max(0, v) for v in z] if net["act"] == "relu" else [1 if v >= 0 else 0 for v in z]
        h = z
    return h


def _argmax_legal(z, board):
    best = None
    for m in range(9):
        if board[m] == G.EMPTY and (best is None or z[m] > z[best]):
            best = m
    return best


def choose_move(net, board):
    """The net's move on a non-terminal board (cell index 0..8 of that board)."""
    if net.get("symmetry"):
        cb, p = G.canonical(board)
        m = _argmax_legal(forward_int(net, G.encode(cb, net["encoding"])), cb)
        return p[m]
    return _argmax_legal(forward_int(net, G.encode(board, net["encoding"])), board)


_FRAMES = {}


def _frames(boards, encoding, symmetry):
    """Cached (encoded inputs, empty mask, symmetry perms or None) for a tuple of boards."""
    import numpy as np

    key = (boards, encoding, bool(symmetry))
    if key not in _FRAMES:
        if symmetry:
            can = [G.canonical(b) for b in boards]
            frames, perms = [c[0] for c in can], np.array([c[1] for c in can])
        else:
            frames, perms = list(boards), None
        x = np.array([G.encode(b, encoding) for b in frames], dtype=np.int64)
        empty = np.array([[c == G.EMPTY for c in b] for b in frames])
        _FRAMES[key] = (x, empty, perms)
    return _FRAMES[key]


def moves_batch(net, boards):
    """Same as choose_move for many boards at once (numpy int64; exact)."""
    import numpy as np

    h, empty, perms = _frames(tuple(boards), net["encoding"], net.get("symmetry"))
    last = len(net["layers"]) - 1
    for li, L in enumerate(net["layers"]):
        z = h @ np.array(L["W"], dtype=np.int64).T + np.array(L["b"], dtype=np.int64)
        if li < last:
            z = np.maximum(z, 0) if net["act"] == "relu" else (z >= 0).astype(np.int64)
        h = z
    lo = np.iinfo(np.int64).min
    m = np.argmax(np.where(empty, h, lo), axis=1)  # first max = lowest index
    if perms is not None:
        m = perms[np.arange(len(m)), m]
    return [int(v) for v in m]


def _decision():
    global _DEC
    try:
        return _DEC
    except NameError:
        _DEC = tuple(G.decision_positions())
        return _DEC


# ---- exhaustive verification ------------------------------------------------------------------------------------
def game_tree(policy, net_side):
    """Play the net (policy: board -> move) as net_side against every opponent reply, from the empty board.

    Returns counts of distinct game lines by outcome for the net, plus the set of positions where the net moved."""
    out = {"win": 0, "draw": 0, "loss": 0}
    visited = set()

    def rec(b):
        w = G.winner(b)
        if w:
            out["win" if w == net_side else "loss"] += 1
            return
        if all(b):
            out["draw"] += 1
            return
        if G.to_move(b) == net_side:
            visited.add(b)
            rec(G.play(b, policy(b)))
        else:
            for m in G.legal(b):
                rec(G.play(b, m))

    rec(G.START)
    return out, visited


def verify(net, boards=None, fast=True):
    """Exhaustive check. Returns a dict; 'perfect' is True iff the net never loses as X or as O (unchanged from the
    first picks). 'finisher' = perfect and takes an immediate win in every one of the 4,520 positions that has
    one ('takes_wins'). 'converter' = finisher and keeps the win in every position whose value is a win for the
    side to move ('converts'), so it wins from all of them against every reply.

    optimal_share: over all 4,520 positions with a move to make (any line of play), the share where the net's move
    is optimal (keeps the minimax value). tree_optimal_share: the same over the positions the net actually meets
    when it plays every opponent line. fast=False uses the pure-Python reference instead of numpy."""
    boards = tuple(boards or _decision())
    if fast:
        mv = dict(zip(boards, moves_batch(net, boards)))
    else:
        mv = {b: choose_move(net, b) for b in boards}
    return verify_moves(mv)


def verify_moves(mv):
    """verify() for any player given as a move table {board: move} over the decision positions (see verify)."""
    boards = tuple(mv)
    ok = {b: (G.value(G.play(b, m)) == -G.value(b)) for b, m in mv.items()}
    res = {"positions": len(boards), "optimal": sum(ok.values())}
    res["optimal_share"] = res["optimal"] / len(boards)
    seen = set()
    for side, name in ((G.X, "as_x"), (G.O, "as_o")):
        lines, visited = game_tree(mv.__getitem__, side)
        res[name] = lines
        seen |= visited
    res["tree_positions"] = len(seen)
    res["tree_optimal"] = sum(ok[b] for b in seen)
    res["tree_optimal_share"] = res["tree_optimal"] / len(seen)
    res["perfect"] = res["as_x"]["loss"] == 0 and res["as_o"]["loss"] == 0
    # (b) takes an immediate win whenever one exists, over all positions (not only its own tree)
    win_now = [b for b in boards if G.winning_moves(b)]
    res["win_now_positions"] = len(win_now)
    res["win_now_taken"] = sum(1 for b in win_now if mv[b] in G.winning_moves(b))
    res["takes_wins"] = res["win_now_taken"] == len(win_now)
    # (c) converts every won position: the move keeps the win in every position whose value is a win for the side
    # to move. Equivalent to winning against every reply from each of them (the game is finite, and any move out
    # of a won position that keeps value +1 leaves the opponent lost, so every reply is again a won position);
    # converts_by_play checks that directly.
    won = [b for b in boards if G.value(b) == 1]
    res["won_positions"] = len(won)
    res["won_kept"] = sum(ok[b] for b in won)
    res["converts"] = res["won_kept"] == len(won)
    res["finisher"] = res["perfect"] and res["takes_wins"]
    res["converter"] = res["finisher"] and res["converts"]
    return res


def converts_by_play(policy, boards=None):
    """Brute force for (c): from every position whose value is a win for the side to move, play the policy for
    that side against every opponent reply. Returns the won positions where some line does not end in a win."""
    memo = {}

    def wins(b, side):  # side to move at b is `side`; True iff the policy wins on every line from here
        if b in memo:
            return memo[b]
        if G.winner(b) or all(b):  # the opponent's reply ended the game: a loss or a draw, not a win
            return False
        c = G.play(b, policy(b))
        if G.winner(c):
            r = True
        elif all(c):
            r = False
        else:
            r = all(wins(G.play(c, m), side) for m in G.legal(c))
        memo[b] = r
        return r

    boards = boards or _decision()
    return [b for b in boards if G.value(b) == 1 and not wins(b, G.to_move(b))]


def summary(res):
    x, o = res["as_x"], res["as_o"]
    return (f"{'PERFECT' if res['perfect'] else 'loses'}, takes wins {res['win_now_taken']}/{res['win_now_positions']}, "
            f"keeps wins {res['won_kept']}/{res['won_positions']}: optimal {res['optimal']}/{res['positions']} "
            f"({100 * res['optimal_share']:.2f} %), in tree {res['tree_optimal']}/{res['tree_positions']}; "
            f"as X lines W/D/L {x['win']}/{x['draw']}/{x['loss']}, as O {o['win']}/{o['draw']}/{o['loss']}")

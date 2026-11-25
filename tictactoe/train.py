"""Ternary QAT for tic-tac-toe MLPs (torch, CPU), plus discrete pruning that keeps the net perfect.

The model is trained directly in integer units: forward uses W_t = clamp(round(W), -1, 1) and b_t = round(b) with
straight-through gradients, hidden h = relu(z) (or a step with a sigmoid surrogate), so the trained forward pass
*is* the integer reference in net.py (the real per-layer scales are all 1; a softmax temperature stands in for the
last one). Training data: every one of the 4,520 positions with a move to make (or the 627 canonical ones when
symmetry is on), target = the set of optimal moves.

    python3 train.py --hidden 16 --seed 0             one run, prints the exhaustive check
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time

import numpy as np
import torch

import game as G
import net as N


# ---- data -------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, encoding, symmetry, target="optimal", sides="xo"):
        """target "optimal": moves that keep the minimax value. "safe": moves that do not lose (in a won position a
        drawing move is also accepted), the weakest target that still implies never losing. "finisher": optimal,
        except that where an immediate win exists only the winning cells count.
        sides "x": train only on positions with X to move (the X-only goal); "xo": all positions."""
        self.encoding, self.symmetry, self.target, self.sides = encoding, symmetry, target, sides
        movers = {"x": (G.X,), "o": (G.O,), "xo": (G.X, G.O)}[sides]
        self.all = tuple(b for b in G.decision_positions() if G.to_move(b) in movers)
        train = sorted({G.canonical(b)[0] for b in self.all}) if symmetry else list(self.all)
        self.boards = train
        self.x = torch.tensor([G.encode(b, encoding) for b in train], dtype=torch.float32)
        self.legal = torch.tensor([[c == G.EMPTY for c in b] for b in train])
        opt = torch.zeros(len(train), 9, dtype=torch.bool)
        for i, b in enumerate(train):
            if target == "optimal" or (target == "finisher" and not G.winning_moves(b)):
                opt[i, G.optimal_moves(b)] = True
            elif target == "finisher":
                opt[i, G.winning_moves(b)] = True
            else:
                v = min(G.value(b), 0)
                opt[i, [m for m in G.legal(b) if -G.value(G.play(b, m)) >= v]] = True
        self.opt = opt
        self.index = {b: i for i, b in enumerate(train)}
        self.weight = torch.ones(len(train))
        self.bad = self.legal & ~opt
        self.has_bad = self.bad.any(1)


# ---- model ------------------------------------------------------------------------------------------------------
def ste_tern(w):
    return w + (torch.clamp(torch.round(w), -1, 1) - w).detach()


def ste_round(b):
    return b + (torch.round(b) - b).detach()


class TernMLP(torch.nn.Module):
    def __init__(self, n_in, hidden, act="relu", init=1.0):
        super().__init__()
        dims = [n_in] + list(hidden) + [9]
        self.act = act
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for a, c in zip(dims[:-1], dims[1:]):
            self.W.append(torch.nn.Parameter(torch.empty(c, a).uniform_(-init, init)))
            self.b.append(torch.nn.Parameter(torch.zeros(c)))

    def forward(self, x, q=1.0):
        """q = 1: the exact ternary/integer net (straight-through grads); q < 1 blends in the float shadow."""
        h = x
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Wq = ste_tern(W) if q == 1.0 else (1 - q) * W + q * ste_tern(W)
            bq = ste_round(b) if q == 1.0 else (1 - q) * b + q * ste_round(b)
            z = h @ Wq.T + bq
            if i < last:
                if self.act == "relu":
                    z = torch.relu(z)
                else:
                    s = torch.sigmoid(2.0 * (z + 0.5))
                    z = s + ((z >= 0).float() - s).detach()
            h = z
        return h

    def clamp_(self, lim=1.5):
        with torch.no_grad():
            for W in self.W:
                W.clamp_(-lim, lim)

    def export(self, encoding, symmetry):
        layers = []
        for W, b in zip(self.W, self.b):
            Wq = torch.clamp(torch.round(W.detach()), -1, 1).to(torch.int64).tolist()
            bq = torch.round(b.detach()).to(torch.int64).tolist()
            layers.append({"W": Wq, "b": [int(v) for v in bq]})
        return {"format": N.FORMAT, "encoding": encoding, "symmetry": bool(symmetry), "act": self.act,
                "layers": layers, "scales": [1.0] * len(layers)}


def set_loss(z, d, tau, margin):
    neg = torch.finfo(z.dtype).min / 4
    zl = torch.where(d.legal, z / tau, torch.full_like(z, neg))
    zo = torch.where(d.opt, z / tau, torch.full_like(z, neg))
    ce = torch.logsumexp(zl, 1) - torch.logsumexp(zo, 1)
    w = d.weight / d.weight.sum()
    best_opt = torch.where(d.opt, z, torch.full_like(z, neg)).max(1).values
    best_bad = torch.where(d.bad, z, torch.full_like(z, neg)).max(1).values
    hinge = torch.where(d.has_bad, torch.relu(margin - (best_opt - best_bad)), torch.zeros_like(best_opt))
    return (w * ce).sum(), (w * hinge).sum()


def n_correct(z, d):
    """Training-set positions whose integer argmax (lowest index on ties) is optimal."""
    neg = torch.finfo(z.dtype).min
    m = torch.where(d.legal, z, torch.full_like(z, neg)).argmax(1)
    return int(d.opt[torch.arange(len(m)), m].sum())


# ---- discrete pruning -------------------------------------------------------------------------------------------
def criterion(kind):
    if kind == "xonly":  # never loses as X from the empty board (O side not checked)
        return lambda net: N.verify(net)["as_x"]["loss"] == 0
    if kind in ("never-loses", "perfect"):  # "perfect" is the old name of the same criterion
        return lambda net: N.verify(net)["perfect"]
    if kind == "optimal":
        return lambda net: N.verify(net)["optimal_share"] == 1.0
    if kind in ("finisher", "converter"):
        return lambda net: N.verify(net)[kind]
    raise ValueError(kind)


def hard_masks(d, goal):
    """(applies, allowed) over the training boards for the hard part of a goal's repair objective.

    finisher: where an immediate win exists, only the winning cells are allowed.
    converter: also, in every position whose value is a win for the mover, only value-keeping moves are allowed
    (and still only the winning cells where one exists). perfect: nothing (the tree term carries it)."""
    n = len(d.boards)
    applies = np.zeros(n, dtype=bool)
    allowed = np.zeros((n, 9), dtype=bool)
    if goal in ("finisher", "converter"):
        for i, b in enumerate(d.boards):
            w = G.winning_moves(b)
            if w:
                applies[i] = True
                allowed[i, w] = True
            elif goal == "converter" and G.value(b) == 1:
                applies[i] = True
                allowed[i, G.optimal_moves(b)] = True
    return applies, allowed


def drop_unit(net, layer, unit):
    """Remove hidden unit `unit` of hidden layer `layer` (its row, its bias, its column in the next layer)."""
    out = copy.deepcopy(net)
    L, nxt = out["layers"][layer], out["layers"][layer + 1]
    del L["W"][unit], L["b"][unit]
    for r in nxt["W"]:
        del r[unit]
    return out


def prune(net, kind="perfect", seed=0, rounds=2):
    """Greedy: delete hidden units, then zero weights and biases, keeping `kind` true. Returns the smaller net."""
    ok = criterion(kind)
    net = copy.deepcopy(net)
    assert ok(net)
    rng = random.Random(seed)
    for _ in range(rounds):
        changed = False
        for li in range(len(net["layers"]) - 1):
            u = len(net["layers"][li]["b"]) - 1
            while u >= 0:
                if len(net["layers"][li]["b"]) > 1:
                    cand = drop_unit(net, li, u)
                    if ok(cand):
                        net, changed = cand, True
                u -= 1
        sites = [(li, r, c) for li, L in enumerate(net["layers"]) for r, row in enumerate(L["W"])
                 for c, v in enumerate(row) if v]
        rng.shuffle(sites)
        for li, r, c in sites:
            old = net["layers"][li]["W"][r][c]
            net["layers"][li]["W"][r][c] = 0
            if ok(net):
                changed = True
            else:
                net["layers"][li]["W"][r][c] = old
        for li, L in enumerate(net["layers"]):
            for r, v in enumerate(L["b"]):
                if v:
                    L["b"][r] = 0
                    if ok(net):
                        changed = True
                    else:
                        L["b"][r] = v
        if not changed:
            break
    assert ok(net)
    return net


# ---- discrete repair --------------------------------------------------------------------------------------------
class Scorer:
    """Counts training-set positions whose integer argmax is not a target move (numpy, exact)."""

    def __init__(self, d, act):
        self.x = d.x.numpy().astype(np.int64)
        self.legal = d.legal.numpy()
        self.opt = d.opt.numpy()
        self.act = act
        self.rows = np.arange(len(self.x))
        safe = np.zeros_like(self.opt)
        for i, b in enumerate(d.boards):
            v = min(G.value(b), 0)
            safe[i, [m for m in G.legal(b) if -G.value(G.play(b, m)) >= v]] = True
        self.safe = safe
        self.tree = None  # optional per-position weight for unsafe moves (repair's tree objective)
        self.hard = None  # optional (applies, allowed, weight): a goal's must-hold moves (see hard_masks)

    def wrong(self, Ws, bs):
        h = self.x
        last = len(Ws) - 1
        for i, (W, b) in enumerate(zip(Ws, bs)):
            z = h @ W.T + b
            if i < last:
                z = np.maximum(z, 0) if self.act == "relu" else (z >= 0).astype(np.int64)
            h = z
        m = np.argmax(np.where(self.legal, h, np.iinfo(np.int64).min), axis=1)
        wrong = int(len(m) - self.opt[self.rows, m].sum())
        if self.tree is not None:
            wrong += int(self.tree[~self.safe[self.rows, m]].sum())
        if self.hard is not None:
            applies, allowed, w = self.hard
            wrong += w * int((applies & ~allowed[self.rows, m]).sum())
        return wrong


def repair(net, d, passes=40, seed=0, plateau=0.2, tree_weight=0, goal="perfect", hard_weight=20):
    """Hill-climb on single ternary weights (to either other value) and biases (+-1, +-2): accept a change that
    lowers the wrong-move count (sideways moves with probability `plateau`). Returns (first net meeting `goal`
    or None, lowest wrong count, net at that count).

    goal "finisher"/"converter" adds hard_weight per training position that breaks the goal's must-hold moves
    (hard_masks); "perfect" adds nothing. The game-tree part of every goal is the tree term below.

    tree_weight > 0 adds the tree objective: after every pass, each position in the current net's own game tree
    (as X and as O) costs tree_weight more when the net's move there loses value (drops a win to a loss or a draw
    to a loss). The count it returns then includes that term."""
    rng = random.Random(seed)
    Ws = [np.array(L["W"], dtype=np.int64) for L in net["layers"]]
    bs = [np.array(L["b"], dtype=np.int64) for L in net["layers"]]
    sc = Scorer(d, net["act"])
    ok = criterion(goal)
    if goal != "perfect":
        sc.hard = (*hard_masks(d, goal), hard_weight)

    def as_net():
        out = dict(net)
        out["layers"] = [{"W": W.tolist(), "b": [int(v) for v in b]} for W, b in zip(Ws, bs)]
        return out

    def set_tree():
        if tree_weight:
            sc.tree = np.zeros(len(sc.x), dtype=np.int64)
            sc.tree[tree_boards(as_net(), d)] = tree_weight

    set_tree()
    cur = sc.wrong(Ws, bs)
    best = (cur, as_net())
    perfect = as_net() if ok(best[1]) else None
    sites = [("W", li, r, c) for li, W in enumerate(Ws) for r in range(W.shape[0]) for c in range(W.shape[1])]
    sites += [("b", li, r, None) for li, b in enumerate(bs) for r in range(len(b))]
    for _ in range(passes):
        rng.shuffle(sites)
        moved = False
        for kind, li, r, c in sites:
            if kind == "W":
                old = Ws[li][r, c]
                alts = [v for v in (-1, 0, 1) if v != old]
            else:
                old = bs[li][r]
                alts = [old + k for k in (-1, 1, -2, 2)]
            for v in rng.sample(alts, len(alts)):
                if kind == "W":
                    Ws[li][r, c] = v
                else:
                    bs[li][r] = v
                s = sc.wrong(Ws, bs)
                if s < cur or (s == cur and rng.random() < plateau):
                    moved |= s < cur
                    cur = s
                    break
                if kind == "W":
                    Ws[li][r, c] = old
                else:
                    bs[li][r] = old
        if cur < best[0]:
            best = (cur, as_net())
        if perfect is None and ok(as_net()):
            perfect = as_net()
            if tree_weight:
                break
        if cur == 0:
            break
        if tree_weight:
            set_tree()
            cur = sc.wrong(Ws, bs)
    return perfect, best[0], best[1]


def shrink(net, d, goal, seed=0, passes=60, log=None):
    """Drop-a-unit-and-repair: for each hidden unit, delete it, repair the rest back to `goal` (hard + tree
    objective), prune, and keep the result if it has fewer nonzero weights. Repeats until no unit can go.
    One hidden layer only."""
    ok = criterion(goal)
    assert ok(net) and len(net["layers"]) == 2
    best = prune(net, goal, seed)
    rng = random.Random(seed)
    improved = True
    while improved:
        improved = False
        units = list(range(len(best["layers"][0]["b"])))
        if len(units) <= 1:
            break
        rng.shuffle(units)
        for u in units:
            cand = drop_unit(best, 0, u)
            if not ok(cand):
                cand = repair(cand, d, passes=passes, seed=rng.randrange(1 << 30), tree_weight=100, goal=goal)[0]
            if cand is None:
                continue
            cand = prune(cand, goal, seed)
            if N.size(cand)["nonzero"] < N.size(best)["nonzero"]:
                best, improved = cand, True
                if log:
                    log(f"shrink {goal}: hidden {N.size(best)['hidden']} nonzero {N.size(best)['nonzero']}")
                break
    return best


# ---- one training run -------------------------------------------------------------------------------------------
def tree_boards(net, d):
    """Training-set boards the net meets while playing every opponent line as X and as O."""
    boards = tuple(G.decision_positions())
    mv = dict(zip(boards, N.moves_batch(net, boards)))
    seen = set()
    for side in ((G.X,) if getattr(d, "sides", "xo") == "x" else (G.X, G.O)):
        seen |= N.game_tree(mv.__getitem__, side)[1]
    if d.symmetry:
        seen = {G.canonical(b)[0] for b in seen}
    return [d.index[b] for b in seen]


def train(hidden, seed=0, encoding="planes", symmetry=False, act="relu", steps=4000, lr=0.01, tau=2.0,
          margin=1.0, w_hinge=0.3, l1=0.0, ramp=0.6, target="optimal", tree_weight=0.0, eval_every=25,
          data=None, log=None):
    """tree_weight > 0: every eval, positions in the current net's game tree get weight 1 + tree_weight (DAgger
    style), so capacity goes to the lines the net actually has to survive."""
    torch.manual_seed(seed)
    d = data or Data(encoding, symmetry, target)
    d.weight = torch.ones(len(d.boards))
    model = TernMLP(G.ENCODINGS[encoding], hidden, act)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = (-1, 0, None)  # (correct, -nonzero, net)
    best_perfect = None  # (nonzero, step, net)
    first_perfect = None
    t0 = time.time()
    for step in range(1, steps + 1):
        # cosine decay to 10 % keeps late training from flipping weights back and forth
        for g in opt.param_groups:
            g["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        q = min(1.0, step / (ramp * steps)) if ramp else 1.0
        z = model(d.x, q)
        ce, hinge = set_loss(z, d, tau, margin)
        loss = ce + w_hinge * hinge
        if l1:
            loss = loss + l1 * sum(W.abs().sum() for W in model.W) / sum(W.numel() for W in model.W)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.clamp_()
        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                c = n_correct(model(d.x), d)
            net = model.export(encoding, symmetry)
            nz = N.size(net)["nonzero"]
            if tree_weight:
                d.weight = torch.ones(len(d.boards))
                d.weight[tree_boards(net, d)] += tree_weight
            if (c, -nz) > best_acc[:2]:
                best_acc = (c, -nz, net)
            if c >= len(d.boards) * 0.5:
                if N.verify(net)["perfect"]:
                    if first_perfect is None:
                        first_perfect = step
                    if best_perfect is None or nz < best_perfect[0]:
                        best_perfect = (nz, step, net)
            if log and step % (eval_every * 20) == 0:
                log(f"step {step} loss {loss.item():.4f} correct {c}/{len(d.boards)} nz {nz}")
    net = best_perfect[2] if best_perfect else best_acc[2]
    return {"net": net, "perfect": best_perfect is not None, "first_perfect_step": first_perfect, "train_correct": best_acc[0],
            "train_positions": len(d.boards), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", default="16", help="comma-separated hidden widths, '' for none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoding", default="planes", choices=sorted(G.ENCODINGS))
    ap.add_argument("--symmetry", action="store_true")
    ap.add_argument("--act", default="relu", choices=["relu", "step"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--target", default="optimal", choices=["optimal", "safe", "finisher"])
    ap.add_argument("--tree-weight", type=float, default=0.0)
    ap.add_argument("--l1", type=float, default=0.0)
    ap.add_argument("--prune", choices=["never-loses", "perfect", "optimal", "finisher", "converter", "xonly"])
    ap.add_argument("--out", help="write the weights JSON here")
    a = ap.parse_args()
    torch.set_num_threads(1)
    hidden = [int(v) for v in a.hidden.split(",") if v]
    r = train(hidden, a.seed, a.encoding, a.symmetry, a.act, a.steps, a.lr, l1=a.l1, target=a.target,
              tree_weight=a.tree_weight, log=print)
    net = r["net"]
    res = N.verify(net)
    print(json.dumps({k: v for k, v in r.items() if k != "net"}), N.size(net))
    print(N.summary(res))
    if a.prune and criterion(a.prune)(net):
        net = prune(net, a.prune, a.seed)
        print("pruned", N.size(net), N.summary(N.verify(net)))
    if a.out:
        N.save(net, a.out)


if __name__ == "__main__":
    main()

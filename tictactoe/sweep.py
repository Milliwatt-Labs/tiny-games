"""Size sweep: ternary QAT -> discrete repair -> pruning, every net checked exhaustively.

Each run (encoding, symmetry, activation, hidden widths, seed) does:
  1. QAT (train.train), exhaustive check of the snapshot it returns            -> "qat_*" fields
  2. discrete repair of that net (train.repair), exhaustive check             -> "rep_*" fields
  3. if a perfect net exists: greedy pruning that keeps it perfect             -> "prp_*" fields
     if an all-optimal net exists: greedy pruning that keeps it all-optimal    -> "pro_*" fields
The finish grid (--grid finish, results/sweep-finish.jsonl; run_finish) targets the stricter goals instead:
QAT on the "finisher" target, repair to a finisher (never loses + takes every immediate win), prune keeping that
("prf_*"); repair the unpruned finisher on to a converter (also keeps every won position won), prune keeping that
("prc_*").
--grid shrink then runs train.shrink (drop a hidden unit, repair, prune; repeat) on the 6 smallest finishers and
the 6 smallest converters of the finish grid -> results/shrink-finish.jsonl, nets as shrink-<key>-<goal>.json.
One JSON line per run goes to results/sweep.jsonl (resumable: finished runs are skipped). Nets that are perfect or
all-optimal are written to --nets (scratch) so report.py can pick the smallest.

    nice -n 10 python3 sweep.py --jobs 6 --nets nets
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))


def grid():
    runs = []
    sym_hidden = [(16,), (12,), (10,), (8,), (7,), (6,), (5,), (4,), (8, 8), (8, 6), (6, 6), (6, 4), (4, 4)]
    raw_hidden = [(128,), (96,), (64,), (48,), (32,), (24,), (16,), (12,), (32, 32), (32, 16), (16, 16)]
    for enc in ("planes", "trits"):
        for h in sym_hidden:
            runs += [dict(encoding=enc, symmetry=True, act="relu", hidden=h, seed=s) for s in range(6)]
        for h in raw_hidden:
            runs += [dict(encoding=enc, symmetry=False, act="relu", hidden=h, seed=s) for s in range(4)]
        for sym in (False, True):
            runs += [dict(encoding=enc, symmetry=sym, act="relu", hidden=(), seed=s) for s in range(2)]
    for h in [(16,), (12,), (8,), (8, 8)]:
        runs += [dict(encoding="planes", symmetry=True, act="step", hidden=h, seed=s) for s in range(3)]
    return runs


def grid_push():
    """Second stage: many more seeds at the widths where perfect nets start to appear, with the tree objective."""
    return [dict(encoding=enc, symmetry=True, act="relu", hidden=(w,), seed=s, tree_repair=100)
            for enc in ("planes", "trits") for w in (10, 9, 8, 7, 6, 5, 4) for s in range(6, 18)]


def grid_tritsopt():
    """Review 2 (2026-11-23): trits input with symmetry at widths above 16, aiming at an all-optimal net, so the
    optimal-level ratio with and without symmetry can be taken at one input encoding."""
    return [dict(encoding="trits", symmetry=True, act="relu", hidden=(w,), seed=s)
            for w in (48, 32, 24, 20) for s in range(6)] + \
        [dict(encoding="trits", symmetry=True, act="relu", hidden=(16,), seed=s) for s in range(6, 18)]


def grid_finish():
    """Third stage (2026-11-21 follow-up): finisher / converter goals, symmetry on."""
    return [dict(encoding=enc, symmetry=True, act="relu", hidden=(w,), seed=s, goal="finish")
            for enc in ("planes", "trits") for w in (12, 14, 16, 20, 24, 32) for s in range(8)]


def grid_xonly():
    """2026-11-21 follow-up (A): never loses as X only, trits and planes, with and without symmetry."""
    runs = []
    for enc in ("trits", "planes"):
        runs += [dict(encoding=enc, symmetry=False, act="relu", hidden=(w,), seed=s, goal="xonly")
                 for w in (1, 2, 3, 4, 6, 8, 12, 16) for s in range(4)]
        runs += [dict(encoding=enc, symmetry=True, act="relu", hidden=(w,), seed=s, goal="xonly")
                 for w in (1, 2, 4) for s in range(2)]
    return runs


def run_xonly(cfg, nets_dir):
    """QAT on the X-to-move positions, repair with the X game-tree term to "never loses as X", prune, shrink."""
    import net as N
    import train as T

    t0 = time.time()
    d = T.Data(cfg["encoding"], cfg["symmetry"], "optimal", sides="x")
    out = dict(cfg, hidden=list(cfg["hidden"]), key=key(cfg))
    r = T.train(list(cfg["hidden"]), cfg["seed"], cfg["encoding"], cfg["symmetry"], cfg["act"], data=d,
                steps=2000)
    ok = T.criterion("xonly")
    net = r["net"]
    out["qat_xonly"] = ok(net)
    if not out["qat_xonly"]:
        f, wrong, _ = T.repair(net, d, passes=150, seed=cfg["seed"], tree_weight=100, goal="xonly")
        out["rep_wrong_train"] = wrong
        net = f
    out["rep_xonly"] = net is not None
    if net is not None:
        p = T.prune(net, "xonly", cfg["seed"])
        out["prx_nonzero"], out["prx_params"], out["prx_hidden"] = (N.size(p)["nonzero"], N.size(p)["params"],
                                                                    N.size(p)["hidden"])
        s = T.shrink(p, d, "xonly", seed=cfg["seed"], passes=60)
        v = N.verify(s)
        sz = N.size(s)
        out.update({"shx_nonzero": sz["nonzero"], "shx_params": sz["params"], "shx_hidden": sz["hidden"],
                    "shx_as_x": v["as_x"], "shx_as_o_losses": v["as_o"]["loss"]})
        os.makedirs(nets_dir, exist_ok=True)
        N.save(s, os.path.join(nets_dir, out["key"] + "-xonly.json"))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def key(r):
    if r.get("goal") == "xonly":
        return "xonly-" + key({k: v for k, v in r.items() if k != "goal"})
    if r.get("goal") == "shrink":
        return f"shrink-{r['target']}-{r['src']}"
    if r.get("goal") == "finish":
        return "finish-" + key({k: v for k, v in r.items() if k != "goal"})
    if r.get("tree_repair"):
        return "push-" + key({k: v for k, v in r.items() if k != "tree_repair"})
    return f"{r['encoding']}-{'sym' if r['symmetry'] else 'raw'}-{r['act']}-{'x'.join(map(str, r['hidden'])) or '0'}-s{r['seed']}"


def _rec(out, prefix, net):
    import net as N

    v = N.verify(net)
    sz = N.size(net)
    out.update({f"{prefix}_perfect": v["perfect"], f"{prefix}_finisher": v["finisher"],
                f"{prefix}_converter": v["converter"], f"{prefix}_wins_taken": v["win_now_taken"],
                f"{prefix}_wins_kept": v["won_kept"], f"{prefix}_optimal_share": round(v["optimal_share"], 6),
                f"{prefix}_losses": v["as_x"]["loss"] + v["as_o"]["loss"], f"{prefix}_nonzero": sz["nonzero"],
                f"{prefix}_params": sz["params"], f"{prefix}_hidden": sz["hidden"]})
    return v


def run_finish(cfg, nets_dir):
    import net as N
    import train as T

    t0 = time.time()
    d = T.Data(cfg["encoding"], cfg["symmetry"], "finisher")
    out = dict(cfg, hidden=list(cfg["hidden"]), key=key(cfg))
    r = T.train(list(cfg["hidden"]), cfg["seed"], cfg["encoding"], cfg["symmetry"], cfg["act"], data=d)
    _rec(out, "qat", r["net"])
    os.makedirs(nets_dir, exist_ok=True)
    f, wrong, best = T.repair(r["net"], d, passes=150, seed=cfg["seed"], tree_weight=100, goal="finisher")
    _rec(out, "rep", f or best)
    out["rep_wrong_train"] = wrong
    if f is not None:
        pf = T.prune(f, "finisher", cfg["seed"])
        _rec(out, "prf", pf)
        N.save(pf, os.path.join(nets_dir, out["key"] + "-finisher.json"))
        c = f if N.verify(f)["converter"] else T.repair(f, d, passes=150, seed=cfg["seed"], tree_weight=100,
                                                        goal="converter")[0]
        if c is not None:
            pc = T.prune(c, "converter", cfg["seed"])
            _rec(out, "prc", pc)
            N.save(pc, os.path.join(nets_dir, out["key"] + "-converter.json"))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def grid_shrink(nets_dir, top=6):
    path = os.path.join(HERE, "results", "sweep-finish.jsonl")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out = []
    for goal, tag in (("finisher", "prf"), ("converter", "prc")):
        c = sorted((r for r in rows if f"{tag}_nonzero" in r), key=lambda r: (r[f"{tag}_nonzero"], r[f"{tag}_params"]))
        out += [dict(goal="shrink", target=goal, src=r["key"], seed=r["seed"], hidden=r[f"{tag}_hidden"])
                for r in c[:top]]
    return out


def run_shrink(cfg, nets_dir):
    import net as N
    import train as T

    t0 = time.time()
    net = N.load(os.path.join(nets_dir, f"{cfg['src']}-{cfg['target']}.json"))
    net = {k: net[k] for k in ("format", "encoding", "symmetry", "act", "layers", "scales")}
    d = T.Data(net["encoding"], net["symmetry"], "finisher")
    out = dict(cfg, key=key(cfg))
    _rec(out, "in", net)
    s = T.shrink(net, d, cfg["target"], seed=cfg["seed"])
    _rec(out, "out", s)
    N.save(s, os.path.join(nets_dir, f"{out['key']}-{cfg['target']}.json"))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run(args):
    cfg, nets_dir = args
    if cfg.get("goal") == "shrink":
        import torch

        torch.set_num_threads(1)
        return run_shrink(cfg, nets_dir)
    if cfg.get("goal") == "xonly":
        import torch

        torch.set_num_threads(1)
        return run_xonly(cfg, nets_dir)
    if cfg.get("goal") == "finish":
        import torch

        torch.set_num_threads(1)
        return run_finish(cfg, nets_dir)
    import torch

    torch.set_num_threads(1)
    import net as N
    import train as T

    t0 = time.time()
    d = T.Data(cfg["encoding"], cfg["symmetry"])
    out = dict(cfg, hidden=list(cfg["hidden"]), key=key(cfg))
    r = T.train(list(cfg["hidden"]), cfg["seed"], cfg["encoding"], cfg["symmetry"], cfg["act"], data=d)

    def rec(prefix, net):
        v = N.verify(net)
        sz = N.size(net)
        out.update({f"{prefix}_perfect": v["perfect"], f"{prefix}_optimal_share": round(v["optimal_share"], 6),
                    f"{prefix}_tree_optimal_share": round(v["tree_optimal_share"], 6),
                    f"{prefix}_losses": v["as_x"]["loss"] + v["as_o"]["loss"],
                    f"{prefix}_nonzero": sz["nonzero"], f"{prefix}_params": sz["params"],
                    f"{prefix}_hidden": sz["hidden"], f"{prefix}_nonzero_biases": sz["nonzero_biases"]})
        return v

    vq = rec("qat", r["net"])
    perfect = r["net"] if vq["perfect"] else None
    allopt = r["net"] if vq["optimal_share"] == 1 else None
    tw = cfg.get("tree_repair", 0)
    p, wrong, best = T.repair(r["net"], d, passes=150 if tw else 30, seed=cfg["seed"], tree_weight=tw)
    vb = rec("rep", best)
    out["rep_wrong_train"] = wrong
    if vb["perfect"]:
        perfect = best  # the repaired net with the fewest wrong moves, if it is perfect
    elif perfect is None and p is not None:
        perfect = p
        rec("rep", p)
    if allopt is None and vb["optimal_share"] == 1:
        allopt = best
    os.makedirs(nets_dir, exist_ok=True)
    if perfect is not None:
        pp = T.prune(perfect, "perfect", cfg["seed"])
        rec("prp", pp)
        N.save(pp, os.path.join(nets_dir, out["key"] + "-perfect.json"))
    if allopt is not None:
        po = T.prune(allopt, "optimal", cfg["seed"])
        rec("pro", po)
        N.save(po, os.path.join(nets_dir, out["key"] + "-optimal.json"))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "sweep.jsonl"))
    ap.add_argument("--nets", required=True, help="directory for the perfect nets (scratch)")
    ap.add_argument("--status", help="status.json path")
    ap.add_argument("--grid", default="main", choices=["main", "push", "finish", "shrink", "xonly", "tritsopt"])
    a = ap.parse_args()
    if a.grid == "xonly" and a.out == ap.get_default("out"):
        a.out = os.path.join(HERE, "results", "sweep-xonly.jsonl")
    if a.grid == "tritsopt" and a.out == ap.get_default("out"):
        a.out = os.path.join(HERE, "results", "sweep-tritsopt.jsonl")
    if a.grid in ("finish", "shrink") and a.out == ap.get_default("out"):
        a.out = os.path.join(HERE, "results", f"{'sweep' if a.grid == 'finish' else 'shrink'}-finish.jsonl")
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    todo = [c for c in {"main": grid, "push": grid_push, "finish": grid_finish,
                                    "shrink": lambda: grid_shrink(a.nets), "xonly": grid_xonly,
                                    "tritsopt": grid_tritsopt}[a.grid]() if key(c) not in done]
    # big nets first so the long tail is short
    todo.sort(key=lambda c: -sum(c["hidden"]))
    print(f"{len(todo)} runs to do, {len(done)} done", flush=True)
    n = 0
    with Pool(a.jobs) as pool, open(a.out, "a") as f:
        for r in pool.imap_unordered(run, [(c, a.nets) for c in todo]):
            f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            if a.grid == "xonly":
                print(f"[{n}/{len(todo)}] {r['key']} qat {r['qat_xonly']} rep {r['rep_xonly']} "
                      f"prune {r.get('prx_nonzero')} shrink {r.get('shx_nonzero')} hidden {r.get('shx_hidden')} "
                      f"{r['seconds']}s", flush=True)
            elif a.grid == "shrink":
                print(f"[{n}/{len(todo)}] {r['key']} {r['target']} {r['in_nonzero']} -> {r['out_nonzero']} "
                      f"hidden {r['out_hidden']} {r['seconds']}s", flush=True)
            elif a.grid == "finish":
                print(f"[{n}/{len(todo)}] {r['key']} rep finisher {r['rep_finisher']} prf {r.get('prf_nonzero')} "
                      f"prc {r.get('prc_nonzero')} {r['seconds']}s", flush=True)
            else:
                print(f"[{n}/{len(todo)}] {r['key']} qat {r['qat_perfect']} {r['qat_optimal_share']:.4f} "
                      f"rep {r['rep_perfect']} {r['rep_optimal_share']:.4f} prp {r.get('prp_nonzero')} "
                      f"pro {r.get('pro_nonzero')} {r['seconds']}s", flush=True)
            if a.status:
                with open(a.status, "w") as s:
                    json.dump({"state": "running", "done": n, "total": len(todo), "updatedAt": time.time()}, s)
    if a.status:
        with open(a.status, "w") as s:
            json.dump({"state": "done", "done": n, "total": len(todo), "updatedAt": time.time()}, s)


if __name__ == "__main__":
    main()

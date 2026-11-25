"""Pick the smallest nets from a sweep, squeeze them further, save them to weights/, print the size/accuracy tables.

    python3 report.py --nets nets [--squeeze 24]  > results/tables.md
    python3 report.py --finish --nets nets --squeeze 12 > results/tables-finish.md

--squeeze N re-prunes each pick (and the 5 smallest candidates in its class) from N random orders and keeps the
smallest result; every saved net is re-verified exhaustively and carries its check in "verified".
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import game as G
import net as N
import train as T

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fam(r):
    f = f"{r['encoding']} / {'sym' if r['symmetry'] else 'raw'} / {r['act']}"
    return f + (" / tree repair" if r.get("tree_repair") else "")


def hid(h):
    return "x".join(map(str, h)) or "0"


def curve(rows):
    out = ["| inputs / symmetry / act | hidden | runs | QAT perfect | QAT best optimal % | repair perfect | "
           "repair best optimal % | smallest perfect (nonzero / params) | smallest all-optimal (nonzero / params) |",
           "|---|---|---|---|---|---|---|---|---|"]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(fam(r), tuple(r["hidden"]))].append(r)

    def order(k):
        f, h = k
        return (f, -len(h) if h else 0, -sum(h))

    for (f, h) in sorted(groups, key=order):
        g = groups[(f, h)]
        pp = [r for r in g if "prp_nonzero" in r]
        po = [r for r in g if "pro_nonzero" in r]
        sp = f"{min(r['prp_nonzero'] for r in pp)} / {min(r['prp_params'] for r in pp)}" if pp else "—"
        so = f"{min(r['pro_nonzero'] for r in po)} / {min(r['pro_params'] for r in po)}" if po else "—"
        out.append(f"| {f} | {hid(h)} | {len(g)} | {sum(r['qat_perfect'] for r in g)} | "
                   f"{100 * max(r['qat_optimal_share'] for r in g):.2f} | {len(pp)} | "
                   f"{100 * max(r['rep_optimal_share'] for r in g):.2f} | {sp} | {so} |")
    return "\n".join(out)


def squeeze(net, kind, tries):
    best = net
    for s in range(tries):
        cand = T.prune(json.loads(json.dumps(net)), kind, seed=1000 + s, rounds=4)
        if (N.size(cand)["nonzero"], N.size(cand)["params"]) < (N.size(best)["nonzero"], N.size(best)["params"]):
            best = cand
    return best


def r_small(c):
    return min(r.get("prp_nonzero", r.get("pro_nonzero", 0)) for r in c) <= 400


VERIFIED = ("perfect", "optimal", "positions", "optimal_share", "tree_positions", "tree_optimal", "tree_optimal_share",
            "as_x", "as_o", "takes_wins", "win_now_positions", "win_now_taken", "converts", "won_positions", "won_kept",
            "finisher", "converter")


def save_pick(net, name, source, kind):
    v = N.verify(net, fast=False)
    assert T.criterion(kind)(net) and v["perfect"]
    if kind == "converter":
        assert not N.converts_by_play(lambda b: N.choose_move(net, b))
    keep = {k: v[k] for k in VERIFIED}
    net = {k: net[k] for k in ("format", "encoding", "symmetry", "act", "layers", "scales")}
    N.save(net, os.path.join(HERE, "weights", name), size=N.size(net), verified=keep, source=source,
           criterion=kind)
    return keep


def finish(a):
    """Smallest finisher and converter from results/sweep-finish.jsonl."""
    rows = load_rows(os.path.join(HERE, "results", "sweep-finish.jsonl"))
    print(f"# Finish sweep ({len(rows)} runs)\n")
    print("QAT on the finisher target, repair, greedy prune (sweep.py --grid finish):\n")
    print("| inputs | hidden | runs | repaired to finisher | smallest finisher (nonzero / params) | "
          "converter | smallest converter (nonzero / params) |")
    print("|---|---|---|---|---|---|---|")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["encoding"], tuple(r["hidden"]))].append(r)
    for (enc, h) in sorted(groups, key=lambda k: (k[0], sum(k[1]))):
        g = groups[(enc, h)]
        f = [r for r in g if "prf_nonzero" in r]
        c = [r for r in g if "prc_nonzero" in r]
        sf = f"{min(r['prf_nonzero'] for r in f)} / {min(r['prf_params'] for r in f)}" if f else "—"
        scv = f"{min(r['prc_nonzero'] for r in c)} / {min(r['prc_params'] for r in c)}" if c else "—"
        print(f"| {enc} / sym | {hid(h)} | {len(g)} | {len(f)} | {sf} | {len(c)} | {scv} |")
    print()
    # candidate nets: (nonzero, params, file, source run, criterion it was saved under)
    pool = []
    for r in rows:
        for kind, tag in (("finisher", "prf"), ("converter", "prc")):
            if f"{tag}_nonzero" in r:
                pool.append((r[f"{tag}_nonzero"], r[f"{tag}_params"], f"{r['key']}-{kind}.json", r["key"], kind))
    srows = load_rows(os.path.join(HERE, "results", "shrink-finish.jsonl")) if os.path.exists(
        os.path.join(HERE, "results", "shrink-finish.jsonl")) else []
    for r in srows:
        pool.append((r["out_nonzero"], r["out_params"], f"{r['key']}-{r['target']}.json", r["key"], r["target"]))
    if srows:
        print("Drop-a-unit-and-repair (train.shrink) on the smallest of each kind:\n")
        print("| goal | from run | before (hidden / nonzero) | after (hidden / nonzero / params) | seconds |")
        print("|---|---|---|---|---|")
        for r in sorted(srows, key=lambda r: (r["target"], r["out_nonzero"])):
            print(f"| {r['target']} | {r['src']} | {hid(r['in_hidden'])} / {r['in_nonzero']} | "
                  f"{hid(r['out_hidden'])} / {r['out_nonzero']} / {r['out_params']} | {r['seconds']:.0f} |")
        print()
    print("| pick | from run | hidden | nonzero weights | weights | biases (nonzero) | total params | "
          "immediate wins taken | won positions kept | optimal % (all 4,520) | lines as X W/D/L | lines as O W/D/L |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, kind in (("smallest-finisher.json", "finisher"), ("smallest-converter.json", "converter")):
        # a converter is also a finisher, so it competes for the finisher pick too
        c = sorted(p for p in pool if kind == "finisher" or p[4] == "converter")
        if not c:
            print(f"| {name} | none | | | | | | | | | | |")
            continue
        best, src = None, None
        for _, _, fname, key, _ in c[:4]:
            net = N.load(os.path.join(a.nets, fname))
            if a.squeeze:
                net = squeeze(net, kind, a.squeeze)
            k = (N.size(net)["nonzero"], N.size(net)["params"])
            if best is None or k < best[0]:
                best, src = (k, net), key
        net = best[1]
        v = save_pick(net, name, src, kind)
        s_ = N.size(net)
        print(f"| {name} | {src} | {hid(s_['hidden'])} | {s_['nonzero']} | {s_['weights']} | {s_['biases']} "
              f"({s_['nonzero_biases']}) | {s_['params']} | {v['win_now_taken']}/{v['win_now_positions']} | "
              f"{v['won_kept']}/{v['won_positions']} | {100 * v['optimal_share']:.2f} | "
              f"{v['as_x']['win']}/{v['as_x']['draw']}/{v['as_x']['loss']} | "
              f"{v['as_o']['win']}/{v['as_o']['draw']}/{v['as_o']['loss']} |", flush=True)
    print()
    print("Earlier picks under the new checks:\n")
    print("| saved net | never loses | immediate wins taken | won positions kept |")
    print("|---|---|---|---|")
    for name in ("smallest-never-loses.json", "smallest-never-loses-params.json", "smallest-optimal.json",
                 "smallest-never-loses-nosym.json", "smallest-optimal-nosym.json", "smallest-never-loses-step.json"):
        v = N.verify(N.load(os.path.join(HERE, "weights", name)))
        print(f"| {name} | {'yes' if v['perfect'] else 'no'} | {v['win_now_taken']}/{v['win_now_positions']} | "
              f"{v['won_kept']}/{v['won_positions']} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default=os.path.join(HERE, "results", "sweep.jsonl"))
    ap.add_argument("--nets", required=True)
    ap.add_argument("--squeeze", type=int, default=0)
    ap.add_argument("--finish", action="store_true", help="pick the finisher / converter nets instead")
    a = ap.parse_args()
    if a.finish:
        return finish(a)
    rows = load_rows(a.rows)
    print(f"# Sweep tables ({len(rows)} runs)\n")
    print(curve(rows))
    print()

    def cands(kind, pred, metric):
        tag = "prp" if kind == "perfect" else "pro"
        c = [r for r in rows if f"{tag}_nonzero" in r and pred(r)]
        return sorted(c, key=lambda r: (r[f"{tag}_{metric}"], r[f"{tag}_params" if metric == "nonzero"
                                                                     else f"{tag}_nonzero"]))

    picks = [
        ("smallest-never-loses.json", "perfect", lambda r: True, "nonzero"),
        ("smallest-never-loses-params.json", "perfect", lambda r: True, "params"),
        ("smallest-optimal.json", "optimal", lambda r: True, "nonzero"),
        ("smallest-never-loses-nosym.json", "perfect", lambda r: not r["symmetry"], "nonzero"),
        ("smallest-optimal-nosym.json", "optimal", lambda r: not r["symmetry"], "nonzero"),
        ("smallest-never-loses-step.json", "perfect", lambda r: r["act"] == "step", "nonzero"),
    ]
    os.makedirs(os.path.join(HERE, "weights"), exist_ok=True)
    print("| pick | from run | hidden | nonzero weights | weights | biases (nonzero) | total params | "
          "optimal % (all 4,520) | optimal % (in its game tree) | lines as X W/D/L | lines as O W/D/L |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, kind, pred, metric in picks:
        c = cands(kind, pred, metric)
        if not c:
            print(f"| {name} | none | | | | | | | | | |")
            continue
        best, src = None, None
        for r in c[: (5 if a.squeeze and r_small(c) else 1)]:
            net = N.load(os.path.join(a.nets, f"{r['key']}-{kind}.json"))
            if a.squeeze and N.size(net)["nonzero"] <= 400:  # big no-symmetry nets: one prune is enough
                net = squeeze(net, kind, a.squeeze)
            k = (N.size(net)[metric], N.size(net)["params" if metric == "nonzero" else "nonzero"])
            if best is None or k < best[0]:
                best, src = (k, net), r["key"]
        net = best[1]
        v = save_pick(net, name, src, kind)
        s = N.size(net)
        print(f"| {name} | {src} | {hid(s['hidden'])} | {s['nonzero']} | {s['weights']} | {s['biases']} "
              f"({s['nonzero_biases']}) | {s['params']} | {100 * v['optimal_share']:.2f} | "
              f"{100 * v['tree_optimal_share']:.2f} | {v['as_x']['win']}/{v['as_x']['draw']}/{v['as_x']['loss']} | "
              f"{v['as_o']['win']}/{v['as_o']['draw']}/{v['as_o']['loss']} |", flush=True)

    # reference counts for the comparison section
    dec = G.decision_positions()
    can = {G.canonical(b)[0] for b in dec}
    xcan = [b for b in can if G.to_move(b) == G.X]
    print(f"\nCanonical decision positions: {len(can)} ({len(xcan)} with X to move, "
          f"{sum(1 for b in xcan if len(G.legal(b)) > 1)} of those with more than one empty cell).")


if __name__ == "__main__":
    main()

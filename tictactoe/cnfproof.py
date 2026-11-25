"""Checkable proofs for the exact search: DIMACS export, external solvers with DRAT/LRAT, cube-and-conquer.

The question (same model class and rules as minsat.py / satnet.py): is there a 1-hidden-layer ternary ReLU net of
width H that never loses as X and as O on the canonical board (trits input, argmax over legal cells, lowest index
on ties)? The CNF is satnet.SatNet with the exact `reach` condition over all 627 canonical positions, written to a
file, so any solver and any proof checker can be pointed at it.

What makes the CNF a proof about *every* integer net (bias mode `general`), given a net (W1, b1, W2, b2):

  1. b1. A unit with r nonzero input weights has W1 x in [-r, r] on every input in {-1, 0, 1}^9. If b1 <= -r the
     unit is 0 everywhere (dead); setting b1 = -r keeps that. If b1 > r it is positive everywhere, so
     h = (W1 x + r) + (b1 - r): set b1 = r and add (b1 - r) W2[k][j] to b2_k. Same moves everywhere. So -r <= b1 <= r.
  2. b2. Then 0 <= h_j <= 2 r_j, and for two cells s, m: |sum_j (W2[s][j] - W2[m][j]) h_j| <= 2 sum_j 2 r_j <= 36 H.
     Whether s beats m depends on d = b2_s - b2_m only through clamp(d, -(36H+1), 36H+1). Shift b2 so min = 0, sort
     it, and cap each gap between consecutive values at 36H+1: every pairwise d is kept or stays beyond the cap on
     the same side. So b2 in [0, 8 (36 H + 1)].
  3. Units. Permuting hidden units (W1 row, b1, W2 column together) changes nothing, so sort units by the key
     (b1, W1 row, W2 column), non-increasing (`strong`; the older runs used (b1, W1 row) only).
  4. Used units (`--require-used`, `strong`): if some unit has no input weight (h constant, folds into b2), no output
     weight, or b1 = -r (dead), the same moves come from a net of width H-1. So once width H-1 is UNSAT for general
     biases, every unit of a width-H net can be assumed used and -r + 1 <= b1. A width-H UNSAT under require-used
     is therefore only a proof together with the width H-1 UNSAT (general); the report states both.
  5. Redundant clauses (`strong`): per position, "s beats m" is a strict total order on the legal cells, so
     transitivity clauses and "some legal cell is the argmax" hold for every real net. They only help propagation.

`normalise` applies 1-3 to a concrete net; tests/test_cnfproof.py checks on saved and random nets (wild biases)
that it never changes a move, and that the CNF accepts every normalised never-losing net and rejects losing ones.
So an UNSAT CNF, checked by drat-trim / lrat-check, means: no net of width H (with the H-1 result for 4), for any
integer biases. The proof checker certifies the CNF is unsatisfiable; that the CNF says what is written above rests
on this encoder and its tests.

Subcommands (pip install -r requirements-sat.txt). External tools are read from $TTT_TOOLS (default: tools/
next to this file), built from source:

    git clone https://github.com/arminbiere/kissat     tools/kissat     (./configure && make  -> build/kissat)
    git clone https://github.com/arminbiere/cadical    tools/cadical    (./configure && make  -> build/cadical)
    git clone https://github.com/marijnheule/drat-trim tools/drat-trim (make                 -> drat-trim, lrat-check)
    git clone https://github.com/marijnheule/CnC       tools/CnC        (make in march_cu     -> march_cu/march_cu)

    build    --hidden 3 --bias general --require-used --strong --out DIR/h3      -> h3.cnf, h3.map.json
    solve    --cnf DIR/h3.cnf --solver kissat|cadical [--proof] [--timeout S]     -> one JSON row
    cubes    --cnf DIR/h3.cnf --out DIR/cubes.icnf -d 12                          march_cu lookahead cubes
    conquer  --cnf DIR/h3.cnf --cubes DIR/cubes.icnf --dir DIR/cnc --workers 8     checkpointed; resumes
    cover    --cnf DIR/h3.cnf --dir DIR/cnc                                        F and not(every leaf cube) is UNSAT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import game as G  # noqa: E402
import minsat as M  # noqa: E402
import net as N  # noqa: E402

TOOLS = os.environ.get("TTT_TOOLS", os.path.join(HERE, "tools"))
KISSAT = os.path.join(TOOLS, "kissat/build/kissat")
CADICAL = os.path.join(TOOLS, "cadical/build/cadical")
DRAT_TRIM = os.path.join(TOOLS, "drat-trim/drat-trim")
LRAT_CHECK = os.path.join(TOOLS, "drat-trim/lrat-check")
MARCH = os.path.join(TOOLS, "CnC/march_cu/march_cu")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ---- normalisation (the argument in the docstring, executable) -----------------------------------------------------
def normalise(W1, b1, W2, b2, strong=True, P=None):
    """Returns an equivalent net inside the `general` ranges, units sorted by the CNF's key (non-increasing).
    With P (a minsat.Problem) also the positions step: a unit active at every position of P is shifted down until
    it is inactive at one, the shift folded into b2 (moves at P's positions unchanged)."""
    H = len(W1)
    W1 = [list(r) for r in W1]
    W2 = [list(r) for r in W2]
    b1 = list(b1)
    b2 = list(b2)
    for j in range(H):
        r = sum(1 for v in W1[j] if v)
        if b1[j] < -r:
            b1[j] = -r
        elif b1[j] > r:
            for k in range(9):
                b2[k] += (b1[j] - r) * W2[k][j]
            b1[j] = r
        if P is not None:
            t = min(sum(w * v for w, v in zip(W1[j], x)) + b1[j] for x in P.x)
            if t > 0:
                for k in range(9):
                    b2[k] += t * W2[k][j]
                b1[j] -= t
    cap = 36 * H + 1
    order = sorted(range(9), key=lambda k: b2[k])
    out = [0] * 9
    prev = None
    for k in order:
        out[k] = 0 if prev is None else out[prev] + min(b2[k] - b2[prev], cap)
        prev = k
    b2 = out

    def key(j):
        t = [b1[j]] + [W1[j][i] for i in range(8, -1, -1)]
        if strong:
            t += [W2[k][j] for k in range(9)]
        return t

    perm = sorted(range(H), key=key, reverse=True)
    return ([W1[j] for j in perm], [b1[j] for j in perm], [[W2[k][j] for j in perm] for k in range(9)], b2)


# ---- DIMACS export -------------------------------------------------------------------------------------------------
class Sink:
    """Clause collector for satnet.SatNet: streams clauses to a body file, counts them."""
    is_sink = True

    def __init__(self, path):
        self.f = open(path, "w")
        self.n = 0

    def add_clause(self, cl):
        self.f.write(" ".join(map(str, cl)) + " 0\n")
        self.n += 1

    def close(self):
        self.f.close()


def max_canonical(b):
    """The largest of the 8 images instead of the smallest (same coding and SYMS order); robustness test only."""
    best = None
    for p in G.SYMS:
        t = G.transform(b, p)
        if best is None or t > best[0]:
            best = (t, p)
    return best


def build(H, bias="general", require_used=False, strong=True, enc="binary", sides="xo", sym=True, max_nonzero=None,
          out="h", positions=None, units=(), tie_order=False, canon="min"):
    import satnet

    if canon == "max":  # the frame the net sees: largest image (only Problem's frame changes)
        saved, G.canonical = G.canonical, max_canonical
        try:
            P = M.Problem(sides, sym)
        finally:
            G.canonical = saved
    else:
        P = M.Problem(sides, sym)
    body = out + ".body"
    sink = Sink(body)
    S = satnet.SatNet(P, H, "reach", bias, 3, 6, require_used=require_used, enc=enc, sink=sink, strong=strong,
                      tie_order=tie_order)
    pos = sorted(range(len(P.boards))) if positions is None else sorted(positions)
    for i in pos:
        S.add_position(i)
    S.finish_positions()
    if max_nonzero is not None:
        for lit in S.set_max_nonzero(max_nonzero):
            sink.add_clause([lit])
    for lit in units:
        sink.add_clause([lit])
    sink.close()
    nv = S.C.top
    with open(out + ".cnf", "w") as f:
        f.write(f"c ttt never-lose H={H} bias={bias} require_used={require_used} strong={strong} enc={enc} "
                f"sides={sides} sym={sym} max_nonzero={max_nonzero} positions={len(pos)}"
                + (f" tie_order={tie_order}" if tie_order else "") + (f" canon={canon}" if canon != "min" else "") + "\n")
        f.write(f"p cnf {nv} {sink.n}\n")
        with open(body) as b:
            shutil.copyfileobj(b, f, 1 << 20)
    os.remove(body)
    mp = {"hidden": H, "bias": bias, "require_used": require_used, "strong": strong, "enc": enc, "sides": sides,
          "sym": sym, "max_nonzero": max_nonzero, "positions": len(pos), "vars": nv, "clauses": sink.n,
          "tie_order": tie_order, "canon": canon, "T": {f"{a},{b}": v for (a, b), v in (S.T or {}).items() if a < b},
          "l1": S.l1, "u1": S.u1, "u2": S.u2, "w1": S.w1, "w2": S.w2, "B1": S.B1, "B2": S.B2,
          "sha256": sha256(out + ".cnf"), "built": now()}
    write_json(out + ".map.json", mp)
    return mp, S


def decode(mp, true):
    """Net from the set of true variables of a model."""
    val = lambda v: v in true  # noqa: E731
    if mp["enc"] == "binary":
        num = lambda bits: sum(1 << t for t, b in enumerate(bits) if val(b))  # noqa: E731
    else:
        num = lambda U: sum(1 for b in U if val(b))  # noqa: E731
    W1 = [[int(val(p)) - int(val(n)) for p, n in row] for row in mp["w1"]]
    W2 = [[int(val(p)) - int(val(n)) for p, n in row] for row in mp["w2"]]
    b1 = [num(B) + mp["l1"] for B in mp["B1"]]
    b2 = [num(B) for B in mp["B2"]]
    return W1, b1, W2, b2


def pin_units(mp, W1, b1, W2, b2):
    """Unit literals that fix the CNF's weights to a (normalised) net."""
    out = []
    for R, Mx in ((mp["w1"], W1), (mp["w2"], W2)):
        for row, mrow in zip(R, Mx):
            for (p, n), v in zip(row, mrow):
                out += [p if v == 1 else -p, n if v == -1 else -n]
    for bits, v in list(zip(mp["B1"], [b - mp["l1"] for b in b1])) + list(zip(mp["B2"], b2)):
        if mp["enc"] == "binary":
            if v < 0 or v >= 1 << len(bits):
                raise ValueError("bias out of range")
            out += [b if (v >> t) & 1 else -b for t, b in enumerate(bits)]
        else:
            if not 0 <= v <= len(bits):
                raise ValueError("bias out of range")
            out += [b if t < v else -b for t, b in enumerate(bits)]
    return out


def check_model(mp, true):
    """Decode, re-check with the pure-Python reference over the full game tree. Returns (ok, net, verify-result)."""
    W1, b1, W2, b2 = decode(mp, true)
    P = M.Problem(mp["sides"], mp["sym"])
    net = P.to_net(W1, b1, W2, b2)
    ok, res = M.check_net(net, mp["sides"])
    return ok and not P.failures(W1, b1, W2, b2, "reach"), net, res


# ---- solvers and checkers ------------------------------------------------------------------------------------------
def run_solver(cnf, solver="kissat", proof=None, timeout=None, seed=0, extra=(), log=None):
    """Returns (status, seconds, true-variable set or None). proof: kissat -> binary DRAT, cadical -> text LRAT."""
    if solver == "kissat":
        cmd = [KISSAT, "-q", f"--seed={seed}"] + list(extra)
        if timeout:
            cmd.append(f"--time={int(timeout)}")
        cmd.append(cnf)
        if proof:
            cmd.append(proof)
    else:
        cmd = [CADICAL, "-q", f"--seed={seed}"] + list(extra)
        if timeout:
            cmd += ["-t", str(int(timeout))]
        if proof:
            cmd += ["--lrat=true", "--binary=false"]
        cmd.append(cnf)
        if proof:
            cmd.append(proof)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    true = set()
    lines = []
    for line in p.stdout:
        if line.startswith("v "):
            true.update(int(t) for t in line[2:].split() if int(t) > 0)
        elif log is not None:
            lines.append(line)
    rc = p.wait()
    dt = time.time() - t0
    if log is not None:
        log.write("".join(lines[-50:]))
    if rc == 10:
        return "SAT", dt, true
    if rc == 20:
        return "UNSAT", dt, None
    return "UNKNOWN", dt, None


def check_proof(cnf, proof, timeout=10 ** 7):
    """drat-trim for DRAT, lrat-check for LRAT (by file suffix). Returns (verified, seconds, tail of output)."""
    t0 = time.time()
    if proof.endswith(".lrat"):
        cmd = [LRAT_CHECK, cnf, proof]
        ok_mark = "c VERIFIED"
    else:
        cmd = [DRAT_TRIM, cnf, proof, "-t", str(timeout)]
        ok_mark = "s VERIFIED"
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    ok = any(line.strip() == ok_mark for line in out.splitlines())
    return ok, time.time() - t0, out[-600:]


def compress(path):
    """xz in place (proofs are kept, not committed). Returns the compressed path."""
    subprocess.run(["xz", "-T1", "-6", "-f", path], check=True)
    return path + ".xz"


# ---- cubes ---------------------------------------------------------------------------------------------------------
def read_cubes(path):
    out = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if t and t[0] == "a":
                out.append([int(x) for x in t[1:] if x != "0"])
    return out


def march(cnf, out, depth=None, nvars=None, extra=()):
    cmd = [MARCH, cnf, "-o", out] + (["-d", str(depth)] if depth else []) + (["-n", str(nvars)] if nvars else [])
    cmd += list(extra)

    def big_stack():  # march_cu recurses deeply and segfaults on these CNFs at the default 8 MB stack
        import resource
        hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, preexec_fn=big_stack)
    return p.returncode, p.stdout[-1500:]


def trit_cubes(mp, spec):
    """Cubes = every value of the listed weight trits, e.g. "w1:0:4,w1:0:2,w2:4:0" (layer:row:col). Each trit is
    (p, n) with not (p and n), so its three values (p, -n), (-p, n), (-p, -n) cover it; the product covers F."""
    pairs = []
    for t in spec.split(","):
        layer, r, c = t.split(":")
        pairs.append(mp[layer][int(r)][int(c)])
    cubes = [[]]
    for p, n in pairs:
        cubes = [cu + v for cu in cubes for v in ([p, -n], [-p, n], [-p, -n])]
    return cubes


def write_cubes(cubes, path):
    with open(path, "w") as f:
        for cu in cubes:
            f.write("a " + " ".join(map(str, cu)) + " 0\n")


def cube_cnf(base, cube, out):
    """base CNF + the cube as unit clauses."""
    with open(base) as f:
        head = []
        while True:
            line = f.readline()
            if line.startswith("p "):
                _, _, nv, nc = line.split()
                break
            head.append(line)
        with open(out, "w") as g:
            g.writelines(head)
            g.write(f"p cnf {nv} {int(nc) + len(cube)}\n")
            shutil.copyfileobj(f, g, 1 << 20)
            for lit in cube:
                g.write(f"{lit} 0\n")


# ---- conquer: checkpointed worker pool -----------------------------------------------------------------------------
def conquer(a):
    """Every cube: solve F + cube with a proof, check it, keep the proof (xz) with its sha256. A cube that times out
    at a level below --max-level is split again by march_cu (its children replace it in the leaf set). Rows go to
    DIR/cubes.jsonl (append-only checkpoint); a restart skips cubes already decided."""
    d = os.path.abspath(a.dir)
    for sub in ("proofs", "tmp", "logs", "nets"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    with open(os.path.join(d, "conquer.pid"), "w") as f:
        f.write(str(os.getpid()))
    mp = json.load(open(a.cnf[:-4] + ".map.json"))
    rows_path = os.path.join(d, "cubes.jsonl")
    leaves_path = os.path.join(d, "leaves.jsonl")  # id -> cube, parent, level; append-only
    if not os.path.exists(leaves_path):
        with open(leaves_path, "w") as f:
            for n, c in enumerate(read_cubes(a.cubes)):
                f.write(json.dumps({"id": str(n), "cube": c, "level": 0}) + "\n")
    leaves = {}
    for line in open(leaves_path):
        r = json.loads(line)
        leaves[r["id"]] = r
    done = {}
    if os.path.exists(rows_path):
        for line in open(rows_path):
            r = json.loads(line)
            done[r["id"]] = r
    for m in ("DONE-UNSAT", "DONE-SAT", "FAILED"):
        if os.path.exists(os.path.join(d, m)):
            os.remove(os.path.join(d, m))
    started = now()
    t_start = time.time()
    running = {}  # id -> (Popen, t0)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    def status(state, extra=None):
        st = {"state": state, "pid": os.getpid(), "started": started, "updatedAt": now(), "cnf": a.cnf,
              "cnfSha256": mp["sha256"], "leaves": len(leaves), "decided": 0, "unsat": 0, "sat": 0, "split": 0,
              "failed": 0, "running": len(running), "cubeSeconds": 0.0, "maxCubeSeconds": 0.0,
              "proofBytesXz": 0, "workers": a.workers, "cubeTimeout": a.cube_timeout}
        for r in done.values():
            st["decided"] += r["status"] in ("UNSAT", "SAT", "SPLIT")
            st["unsat"] += r["status"] == "UNSAT"
            st["sat"] += r["status"] == "SAT"
            st["split"] += r["status"] == "SPLIT"
            st["failed"] += r["status"] in ("CHECK-FAILED", "ERROR", "UNKNOWN")
            st["cubeSeconds"] += r.get("seconds", 0)
            st["maxCubeSeconds"] = max(st["maxCubeSeconds"], r.get("seconds", 0))
            st["proofBytesXz"] += r.get("proofBytesXz", 0)
        open_ = [i for i in leaves if i not in done]
        st["open"] = len(open_)
        st["cubeSeconds"] = round(st["cubeSeconds"])
        st["maxCubeSeconds"] = round(st["maxCubeSeconds"])
        n_unsat = max(1, st["unsat"])
        st["etaHours"] = round(st["cubeSeconds"] / n_unsat * st["open"] / max(1, a.workers) / 3600, 1)
        st["wallHours"] = round((time.time() - t_start) / 3600, 2)
        if extra:
            st.update(extra)
        write_json(os.path.join(d, "status.json"), st)

    worker_py = [sys.executable, os.path.abspath(__file__), "one-cube", "--cnf", a.cnf, "--dir", d,
                 "--solver", a.solver, "--cube-timeout", str(a.cube_timeout), "--final-timeout", str(a.final_timeout),
                 "--max-level", str(a.max_level), "--split", a.split, "--keep-gb", str(a.keep_gb),
                 "--split-depth", str(a.split_depth)]
    while True:
        for cid, (p, t0) in list(running.items()):
            if p.poll() is not None:
                del running[cid]
                rp = os.path.join(d, "tmp", cid + ".row.json")
                if os.path.exists(rp):
                    r = json.load(open(rp))
                    os.remove(rp)
                else:
                    r = {"id": cid, "status": "ERROR", "returncode": p.returncode}
                done[cid] = r
                with open(rows_path, "a") as f:
                    f.write(json.dumps(r) + "\n")
                if r["status"] == "SPLIT":
                    with open(leaves_path, "a") as f:
                        for ch in r["children"]:
                            f.write(json.dumps(ch) + "\n")
                            leaves[ch["id"]] = ch
                if r["status"] == "SAT":
                    for q, _ in running.values():
                        q.terminate()
                    open(os.path.join(d, "DONE-SAT"), "w").write(json.dumps(r) + "\n")
                    status("done-sat")
                    return 0
                if r["status"] in ("CHECK-FAILED", "ERROR"):
                    open(os.path.join(d, "FAILED"), "w").write(json.dumps(r) + "\n")
        if stop["flag"]:
            for q, _ in running.values():
                q.terminate()
            status("stopped")
            return 1
        todo = [i for i in leaves if i not in done and i not in running]
        # deeper (re-split) cubes first, so a hard region is finished before new ones open
        todo.sort(key=lambda i: (-leaves[i]["level"], i))
        while todo and len(running) < a.workers and mem_free_gb() >= a.min_free_gb:
            cid = todo.pop(0)
            cf = os.path.join(d, "tmp", cid + ".cube.json")
            write_json(cf, leaves[cid])
            log = open(os.path.join(d, "logs", cid + ".log"), "w")
            p = subprocess.Popen(["nice", "-n", str(a.nice)] + worker_py + ["--cube-file", cf],
                                 stdout=log, stderr=subprocess.STDOUT)
            running[cid] = (p, time.time())
        status("running")
        if not running and not todo:
            bad = [r for r in done.values() if r["status"] in ("CHECK-FAILED", "ERROR", "UNKNOWN")]
            if bad:
                open(os.path.join(d, "FAILED"), "w").write(json.dumps(bad[:5]) + "\n")
                status("failed")
                return 1
            open(os.path.join(d, "DONE-UNSAT"), "w").write(now() + "\n")
            status("done-unsat")
            return 0
        time.sleep(5)


def mem_free_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 1e9


def one_cube(a):
    leaf = json.load(open(a.cube_file))
    cid, cube = leaf["id"], leaf["cube"]
    d = a.dir
    mp = json.load(open(a.cnf[:-4] + ".map.json"))
    tmp_cnf = os.path.join(d, "tmp", cid + ".cnf")
    cube_cnf(a.cnf, cube, tmp_cnf)
    ext = ".lrat" if a.solver == "cadical" else ".drat"
    proof = os.path.join(d, "tmp", cid + ext)
    row = {"id": cid, "level": leaf["level"], "len": len(cube), "solver": a.solver, "startedAt": now()}
    timeout = a.cube_timeout if leaf["level"] < a.max_level else a.final_timeout
    st, dt, true = run_solver(tmp_cnf, a.solver, proof, timeout)
    row["seconds"] = round(dt, 2)
    row["status"] = st
    if st == "UNSAT":
        ok, ct, tail = check_proof(tmp_cnf, proof)
        row["checkSeconds"] = round(ct, 2)
        row["proofBytes"] = os.path.getsize(proof)
        row["proofSha256"] = sha256(proof)
        if not ok:
            row["status"] = "CHECK-FAILED"
            row["checkTail"] = tail
        else:
            kept = sum(os.path.getsize(os.path.join(d, "proofs", f)) for f in os.listdir(os.path.join(d, "proofs")))
            if kept < a.keep_gb * 1e9:
                dst = os.path.join(d, "proofs", cid + ext)
                os.replace(proof, dst)
                z = compress(dst)
                row["proof"] = os.path.relpath(z, d)
                row["proofBytesXz"] = os.path.getsize(z)
            else:  # over budget: checked, sha256 recorded, deleted (CaDiCaL is deterministic: re-run to regenerate)
                row["proof"] = None
    elif st == "SAT":
        ok, net, res = check_model(mp, true)
        row["verified"] = ok
        row["as_x"], row["as_o"] = res["as_x"], res["as_o"]
        row["nonzero"] = N.size(net)["nonzero"]
        N.save(net, os.path.join(d, "nets", cid + ".json"), verified={k: res[k] for k in ("as_x", "as_o", "perfect")},
               size=N.size(net), source={"cnf": a.cnf, "cnfSha256": mp["sha256"], "cube": cid})
        if not ok:
            row["status"] = "ERROR"
            row["error"] = "SAT model fails the reference verifier"
    else:  # timed out: split again
        if leaf["level"] < a.max_level:
            kids = []
            if a.split == "trits":  # the next split_depth undecided trits of the fixed order: 3^k children
                order = [mp["w1"][j][i] for j in range(mp["hidden"]) for i in range(8, -1, -1)]
                order += [mp["w2"][k][j] for j in range(mp["hidden"]) for k in range(9)]
                used = {abs(x) for x in cube}
                nxt = [pn for pn in order if pn[0] not in used and pn[1] not in used][:a.split_depth]
                kids = [[]]
                for p, n in nxt:
                    kids = [cu + v for cu in kids for v in ([p, -n], [-p, n], [-p, -n])]
                if not nxt:
                    kids = []
            else:
                sub = os.path.join(d, "tmp", cid + ".icnf")
                rc, out = march(tmp_cnf, sub, depth=a.split_depth)
                kids = read_cubes(sub) if os.path.exists(sub) else []
                row["march"] = out[-300:]
                if os.path.exists(sub):
                    os.remove(sub)
            if kids:
                row["status"] = "SPLIT"
                row["children"] = [{"id": f"{cid}.{n}", "cube": cube + [x for x in k if x not in cube],
                                    "level": leaf["level"] + 1, "parent": cid} for n, k in enumerate(kids)]
            else:  # nothing left to split on, or march produced nothing: stays an open leaf (UNKNOWN)
                row["status"] = "UNKNOWN"
    for f in (tmp_cnf, proof):
        if os.path.exists(f):
            os.remove(f)
    row["finishedAt"] = now()
    write_json(os.path.join(d, "tmp", cid + ".row.json"), row)
    return 0


def cover(a):
    """The leaf cubes (cubes never split, whatever their status must be UNSAT) cover F: F and not(c) for every leaf c
    is UNSAT, solved with a proof and checked. A SPLIT cube is replaced by its children (march_cu's leaves, which
    cover F and the parent), so the leaf set at the end covers F."""
    d = os.path.abspath(a.dir)
    rows = {}
    for line in open(os.path.join(d, "cubes.jsonl")):
        r = json.loads(line)
        rows[r["id"]] = r
    leaves = [json.loads(line) for line in open(os.path.join(d, "leaves.jsonl"))]
    leaf_cubes = [lv["cube"] for lv in leaves if rows.get(lv["id"], {}).get("status") != "SPLIT"]
    open_ = [lv["id"] for lv in leaves if lv["id"] not in rows]
    bad = [lv["id"] for lv in leaves if rows.get(lv["id"], {}).get("status") not in ("UNSAT", "SPLIT", None)]
    out = os.path.join(d, "cover.cnf")
    with open(a.cnf) as f:
        head = []
        while True:
            line = f.readline()
            if line.startswith("p "):
                _, _, nv, nc = line.split()
                break
            head.append(line)
        with open(out, "w") as g:
            g.writelines(head)
            g.write(f"p cnf {nv} {int(nc) + len(leaf_cubes)}\n")
            shutil.copyfileobj(f, g, 1 << 20)
            for c in leaf_cubes:
                g.write(" ".join(str(-x) for x in c) + " 0\n")
    proof = os.path.join(d, "cover.lrat")
    st, dt, _ = run_solver(out, "cadical", proof, a.timeout)
    row = {"leaves": len(leaf_cubes), "open": len(open_), "notUnsat": bad[:10], "status": st, "seconds": round(dt, 1)}
    if st == "UNSAT":
        ok, ct, tail = check_proof(out, proof)
        row.update({"verified": ok, "checkSeconds": round(ct, 1), "proofSha256": sha256(proof),
                    "proofBytes": os.path.getsize(proof), "coverCnfSha256": sha256(out)})
        z = compress(proof)
        row["proof"] = os.path.relpath(z, d)
        if not ok:
            row["checkTail"] = tail
    os.remove(out)
    row["finishedAt"] = now()
    write_json(os.path.join(d, "cover.json"), row)
    print(json.dumps(row))
    return 0 if row.get("verified") and not open_ and not bad else 1


# ---- solve: one CNF, one solver, optional proof --------------------------------------------------------------------
def solve_cmd(a):
    mp = json.load(open(a.cnf[:-4] + ".map.json"))
    ext = ".lrat" if a.solver == "cadical" else ".drat"
    proof = (a.cnf[:-4] + f".{a.solver}{a.seed}{ext}") if a.proof else None
    extra = a.extra.split() if a.extra else []
    st, dt, true = run_solver(a.cnf, a.solver, proof, a.timeout, a.seed, extra, log=sys.stdout)
    row = {"cnf": a.cnf, "cnfSha256": mp["sha256"], "hidden": mp["hidden"], "bias": mp["bias"],
           "require_used": mp["require_used"], "strong": mp["strong"], "enc": mp["enc"],
           "max_nonzero": mp["max_nonzero"], "vars": mp["vars"], "clauses": mp["clauses"], "solver": a.solver,
           "seed": a.seed, "extra": a.extra, "status": st, "seconds": round(dt, 1), "finishedAt": now()}
    if st == "UNSAT" and proof:
        row["proofBytes"] = os.path.getsize(proof)
        row["proofSha256"] = sha256(proof)
        ok, ct, tail = check_proof(a.cnf, proof)
        row.update({"verified": ok, "checkSeconds": round(ct, 1)})
        if not ok:
            row["checkTail"] = tail
        if a.keep_proof:
            row["proof"] = compress(proof)
        else:
            os.remove(proof)
    elif st == "SAT":
        ok, net, res = check_model(mp, true)
        row.update({"verified": ok, "as_x": res["as_x"], "as_o": res["as_o"], "nonzero": N.size(net)["nonzero"]})
        if a.net_out:
            N.save(net, a.net_out, verified={k: res[k] for k in ("as_x", "as_o", "perfect")}, size=N.size(net),
                   source={"cnf": os.path.basename(a.cnf), "cnfSha256": mp["sha256"], "solver": a.solver})
    elif proof and os.path.exists(proof):
        os.remove(proof)
    print(json.dumps(row), flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(row) + "\n")
    return 0


# ---- portfolio: several configs on one problem, first checked answer wins ------------------------------------------
def portfolio(a):
    """Members "cnf|solver|seed|proof(0/1)|extra" run in parallel under nice. The first SAT (verified net) or
    UNSAT (proof checked, when the member writes one) stops the rest. An UNSAT from a member without a proof is
    recorded but does not stop the run (it is re-run with a proof when --rerun-with-proof)."""
    d = os.path.abspath(a.dir)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "portfolio.pid"), "w") as f:
        f.write(str(os.getpid()))
    for m in ("DONE-UNSAT", "DONE-SAT", "DONE-UNSAT-UNCHECKED", "FAILED"):
        if os.path.exists(os.path.join(d, m)):
            os.remove(os.path.join(d, m))
    rows_path = os.path.join(d, "results.jsonl")
    members = []
    for n, spec in enumerate(a.member):
        cnf, solver, seed, proof, *extra = spec.split("|")
        members.append({"n": n, "cnf": cnf, "solver": solver, "seed": int(seed), "proof": proof == "1",
                        "extra": extra[0] if extra else ""})
    started, t0 = now(), time.time()
    running = {}
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    def launch(m):
        cmd = ["nice", "-n", str(a.nice), sys.executable, os.path.abspath(__file__), "solve", "--cnf", m["cnf"],
               "--solver", m["solver"], "--seed", str(m["seed"]), "--json", rows_path,
               "--net-out", os.path.join(d, f"net-{m['n']}.json")]
        if a.timeout:
            cmd += ["--timeout", str(a.timeout)]
        if m["extra"]:
            cmd += ["--extra=" + m["extra"]]
        if m["proof"]:
            cmd += ["--proof"] + (["--keep-proof"] if a.keep_proof else [])
        log = open(os.path.join(d, f"member-{m['n']}.log"), "w")
        running[m["n"]] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), time.time(), m)

    def rows():
        return [json.loads(line) for line in open(rows_path)] if os.path.exists(rows_path) else []

    def status(state):
        st = {"state": state, "name": a.name, "pid": os.getpid(), "started": started, "updatedAt": now(),
              "wallHours": round((time.time() - t0) / 3600, 2), "timeout": a.timeout,
              "running": [{"n": n, "solver": m["solver"], "seed": m["seed"], "proof": m["proof"],
                           "cnf": os.path.basename(m["cnf"]), "extra": m["extra"], "pid": p.pid,
                           "hours": round((time.time() - t) / 3600, 2)} for n, (p, t, m) in running.items()],
              "results": [{k: r.get(k) for k in ("solver", "seed", "cnf", "status", "seconds", "verified",
                                                  "checkSeconds", "nonzero")} for r in rows()],
              "memFreeGB": round(mem_free_gb(), 1)}
        write_json(os.path.join(d, "status.json"), st)

    def kill_all():
        for p, _, _ in running.values():
            # the solve wrapper and its solver child: kill the solver first
            subprocess.run(["pkill", "-TERM", "-P", str(p.pid)])
            p.terminate()

    for m in members:
        launch(m)
    while True:
        for n, (p, _, m) in list(running.items()):
            if p.poll() is not None:
                del running[n]
        win = [r for r in rows() if r.get("verified") and r["status"] in ("SAT", "UNSAT")]
        if win:
            kill_all()
            r = win[0]
            open(os.path.join(d, "DONE-" + r["status"]), "w").write(json.dumps(r) + "\n")
            status("done-" + r["status"].lower())
            return 0
        bad = [r for r in rows() if r["status"] in ("SAT", "UNSAT") and r.get("verified") is False]
        if bad:
            kill_all()
            open(os.path.join(d, "FAILED"), "w").write(json.dumps(bad[0]) + "\n")
            status("failed")
            return 1
        if stop["flag"]:
            kill_all()
            status("stopped")
            return 1
        if not running:
            un = [r for r in rows() if r["status"] == "UNSAT"]
            if un:  # answered, but by members without a proof: a cross-check, not a checked result
                open(os.path.join(d, "DONE-UNSAT-UNCHECKED"), "w").write(json.dumps(un[0]) + "\n")
                status("done-unsat-unchecked")
                return 0
            open(os.path.join(d, "FAILED"), "w").write("no member answered (timeouts)\n")
            status("failed-timeout")
            return 1
        status("running")
        time.sleep(15)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--hidden", type=int, required=True)
    b.add_argument("--bias", default="general", choices=["small", "general"])
    b.add_argument("--require-used", action="store_true")
    b.add_argument("--no-strong", action="store_true")
    b.add_argument("--enc", default="binary", choices=["binary", "unary"])
    b.add_argument("--sides", default="xo", choices=["x", "o", "xo"])
    b.add_argument("--no-sym", action="store_true")
    b.add_argument("--max-nonzero", type=int)
    b.add_argument("--tie-order", action="store_true", help="free strict total tie order (real-valued b2)")
    b.add_argument("--canon", default="min", choices=["min", "max"], help="canonical image (max: robustness test)")
    b.add_argument("--out", required=True, help="path prefix: writes PREFIX.cnf and PREFIX.map.json")
    s = sp.add_parser("solve")
    s.add_argument("--cnf", required=True)
    s.add_argument("--solver", default="kissat", choices=["kissat", "cadical"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--timeout", type=float)
    s.add_argument("--extra", default="", help="extra solver options, space separated")
    s.add_argument("--proof", action="store_true")
    s.add_argument("--keep-proof", action="store_true")
    s.add_argument("--net-out")
    s.add_argument("--json")
    c = sp.add_parser("cubes")
    c.add_argument("--cnf", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("-d", type=int)
    c.add_argument("-n", type=int)
    c.add_argument("--trits", help='split on weight trits instead of march_cu, e.g. "w1:0:4,w1:0:0"')
    q = sp.add_parser("conquer")
    q.add_argument("--cnf", required=True)
    q.add_argument("--cubes", required=True)
    q.add_argument("--dir", required=True)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--solver", default="cadical", choices=["kissat", "cadical"])
    q.add_argument("--cube-timeout", type=float, default=1800)
    q.add_argument("--final-timeout", type=float, default=7 * 86400)
    q.add_argument("--max-level", type=int, default=3)
    q.add_argument("--split-depth", type=int, default=2)
    q.add_argument("--split", default="trits", choices=["trits", "march"])
    q.add_argument("--keep-gb", type=float, default=150, help="keep checked cube proofs (xz) up to this much")
    q.add_argument("--nice", type=int, default=19)
    q.add_argument("--min-free-gb", type=float, default=16.0)
    o = sp.add_parser("one-cube")
    for k in ("--cnf", "--dir", "--cube-file", "--solver"):
        o.add_argument(k, required=True)
    o.add_argument("--cube-timeout", type=float, required=True)
    o.add_argument("--final-timeout", type=float, default=7 * 86400)
    o.add_argument("--max-level", type=int, required=True)
    o.add_argument("--split-depth", type=int, required=True)
    o.add_argument("--split", default="trits", choices=["trits", "march"])
    o.add_argument("--keep-gb", type=float, default=150)
    v = sp.add_parser("cover")
    v.add_argument("--cnf", required=True)
    v.add_argument("--dir", required=True)
    v.add_argument("--timeout", type=float, default=86400)
    f = sp.add_parser("portfolio")
    f.add_argument("--name", required=True)
    f.add_argument("--dir", required=True)
    f.add_argument("--member", action="append", required=True, help='"cnf|solver|seed|proof(0/1)|extra opts"')
    f.add_argument("--timeout", type=float)
    f.add_argument("--keep-proof", action="store_true")
    f.add_argument("--nice", type=int, default=19)
    a = ap.parse_args()
    if a.cmd == "portfolio":
        return portfolio(a)
    if a.cmd == "build":
        mp, _ = build(a.hidden, a.bias, a.require_used, not a.no_strong, a.enc, a.sides, not a.no_sym, a.max_nonzero,
                      a.out, tie_order=a.tie_order, canon=a.canon)
        print(json.dumps({k: mp[k] for k in ("hidden", "bias", "require_used", "strong", "enc", "positions", "vars",
                                             "clauses", "u2", "sha256")}))
        return 0
    if a.cmd == "solve":
        return solve_cmd(a)
    if a.cmd == "cubes" and a.trits:
        cubes = trit_cubes(json.load(open(a.cnf[:-4] + ".map.json")), a.trits)
        write_cubes(cubes, a.out)
        print(json.dumps({"cubes": len(cubes), "scheme": "trits", "trits": a.trits}))
        return 0
    if a.cmd == "cubes":
        rc, out = march(a.cnf, a.out, a.d, a.n)
        print(out)
        print(json.dumps({"cubes": len(read_cubes(a.out)) if os.path.exists(a.out) else 0, "rc": rc}))
        return 0
    if a.cmd == "conquer":
        return conquer(a)
    if a.cmd == "one-cube":
        return one_cube(a)
    if a.cmd == "cover":
        return cover(a)
    return 2


if __name__ == "__main__":
    sys.exit(main())

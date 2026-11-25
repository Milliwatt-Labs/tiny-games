#!/usr/bin/env python3
"""'More games than humanity': the ternary tic-tac-toe nets play >= 2e12 games on the GPU, with exact counts.

Humanity has played on the order of 1e12 games of tic-tac-toe (an order-of-magnitude guess), so the target is
2e12+ games in total across the arms below. The GPU player is ttt_gpu.cu; this file never touches CUDA itself, it
only builds and runs the binary.

Arms (finisher and satxo as X and as O; xonly, the 10-weight X-only net, as X only, arms 39-46):
  A  perfect   opponent picks uniformly among value-keeping moves (game.optimal_moves): expect 100 % draws
  B  random    uniform over empty cells
     kid50/90  takes a win with prob p, else blocks with prob p, else random
     mix25..90 ladder: a perfect move with prob q, else a random one
  C  open1-3   1-3 uniformly random opening moves, then the net plays both sides (losses from lost openings are
               expected and reported by the opening's minimax value, never mixed into A/B)

Every game has its own Philox4x32-10 stream: counter (game index, arm id, block), key (SEED0, SEED1). A count
depends only on (arm, seed, game range), so a chunked, paused and resumed run gives the same totals as one launch.

    python3 humanity.py check            CPU: build ttt_cpu, parity + trace against net.py / game.py
    python3 humanity.py bench            GPU (needs CUDA): parity + trace, then a short timed run per arm kind
    python3 humanity.py supervise        GPU: the long run (run it detached with nohup); checkpoints in state/
    python3 humanity.py status           print status.json
    python3 humanity.py report [results.json]   -> tictactoe/results/humanity.md
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TTT = os.path.dirname(HERE)
sys.path.insert(0, TTT)

import game as G  # noqa: E402
import net as N  # noqa: E402

ROOT = os.environ.get("TTT_HUMANITY_ROOT") or os.path.join(os.getcwd(), "humanity-run")
SEED0, SEED1 = 20261122, 0x48554D41  # "HUMA"
HUMANITY = 1e12
TARGET = 2e12
YIELD_SLEEP_S = 600
CHUNK_S = 30.0  # wall time per binary call, so progress is checkpointed often

NETS = {"finisher": "smallest-finisher.json", "satxo": "smallest-sat-xo.json", "xonly": "smallest-x-only.json"}
NET_SIDES = {"finisher": (1, 2), "satxo": (1, 2), "xonly": (1,)}  # 1 = X, 2 = O; xonly never plays O
BOTH_SIDES = [k for k, v in NET_SIDES.items() if v == (1, 2)]
OPPS = {"perfect": 0, "random": 1, "kid": 2, "mix": 3}


def thresh(p):
    return min(int(round(p * 2 ** 32)), 2 ** 32 - 1)


def _arms():
    """(name, group, net, side, opp kind, p, open_k, budget). Arm ids 1-38 (both-side nets) add up to TARGET + 4 %;
    the X-only net's arms are appended after them (ids 39-46) so earlier ids, and their counts, never move."""
    out = []
    aid = 1
    B_OPPS = [("random", "random", 0.0), ("kid50", "kid", 0.5), ("kid90", "kid", 0.9), ("mix25", "mix", 0.25),
              ("mix50", "mix", 0.5), ("mix75", "mix", 0.75), ("mix90", "mix", 0.9)]
    for net in BOTH_SIDES:
        for side, sname in ((1, "x"), (2, "o")):
            out.append(dict(name=f"A-perfect-{net}-{sname}", group="A", net=net, side=side, opp="perfect", p=0.0,
                            open_k=0, budget=260_000_000_000, arm_id=aid))
            aid += 1
    for net in BOTH_SIDES:
        for side, sname in ((1, "x"), (2, "o")):
            for oname, kind, p in B_OPPS:
                out.append(dict(name=f"B-{oname}-{net}-{sname}", group="B", net=net, side=side, opp=kind, p=p,
                                open_k=0, budget=36_000_000_000, arm_id=aid))
                aid += 1
    for net in BOTH_SIDES:
        for k in (1, 2, 3):
            out.append(dict(name=f"C-open{k}-{net}", group="C", net=net, side=0, opp="random", p=0.0, open_k=k,
                            budget=5_000_000_000, arm_id=aid))
            aid += 1
    for net in (k for k, v in NET_SIDES.items() if v == (1,)):
        out.append(dict(name=f"A-perfect-{net}-x", group="A", net=net, side=1, opp="perfect", p=0.0, open_k=0,
                        budget=260_000_000_000, arm_id=aid))
        aid += 1
        for oname, kind, p in B_OPPS:
            out.append(dict(name=f"B-{oname}-{net}-x", group="B", net=net, side=1, opp=kind, p=p, open_k=0,
                            budget=36_000_000_000, arm_id=aid))
            aid += 1
    return out


ARMS = _arms()
assert all(a["side"] in NET_SIDES[a["net"]] or (a["side"] == 0 and a["net"] in BOTH_SIDES) for a in ARMS)
assert len({a["arm_id"] for a in ARMS}) == len(ARMS)


def arm_argv(a):
    return [str(a["side"]), str(OPPS[a["opp"]]), str(thresh(a["p"])), str(a["open_k"]), str(SEED0), str(SEED1),
            str(a["arm_id"])]


# ---- tables from game.py ------------------------------------------------------------------------------------------
def code3(b):
    return sum(v * 3 ** i for i, v in enumerate(b))


def board_of(c):
    out = []
    for _ in range(9):
        out.append(c % 3)
        c //= 3
    return tuple(out)


def write_tables(path):
    opt = [0] * 19683
    valx = [0] * 19683
    for b in G.all_positions():
        c = code3(b)
        if G.is_terminal(b):
            w = G.winner(b)
            valx[c] = 1 if w == G.X else (-1 if w == G.O else 0)
        else:
            v = G.value(b)
            valx[c] = v if G.to_move(b) == G.X else -v
            opt[c] = sum(1 << m for m in G.optimal_moves(b))
    with open(path, "wb") as f:
        f.write(b"TTT1")
        f.write(bytes(v for p in G.SYMS for v in p))
        f.write(struct.pack("<19683H", *opt))
        f.write(struct.pack("<19683b", *valx))


def write_net(net, path):
    L = net["layers"]
    assert len(L) == 2, "ttt_gpu.cu handles one hidden layer"
    W1, b1, W2, b2 = L[0]["W"], L[0]["b"], L[1]["W"], L[1]["b"]
    H = len(b1)
    assert H <= 32
    pos = [sum(1 << i for i, w in enumerate(r) if w == 1) for r in W1]
    neg = [sum(1 << i for i, w in enumerate(r) if w == -1) for r in W1]
    with open(path, "wb") as f:
        f.write(b"TNET")
        f.write(struct.pack("<4i", H, net["encoding"] == "planes", net["act"] == "step", bool(net.get("symmetry"))))
        f.write(struct.pack(f"<{H}I", *pos))
        f.write(struct.pack(f"<{H}I", *neg))
        f.write(struct.pack(f"<{H}i", *b1))
        for k in range(9):
            f.write(struct.pack(f"<{H}i", *[W2[k][j] for j in range(H)]))
        f.write(struct.pack("<9i", *b2))


def load_nets():
    return {k: N.load(os.path.join(TTT, "weights", v)) for k, v in NETS.items()}


# ---- pure-Python reference player (independent of the binary) -----------------------------------------------------
M32 = 0xFFFFFFFF


def philox(c, k):
    c0, c1, c2, c3 = c
    k0, k1 = k
    for _ in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & M32, p1 & M32, ((p0 >> 32) ^ c3 ^ k1) & M32, p0 & M32
        k0, k1 = (k0 + 0x9E3779B9) & M32, (k1 + 0xBB67AE85) & M32
    return [c0, c1, c2, c3]


class Rng:
    def __init__(self, game, arm):
        self.c = [game & M32, game >> 32, arm["arm_id"], 0]
        self.k = (SEED0, SEED1)
        self.buf, self.i = [], 4

    def next(self):
        if self.i == 4:
            self.buf = philox(self.c, self.k)
            self.c[3] += 1
            self.i = 0
        self.i += 1
        return self.buf[self.i - 1]


def pick(cells, u):
    """cells ascending; the floor(u * len / 2^32)-th one."""
    return sorted(cells)[(u * len(cells)) >> 32]


def completing_for(b, side):
    out = []
    for m in G.legal(b):
        s = list(b)
        s[m] = side
        if G.winner(tuple(s)) == side:
            out.append(m)
    return out


class Reference:
    def __init__(self, nets):
        self.nets = nets
        self.moves = {}  # (net, board) -> net.choose_move, memoised

    def net_move(self, name, b):
        key = (name, b)
        if key not in self.moves:
            self.moves[key] = N.choose_move(self.nets[name], b)
        return self.moves[key]

    def opp_move(self, arm, g, b):
        kind, t = arm["opp"], thresh(arm["p"])
        me = G.to_move(b)
        if kind == "perfect":
            return pick(G.optimal_moves(b), g.next())
        if kind == "kid":
            w = completing_for(b, me)
            if w and g.next() < t:
                return pick(w, g.next())
            bl = completing_for(b, G.O if me == G.X else G.X)
            if bl and g.next() < t:
                return pick(bl, g.next())
            return pick(G.legal(b), g.next())
        if kind == "mix":
            if g.next() < t:
                return pick(G.optimal_moves(b), g.next())
        return pick(G.legal(b), g.next())

    def play(self, arm, game):
        g = Rng(game, arm)
        b = G.START
        moves = []
        for _ in range(arm["open_k"]):
            m = pick(G.legal(b), g.next())
            moves.append(m)
            b = G.play(b, m)
        v = G.value(b)
        vx = v if G.to_move(b) == G.X else -v
        theory = 0 if vx > 0 else (2 if vx < 0 else 1)
        while not G.is_terminal(b):
            mover = G.to_move(b)
            if arm["side"] == 0 or arm["side"] == mover:
                m = self.net_move(arm["net"], b)
            else:
                m = self.opp_move(arm, g, b)
            moves.append(m)
            b = G.play(b, m)
        w = G.winner(b)
        outcome = 0 if w == G.X else (2 if w == G.O else 1)
        return outcome, theory, len(moves), seq_index(moves)


def seq_index(moves):
    left = list(range(9))
    idx = 0
    for ply, m in enumerate(moves):
        idx = idx * (9 - ply) + left.index(m)
        left.remove(m)
    return SEQ_OFF[len(moves)] + idx


def _seq_off():
    out, acc, perm = [0] * 10, 0, 1
    for L in range(1, 10):
        perm *= 10 - L
        if L >= 5:
            out[L] = acc
            acc += perm
    return out, acc


SEQ_OFF, SEQ_TOTAL = _seq_off()
BITMAP_WORDS = (SEQ_TOTAL + 31) // 32


def all_game_indices():
    """seq indices of every complete game (255,168)."""
    out = []

    def rec(b, moves):
        if G.is_terminal(b):
            out.append(seq_index(moves))
            return
        for m in G.legal(b):
            rec(G.play(b, m), moves + [m])

    rec(G.START, [])
    assert len(out) == 255168 == len(set(out))
    return out


def popcount_bytes(bm):
    return int.from_bytes(bm, "little").bit_count()


# ---- binary -----------------------------------------------------------------------------------------------------
def work():
    d = os.path.join(ROOT, "work")
    os.makedirs(d, exist_ok=True)
    return d


def prepare(binary):
    """tables.bin + one net.bin per net, in ROOT/work."""
    w = work()
    write_tables(os.path.join(w, "tables.bin"))
    for k, net in load_nets().items():
        write_net(net, os.path.join(w, f"net-{k}.bin"))
    return w


def build(cpu):
    src = os.path.join(HERE, "ttt_gpu.cu")
    w = work()
    if cpu:
        exe = os.path.join(w, "ttt_cpu")
        cmd = ["c++", "-O2", "-std=c++17", "-x", "c++", "-o", exe, src]
    else:
        exe = os.path.join(w, "ttt_gpu")
        nvcc = os.environ.get("NVCC") or "/usr/local/cuda/bin/nvcc"
        cmd = [nvcc, "-O3", "-arch=native", "-o", exe, src]
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(src):
        subprocess.run(cmd, check=True)
    return exe


def run_bin(exe, *args, capture=True):
    r = subprocess.run([exe, *map(str, args)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{os.path.basename(exe)} {args[0]} failed rc={r.returncode}: {r.stderr.strip()}")
    return r.stdout


def parity_positions(k):
    """Every to-move position the net can meet: all 4,520 for a both-side net, the 2,423 with X to move for xonly."""
    return [b for b in G.decision_positions() if G.to_move(b) in NET_SIDES[k]]


def parity(exe, w, log=print):
    """GPU (or CPU build) move == net.choose_move on every to-move position of each net's sides (4,520, or 2,423 for
    the X-only net). Raises on any mismatch."""
    out = {}
    for k, net in load_nets().items():
        dec = parity_positions(k)
        codes = os.path.join(w, f"codes-{k}.txt")
        with open(codes, "w") as f:
            f.write("\n".join(str(code3(b)) for b in dec) + "\n")
        got = [int(v) for v in run_bin(exe, "parity", os.path.join(w, "tables.bin"), os.path.join(w, f"net-{k}.bin"),
                                       codes).split()]
        bad = [(b, m) for b, m in zip(dec, got) if m != N.choose_move(net, b)]
        if len(got) != len(dec) or bad:
            raise SystemExit(f"PARITY FAIL {k}: {len(bad)} mismatches, e.g. {bad[:3]}")
        out[k] = len(dec)
        log(f"parity {k}: {len(dec)}/{len(dec)} moves equal net.choose_move "
            f"({'all to-move' if NET_SIDES[k] == (1, 2) else 'X-to-move'} positions)")
    return out


def trace_check(exe, w, n=256, far=1_000_000_000_000, log=print):
    """Per-game (outcome, theory, plies, sequence) of the binary == the pure-Python reference, for every arm, at
    the start of its game range and near 1e12 (both counter words in use)."""
    ref = Reference(load_nets())
    total = 0
    for a in ARMS:
        for start in (0, far + a["arm_id"] * 7919):
            got = run_bin(exe, "trace", os.path.join(w, "tables.bin"), os.path.join(w, f"net-{a['net']}.bin"),
                          *arm_argv(a), start, n).split("\n")
            for i, ln in enumerate(got[:n]):
                want = ref.play(a, start + i)
                if tuple(int(v) for v in ln.split()) != want:
                    raise SystemExit(f"TRACE FAIL {a['name']} game {start + i}: binary {ln!r} reference {want}")
            total += n
    log(f"trace: {total} games over {len(ARMS)} arms equal the Python reference")
    return total


# ---- GPU gate ---------------------------------------------------------------------------------------------------
def gate(holding):
    """None if the GPU may be used now, else why not. Always free here: on a shared GPU, put your own check in."""
    return None


class Lock:
    """Placeholder for a GPU lock shared with other jobs; take() always succeeds here."""

    def take(self):
        return True

    def release(self):
        pass


# ---- power ------------------------------------------------------------------------------------------------------
class Power:
    """nvidia-smi power.draw sampled every 500 ms while the binary runs; trapezoid integral in joules."""

    def __init__(self):
        self.samples = []
        self.p = None

    def __enter__(self):
        self.samples = []
        try:
            self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                                       "-lms", "500"], stdout=subprocess.PIPE, text=True)
        except OSError:
            self.p = None
            return self
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()
        return self

    def _read(self):
        for ln in self.p.stdout:
            try:
                self.samples.append((time.monotonic(), float(ln.strip())))
            except ValueError:
                pass

    def __exit__(self, *exc):
        if self.p:
            self.p.terminate()
            self.p.wait()
            self.t.join(timeout=5)

    def joules(self, baseline=0.0):
        s = self.samples
        return sum((t1 - t0) * ((p0 + p1) / 2 - baseline) for (t0, p0), (t1, p1) in zip(s, s[1:]))

    def mean(self):
        return sum(p for _, p in self.samples) / len(self.samples) if self.samples else None


def idle_power(seconds=6):
    with Power() as pw:
        time.sleep(seconds)
    return pw.mean()


# ---- state ------------------------------------------------------------------------------------------------------
def iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def arm_state(a):
    d = os.path.join(ROOT, "state")
    os.makedirs(d, exist_ok=True)
    st = read_json(os.path.join(d, a["name"] + ".json"))
    if st is None:
        st = {"name": a["name"], "games": 0, "counts": [0] * 9, "plies": 0, "kernel_s": 0.0, "wall_s": 0.0,
              "energy_j": 0.0, "energy_above_idle_j": 0.0, "chunks": 0, "power_samples": 0}
    bm_path = os.path.join(d, a["name"] + ".bm")
    bm = open(bm_path, "rb").read() if os.path.exists(bm_path) else bytes(4 * BITMAP_WORDS)
    return st, bytearray(bm)


def save_arm(a, st, bm):
    d = os.path.join(ROOT, "state")
    with open(os.path.join(d, a["name"] + ".bm.tmp"), "wb") as f:
        f.write(bm)
    os.replace(os.path.join(d, a["name"] + ".bm.tmp"), os.path.join(d, a["name"] + ".bm"))
    write_json(os.path.join(d, a["name"] + ".json"), st)  # counts last: a crash between the two re-plays a chunk


def status(**kw):
    p = os.path.join(ROOT, "status.json")
    st = read_json(p, {})
    st.update(kw)
    st["updatedAt"] = iso()
    st["pid"] = os.getpid()
    write_json(p, st)
    return st


def log(msg):
    line = f"{iso()} {msg}"
    print(line, flush=True)


# ---- results ----------------------------------------------------------------------------------------------------
def arm_result(a, st, bm, valid):
    c = st["counts"]  # theory * 3 + outcome; outcome/theory 0 X wins, 1 draw, 2 O wins
    xw, dr, ow = (sum(c[t * 3 + o] for t in range(3)) for o in range(3))
    bmi = int.from_bytes(bm, "little")
    distinct = (bmi & valid).bit_count()
    invalid = (bmi & ~valid).bit_count()
    r = {"name": a["name"], "group": a["group"], "net": a["net"], "net_file": NETS[a["net"]],
         "side": {1: "X", 2: "O", 0: "both"}[a["side"]], "opponent": a["opp"], "p": a["p"], "open_k": a["open_k"],
         "arm_id": a["arm_id"], "seeds": [SEED0, SEED1], "budget": a["budget"], "games": st["games"],
         "plies": st["plies"], "distinct_games": distinct, "invalid_sequences": invalid,
         "wall_s": round(st["wall_s"], 3), "kernel_s": round(st["kernel_s"], 3),
         "gpu_hours": round(st["wall_s"] / 3600, 4), "energy_j": round(st["energy_j"], 1),
         "energy_above_idle_j": round(st["energy_above_idle_j"], 1), "chunks": st["chunks"],
         "done": st["games"] >= a["budget"]}
    if a["side"]:
        win, loss = (xw, ow) if a["side"] == 1 else (ow, xw)
        r.update(wins=win, draws=dr, losses=loss)
    else:
        r.update(x_wins=xw, draws=dr, o_wins=ow,
                 by_opening_value={name: {"games": sum(c[t * 3:t * 3 + 3]), "x_wins": c[t * 3], "draws": c[t * 3 + 1],
                                          "o_wins": c[t * 3 + 2]}
                                   for t, name in enumerate(("x_wins", "draw", "o_wins"))},
                 lost_from_draw_or_better=c[1 * 3 + 0] + c[1 * 3 + 2] + c[0 * 3 + 2] + c[2 * 3 + 0],
                 won_not_converted=c[0 * 3 + 1] + c[2 * 3 + 1])
    return r


def write_results(valid, extra=None):
    arms = []
    union = 0
    for a in ARMS:
        st, bm = arm_state(a)
        arms.append(arm_result(a, st, bm, valid))
        union |= int.from_bytes(bm, "little")
    tot = {k: sum(r[k] for r in arms) for k in ("games", "plies", "wall_s", "kernel_s", "energy_j",
                                                    "energy_above_idle_j")}
    tot["gpu_hours"] = round(tot["wall_s"] / 3600, 4)
    tot["by_net"] = {}
    for k in NETS:
        rs = [r for r in arms if r["net"] == k and r["group"] in "AB"]
        nt = {x: sum(r[x] for r in rs) for x in ("games", "wins", "draws", "losses", "wall_s", "energy_j",
                                                    "energy_above_idle_j")}
        nt["gpu_hours"] = round(nt["wall_s"] / 3600, 4)
        nt["sides"] = "".join({1: "X", 2: "O"}[s] for s in NET_SIDES[k])
        nt["net_file"] = NETS[k]
        tot["by_net"][k] = nt
    tot["distinct_games_all_arms"] = (union & valid).bit_count()
    for g in "ABC":
        rs = [r for r in arms if r["group"] == g]
        tot[g] = {"games": sum(r["games"] for r in rs)}
        if g != "C":
            for k in ("wins", "draws", "losses"):
                tot[g][k] = sum(r[k] for r in rs)
        else:
            for k in ("x_wins", "draws", "o_wins", "lost_from_draw_or_better", "won_not_converted"):
                tot[g][k] = sum(r[k] for r in rs)
    doc = {"what": "ternary tic-tac-toe nets vs opponents: more games than humanity", "updatedAt": iso(),
           "humanity_estimate": HUMANITY, "target": TARGET, "complete_games_possible": 255168,
           "rng": "Philox4x32-10, counter (game lo, game hi, arm_id, block), key (seed0, seed1)",
           "totals": tot, "arms": arms}
    if extra:
        doc.update(extra)
    write_json(os.path.join(ROOT, "results.json"), doc)
    return doc


# ---- commands ---------------------------------------------------------------------------------------------------
def cmd_check(args):
    exe = build(cpu=True)
    w = prepare(exe)
    parity(exe, w)
    trace_check(exe, w, n=args.n)
    valid = 0
    for i in all_game_indices():
        valid |= 1 << i
    # a small CPU run per group through the counting path
    by = {a["name"]: a for a in ARMS}
    for a in (ARMS[0], ARMS[5], by["C-open3-satxo"], by["A-perfect-xonly-x"], by["B-kid90-xonly-x"]):
        bmf = os.path.join(w, "cpu.bm")
        out = json.loads(run_bin(exe, "run", os.path.join(w, "tables.bin"), os.path.join(w, f"net-{a['net']}.bin"),
                                 *arm_argv(a), 0, 20000, bmf))
        bm = open(bmf, "rb").read()
        bmi = int.from_bytes(bm, "little")
        ref = Reference(load_nets())
        want = [0] * 9
        seen = 0
        for gi in range(20000):
            o, t, _, s = ref.play(a, gi)
            want[t * 3 + o] += 1
            seen |= 1 << s
        assert out["counts"] == want, (a["name"], out["counts"], want)
        assert bmi == seen and not (bmi & ~valid), a["name"]
        print(f"cpu run {a['name']}: 20000 games, counts {out['counts']} = reference, distinct {bmi.bit_count()}")
    print("CHECK PASS")


def session_checks(exe, w):
    p = parity(exe, w, log=log)
    t = trace_check(exe, w, log=log)
    return {"parity_positions": p, "trace_games": t, "at": iso()}


def cmd_bench(args):
    exe = build(cpu=False)
    w = prepare(exe)
    why = gate(False)
    if why:
        raise SystemExit(f"GPU not free: {why}")
    lk = Lock()
    if not lk.take():
        raise SystemExit("GPU busy")
    try:
        session_checks(exe, w)
        by = {a["name"]: a for a in ARMS}
        for a in (ARMS[0], ARMS[1], ARMS[4], ARMS[6], by["C-open3-satxo"], by["A-perfect-xonly-x"],
                  by["B-random-xonly-x"]):
            out = json.loads(run_bin(exe, "run", os.path.join(w, "tables.bin"), os.path.join(w, f"net-{a['net']}.bin"),
                                     *arm_argv(a), 0, args.n, os.path.join(w, "bench.bm")))
            print(f"{a['name']}: {args.n:.3g} games in {out['kernel_ms'] / 1e3:.2f} s kernel / "
                  f"{out['wall_ms'] / 1e3:.2f} s wall = {args.n / out['kernel_ms'] * 1e3:.3g} games/s; {out['counts']}")
    finally:
        lk.release()


def cmd_supervise(args):
    os.makedirs(ROOT, exist_ok=True)
    with open(os.path.join(ROOT, "pid"), "w") as f:
        f.write(f"{os.getpid()}\n")
    for m in ("DONE", "FAILED"):
        if os.path.exists(os.path.join(ROOT, m)):
            os.remove(os.path.join(ROOT, m))
    try:
        _supervise()
    except BaseException as e:  # noqa: BLE001
        with open(os.path.join(ROOT, "FAILED"), "w") as f:
            f.write(f"{iso()} {type(e).__name__}: {e}\n")
        status(state="failed", error=f"{type(e).__name__}: {e}")
        log(f"FAILED {type(e).__name__}: {e}")
        raise


def _supervise():
    exe = build(cpu=False)
    w = prepare(exe)
    valid = 0
    for i in all_game_indices():
        valid |= 1 << i
    started = time.time()
    status(state="starting", startedAt=iso(), target=TARGET, arms=len(ARMS))
    checks = (read_json(os.path.join(ROOT, "results.json"), {}) or {}).get("session_checks", [])  # earlier runs
    lk = Lock()
    rate = None  # games/s of the current arm, for chunk sizing
    while True:
        todo = [a for a in ARMS if arm_state(a)[0]["games"] < a["budget"]]
        if not todo:
            break
        why = gate(False)
        if why or not lk.take():
            why = why or "GPU busy"
            log(f"waiting: {why}")
            status(state="waiting", reason=why, retryInS=YIELD_SLEEP_S)
            time.sleep(YIELD_SLEEP_S)
            continue
        try:
            log("parity + trace checks")
            status(state="checking", reason=None)
            checks.append(session_checks(exe, w))
            base = idle_power()
            log(f"idle power {base} W")
            yielded = None
            for a in todo:
                st, bm = arm_state(a)
                rate = None
                while st["games"] < a["budget"]:
                    why = gate(True)
                    if why:
                        yielded = why
                        break
                    n = min(a["budget"] - st["games"], int(rate * CHUNK_S) if rate else 2_000_000_000)
                    n = max(n, 1)
                    bmf = os.path.join(w, "chunk.bm")
                    with Power() as pw:
                        out = json.loads(run_bin(exe, "run", os.path.join(w, "tables.bin"),
                                                 os.path.join(w, f"net-{a['net']}.bin"), *arm_argv(a),
                                                 st["games"], n, bmf))
                    assert out["start"] == st["games"] and out["n"] == n and sum(out["counts"]) == n
                    chunk_bm = open(bmf, "rb").read()
                    bm = bytearray((int.from_bytes(bm, "little") | int.from_bytes(chunk_bm, "little"))
                                   .to_bytes(4 * BITMAP_WORDS, "little"))
                    st["counts"] = [x + y for x, y in zip(st["counts"], out["counts"])]
                    st["plies"] += out["plies"]
                    st["games"] += n
                    st["kernel_s"] += out["kernel_ms"] / 1e3
                    st["wall_s"] += out["wall_ms"] / 1e3
                    st["energy_j"] += pw.joules()
                    if base is not None:
                        st["energy_above_idle_j"] += pw.joules(base)
                    st["power_samples"] += len(pw.samples)
                    st["chunks"] += 1
                    save_arm(a, st, bm)
                    rate = n / max(out["wall_ms"] / 1e3, 1e-3)
                    total = sum(arm_state(x)[0]["games"] for x in ARMS)
                    status(state="running", reason=None, arm=a["name"], armGames=st["games"], armBudget=a["budget"],
                           gamesTotal=total, gamesPerS=round(rate), lastPowerW=pw.mean(), idlePowerW=base,
                           elapsedS=round(time.time() - started))
                if yielded:
                    break
                log(f"arm done {a['name']}: {st['games']} games {st['counts']}")
                write_results(valid, {"session_checks": checks})
        finally:
            lk.release()
        if yielded:
            log(f"yielded the GPU: {yielded}")
            status(state="yielded", reason=yielded, retryInS=YIELD_SLEEP_S)
            time.sleep(YIELD_SLEEP_S)
    doc = write_results(valid, {"session_checks": checks})
    status(state="done", reason=None, gamesTotal=doc["totals"]["games"], finishedAt=iso())
    with open(os.path.join(ROOT, "DONE"), "w") as f:
        f.write(f"{iso()} games {doc['totals']['games']}\n")
    log(f"DONE {doc['totals']['games']} games")


def cmd_status(args):
    print(json.dumps(read_json(os.path.join(ROOT, "status.json"), {}), indent=1))


def fmt(n):
    return f"{n:,}"


def cmd_report(args):
    doc = read_json(args.results or os.path.join(ROOT, "results.json"))
    if doc is None:
        raise SystemExit("no results.json")
    t = doc["totals"]
    L = ["# Tic-tac-toe: more games than humanity", "",
         f"Updated {doc['updatedAt']}. Code: `tictactoe/humanity/`. Data: `results/humanity.json`.", "",
         f"Humanity's games of tic-tac-toe, order of magnitude: {doc['humanity_estimate']:.0e}. "
         f"Games played here: **{fmt(t['games'])}** ({t['games'] / doc['humanity_estimate']:.2f}x), "
         f"{fmt(t['plies'])} moves, {t['gpu_hours']:.3f} GPU-hours on one NVIDIA GB10, "
         f"{t['energy_j'] / 3.6e6:.3f} kWh measured at the GPU ({t['energy_above_idle_j'] / 3.6e6:.3f} kWh above idle). "
         f"Distinct complete games seen: {fmt(t['distinct_games_all_arms'])} of 255,168.", "",
         f"| group | games | net wins | draws | net losses |", "|---|---:|---:|---:|---:|"]
    for g in "AB":
        x = t[g]
        L.append(f"| {g} | {fmt(x['games'])} | {fmt(x['wins'])} | {fmt(x['draws'])} | {fmt(x['losses'])} |")
    L += ["", "| net | plays | games | wins | draws | losses | GPU-h | kWh |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for k, x in t.get("by_net", {}).items():
        L.append(f"| {k} (`{x['net_file']}`) | {x['sides']} | {fmt(x['games'])} | {fmt(x['wins'])} | "
                 f"{fmt(x['draws'])} | {fmt(x['losses'])} | {x['gpu_hours']:.3f} | {x['energy_j'] / 3.6e6:.4f} |")
    L += ["", "Every count is exact. At the start of every GPU session the GPU move function equalled "
          "`net.choose_move` on all 4,520 to-move positions for finisher and satxo, and on all 2,423 X-to-move "
          "positions for xonly (no symmetry, plays X only), and per-game traces equalled a pure-Python reference "
          "player (same Philox streams) for every arm.", "",
          "## A: net vs a perfect opponent (uniform over value-keeping moves)", "",
          "| arm | games | wins | draws | losses | distinct games | GPU-h |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in doc["arms"]:
        if r["group"] == "A":
            L.append(f"| {r['name']} | {fmt(r['games'])} | {fmt(r['wins'])} | {fmt(r['draws'])} | "
                     f"{fmt(r['losses'])} | {fmt(r['distinct_games'])} | {r['gpu_hours']:.3f} |")
    L += ["", "## B: net vs human-like opponents", "",
          "random = uniform over empty cells; kidP = takes a win with probability P, else blocks with probability P, "
          "else random; mixQ = a perfect move with probability Q, else a random one.", "",
          "| arm | games | wins | draws | losses | distinct games | GPU-h |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in doc["arms"]:
        if r["group"] == "B":
            L.append(f"| {r['name']} | {fmt(r['games'])} | {fmt(r['wins'])} | {fmt(r['draws'])} | "
                     f"{fmt(r['losses'])} | {fmt(r['distinct_games'])} | {r['gpu_hours']:.3f} |")
    L += ["", "## C: random opening, then the net plays both sides", "",
          "Not part of the zero-loss claim. 'opening value' is the minimax value after the random moves. "
          "'lost from a draw or better' counts games where the losing side's position was at least a draw after "
          "the opening; 'won, not converted' counts won openings that ended in a draw.", "",
          "| arm | games | X wins | draws | O wins | lost from a draw or better | won, not converted | distinct |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in doc["arms"]:
        if r["group"] == "C":
            L.append(f"| {r['name']} | {fmt(r['games'])} | {fmt(r['x_wins'])} | {fmt(r['draws'])} | "
                     f"{fmt(r['o_wins'])} | {fmt(r['lost_from_draw_or_better'])} | {fmt(r['won_not_converted'])} | "
                     f"{fmt(r['distinct_games'])} |")
    L += ["", "By opening value (X wins / draws / O wins):", ""]
    for r in doc["arms"]:
        if r["group"] == "C":
            bo = r["by_opening_value"]
            L.append(f"- {r['name']}: " + "; ".join(
                f"opening {k.replace('_', ' ')} {fmt(v['games'])} -> {fmt(v['x_wins'])} / {fmt(v['draws'])} / "
                f"{fmt(v['o_wins'])}" for k, v in bo.items()))
    L += ["", f"RNG: {doc['rng']}; seeds {SEED0}, {SEED1}; arm ids in the JSON. Energy: nvidia-smi power.draw "
          "every 500 ms while the binary ran (the whole GPU, resident services included); 'above idle' subtracts "
          "the mean draw measured for 6 s before each session.", ""]
    out = args.out or os.path.join(TTT, "results", "humanity.md")
    with open(out, "w") as f:
        f.write("\n".join(L))
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("check")
    p.add_argument("--n", type=int, default=64)
    p = sp.add_parser("bench")
    p.add_argument("--n", type=int, default=1 << 32)
    sp.add_parser("supervise")
    sp.add_parser("status")
    p = sp.add_parser("report")
    p.add_argument("results", nargs="?")
    p.add_argument("--out")
    a = ap.parse_args()
    {"check": cmd_check, "bench": cmd_bench, "supervise": cmd_supervise, "status": cmd_status,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()

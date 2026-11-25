"""Bit-level serialized size of the saved nets: an actual encoder, a decoder, and a round-trip check.

Format (a bit string, most significant bit first):

    header   1 bit encoding (0 trits, 1 planes), 1 bit symmetry, 8 bits H, 4 bits w1, 4 bits w2, 1 bit weight mode
    weights  W1 (H x n_in) then W2 (9 x H), row-major, in one of two modes, whichever is shorter:
               dense   5 trits per byte (3^5 = 243 <= 256), so ceil(n / 5) bytes
               sparse  one bit per slot (nonzero or not), then one sign bit per nonzero weight
    b1       H integers, two's complement, w1 bits each (w1 = 0 when every b1 is 0)
    b2       9 integers after subtracting their minimum (the argmax ignores a common shift), unsigned, w2 bits each

The decoder rebuilds the net and the check compares its move with the saved net's on all 4,520 positions. The
information bound of the weights alone, ceil(n log2 3) bits for n dense slots, is reported for comparison. The
canonicalisation tables (7 non-identity permutations of 9 cells, 4 bits per entry, and the 8-by-9 map back) are
reported separately; they are part of the player, not of any one net.

    python3 serial_size.py        -> results/serial-size.json, and a table on stdout
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import game as G  # noqa: E402
import net as N  # noqa: E402

NETS = [("85", "smallest-never-loses-h4.json"), ("104", "smallest-never-loses-h5.json"),
        ("180", "smallest-never-loses-params.json"), ("199", "smallest-never-loses.json"),
        ("560", "smallest-never-loses-nosym.json"), ("123", "smallest-never-loses-nosym-sat.json"), ("142", "smallest-never-loses-nosym-sat-h7.json"), ("275", "smallest-finisher.json"),
        ("332", "smallest-converter.json"), ("389", "smallest-optimal-trits.json"), ("457", "smallest-optimal.json"), ("2346", "smallest-optimal-nosym.json"),
        ("47-x-only", "smallest-x-only.json")]


class Bits:
    def __init__(self, s=""):
        self.s, self.i = s, 0

    def put(self, v, w):
        if w:
            assert 0 <= v < (1 << w), (v, w)
            self.s += format(v, f"0{w}b")

    def get(self, w):
        if not w:
            return 0
        v = int(self.s[self.i:self.i + w], 2)
        self.i += w
        return v


def twos_width(vals):
    w = 0
    while not all(-(1 << (w - 1)) <= v < (1 << (w - 1)) for v in vals) if w else any(vals):
        w += 1
    return w


def encode(net, mode):
    (L1, L2) = net["layers"]
    H, n_in = len(L1["b"]), len(L1["W"][0])
    w = [v for r in L1["W"] for v in r] + [v for r in L2["W"] for v in r]
    b1 = L1["b"]
    m = min(L2["b"])
    b2 = [v - m for v in L2["b"]]
    w1, w2 = twos_width(b1), max(b2).bit_length()
    B = Bits()
    B.put(int(net["encoding"] == "planes"), 1)
    B.put(int(bool(net.get("symmetry"))), 1)
    B.put(H, 8)
    B.put(w1, 4)
    B.put(w2, 4)
    B.put(int(mode == "sparse"), 1)
    if mode == "dense":
        for k in range(0, len(w), 5):
            chunk = w[k:k + 5]
            B.put(sum((v + 1) * 3 ** t for t, v in enumerate(chunk)), 8)
    else:
        for v in w:
            B.put(int(v != 0), 1)
        for v in w:
            if v:
                B.put(int(v < 0), 1)
    for v in b1:
        B.put(v & ((1 << w1) - 1), w1)
    for v in b2:
        B.put(v, w2)
    assert n_in == (18 if net["encoding"] == "planes" else 9)
    return B.s


def decode(s, act="relu"):
    B = Bits(s)
    planes, sym, H, w1, w2, sparse = B.get(1), B.get(1), B.get(8), B.get(4), B.get(4), B.get(1)
    n_in = 18 if planes else 9
    n = H * n_in + 9 * H
    if not sparse:
        w = []
        while len(w) < n:
            c = B.get(8)
            for _ in range(min(5, n - len(w))):
                w.append(c % 3 - 1)
                c //= 3
    else:
        nzmask = [B.get(1) for _ in range(n)]
        w = [(-1 if B.get(1) else 1) if z else 0 for z in nzmask]
    b1 = []
    for _ in range(H):
        v = B.get(w1)
        b1.append(v - (1 << w1) if w1 and v >= 1 << (w1 - 1) else v)
    b2 = [B.get(w2) for _ in range(9)]
    assert B.i == len(s), "trailing bits"
    W1 = [w[j * n_in:(j + 1) * n_in] for j in range(H)]
    off = H * n_in
    W2 = [w[off + k * H: off + (k + 1) * H] for k in range(9)]
    return {"format": N.FORMAT, "encoding": "planes" if planes else "trits", "symmetry": bool(sym), "act": act,
            "layers": [{"W": W1, "b": b1}, {"W": W2, "b": b2}]}


def main():
    boards = G.decision_positions()
    out = {}
    for key, fn in NETS:
        net = N.load(os.path.join(HERE, "weights", fn))
        assert net["act"] == "relu" and len(net["layers"]) == 2
        sz = N.size(net)
        res = {"file": fn, "params": sz["params"], "nonzero": sz["nonzero"], "weight_slots": sz["weights"],
               "json_bytes": os.path.getsize(os.path.join(HERE, "weights", fn))}
        for mode in ("dense", "sparse"):
            s = encode(net, mode)
            d = decode(s)
            same = all(N.choose_move(d, b) == N.choose_move(net, b) for b in boards)
            assert same, (key, mode)
            res[f"{mode}_bits"] = len(s)
        res["weights_info_bound_bits"] = math.ceil(sz["weights"] * math.log2(3))
        best = min(("dense", "sparse"), key=lambda m: res[f"{m}_bits"])
        res["mode"], res["bits"] = best, res[f"{best}_bits"]
        res["bytes"] = math.ceil(res["bits"] / 8)
        res["roundtrip_positions"] = len(boards)
        out[key] = res
        print(f"{key:10s} params {sz['params']:5d} nz {sz['nonzero']:5d}  dense {res['dense_bits']:6d} b  "
              f"sparse {res['sparse_bits']:6d} b  -> {res['bits']} bits = {res['bytes']} bytes "
              f"(info bound of weights {res['weights_info_bound_bits']} b)")
    canon = {"forward_permutations_bits": 7 * 9 * 4, "map_back_bits": 8 * 9 * 4}
    canon["total_bits"] = canon["forward_permutations_bits"] + canon["map_back_bits"]
    print("canonicalisation tables:", canon)
    with open(os.path.join(HERE, "results", "serial-size.json"), "w") as f:
        json.dump({"what": __doc__.split("\n")[0], "positions": len(boards), "nets": out, "canonicalisation": canon},
                  f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()

"""Markdown table of the exact-search results (results/minsat.jsonl, merged from the prove.py run dirs).

    python3 minsat_table.py results/minsat.jsonl
    python3 minsat_table.py --merge runs run1 run2 run3 > /dev/null   merge prove.py run dirs (runs/run1, ...) first

Jobs stopped by hand because they had become moot (a smaller width had already been found SAT) are left out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "minsat.jsonl")
KEEP = ("name", "status", "hidden", "sides", "sym", "cond", "bias", "b1", "b2", "engine", "enc", "full",
        "min_nonzero", "max_nonzero", "require_used", "nonzero", "optimal", "rounds", "subset", "positions",
        "seconds", "wall", "vars", "clauses", "verified", "as_x", "as_o", "size", "last_round", "finishedAt")


def fetch(remote, dirs):
    base = remote
    rows = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["name"]] = r
    for d in dirs:
        with open(os.path.join(base, d, "results.jsonl")) as f:
            txt = f.read()
        for line in txt.splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["status"] == "ERROR" and r.get("returncode") == -15:
                continue  # stopped by hand as moot
            rows[r["name"]] = {k: r[k] for k in KEEP if k in r}
    with open(OUT, "w") as f:
        for r in sorted(rows.values(), key=lambda r: r["name"]):
            f.write(json.dumps(r) + "\n")
    return list(rows.values())


def table(rows):
    out = ["| job | sides | symmetry | H | biases | extra | result | nonzero | positions used | time |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r.get("sides", ""), not r.get("sym", False), r.get("hidden", 0), r["name"])):
        bias = "general (any integer)" if r.get("bias") == "general" else f"b1 in ±{r.get('b1')}, b2 spread ≤ {2 * r.get('b2', 0)}"
        extra = []
        if r.get("engine") == "cpsat":
            extra.append("CP-SAT")
        if r.get("enc") == "unary":
            extra.append("unary")
        if r.get("full"):
            extra.append("all positions")
        if r.get("max_nonzero") is not None:
            extra.append(f"nonzero ≤ {r['max_nonzero']}")
        if r.get("min_nonzero"):
            extra.append("minimise nonzero")
        if r.get("require_used"):
            extra.append("every unit used")
        st = r["status"]
        if st == "SAT" and r.get("min_nonzero"):
            st = "SAT, minimum" if r.get("optimal") else "SAT"
        if st in ("TIMEOUT", "ERROR", "UNKNOWN"):
            st = f"open ({st.lower()} after {r.get('wall', r.get('seconds', 0)) / 3600:.1f} h)"
        out.append(f"| {r['name']} | {r.get('sides')} | {'yes' if r.get('sym') else 'no'} | {r.get('hidden')} | {bias} | "
                   f"{', '.join(extra)} | {st} | {r.get('nonzero', '')} | "
                   f"{r.get('subset', '')}{'/' + str(r['positions']) if r.get('positions') else ''} | "
                   f"{r.get('seconds', r.get('wall', 0)):.0f} s |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=OUT)
    ap.add_argument("--merge", nargs="+", metavar=("DIR", "RUN"))
    a = ap.parse_args()
    if a.merge:
        rows = fetch(a.merge[0], a.merge[1:])
    else:
        with open(a.path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())

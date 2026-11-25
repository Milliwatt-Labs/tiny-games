#!/usr/bin/env python3
"""Build the playable page: page/template.html + page/copy.json + weights -> page/index.html (one static file).

    python3 page_build.py           write page/index.html
    python3 page_build.py --check   exit 1 if page/index.html is out of date

The five nets are embedded as JSON (only the fields the forward pass reads), and each selector label is copy.json
"net.name" filled from that net's weight file (size.params, then size.nonzero), so the labels cannot drift from the
weights. tests/test_page.py proves the page's
chooseMove matches net.choose_move on every position each net can face (all 4,520; the X-only net
the X-to-move ones).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import net as N  # noqa: E402

NETS = {"xonly": "smallest-x-only.json", "never": "smallest-never-loses-h4.json", "finisher": "smallest-finisher.json",
        "converter": "smallest-converter.json", "optimal": "smallest-optimal.json"}
KEEP = ("format", "encoding", "symmetry", "act", "layers", "x_only")  # x_only: the net only plays X (you play O)
PAGE = os.path.join(HERE, "page")


def embedded():
    out = {}
    for k, f in NETS.items():
        net = N.load(os.path.join(HERE, "weights", f))
        out[k] = {f: net[f] for f in KEEP if f in net}
    return out


def labels(fmt):
    """Selector label per net, total parameters first, nonzero weights second: {"net.xonly": "47 (10)", ...}."""
    out = {}
    for k, f in NETS.items():
        size = N.load(os.path.join(HERE, "weights", f))["size"]
        out[f"net.{k}"] = fmt.format(params=size["params"], nonzero=size["nonzero"])
    return out


def build():
    with open(os.path.join(PAGE, "template.html")) as f:
        html = f.read()
    with open(os.path.join(PAGE, "copy.json")) as f:
        copy = json.load(f)
    copy.update(labels(copy.pop("net.name")))
    nets = json.dumps(embedded(), separators=(",", ":"))
    cp = json.dumps(copy, separators=(",", ":"), ensure_ascii=False)
    assert "</" not in nets and "</" not in cp
    return html.replace("__NETS__", nets).replace("__COPY__", cp)


def main():
    out = os.path.join(PAGE, "index.html")
    html = build()
    if "--check" in sys.argv:
        with open(out) as f:
            ok = f.read() == html
        print("page/index.html up to date" if ok else "page/index.html is stale: run page_build.py")
        sys.exit(0 if ok else 1)
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()

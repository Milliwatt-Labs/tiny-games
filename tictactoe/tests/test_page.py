#!/usr/bin/env python3
"""The playable page (page/index.html) plays exactly like net.choose_move.

Runs the page's own engine script in Node on every position each embedded net can face (all 4,520 positions with a
move to make; for the X-only net, which always plays X, the X-to-move ones) and checks every move against the Python
integer reference. Also checks the embedded weights are the weight files and
that page/index.html is the current build of page/template.html.

    python3 -m pytest tictactoe/tests/test_page.py -q
    TTT_PAGE=<file> python3 -m pytest tictactoe/tests/test_page.py -q -k every_position   # a served copy
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import game as G  # noqa: E402
import net as N  # noqa: E402
import page_build as P  # noqa: E402

PAGE = os.path.join(ROOT, "page", "index.html")
# TTT_PAGE: run the parity test on another copy (for example a served copy).
SERVED = os.environ.get("TTT_PAGE") or PAGE


def _node():
    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not on PATH")
    return node


def test_page_is_current_build():
    with open(PAGE) as f:
        assert f.read() == P.build(), "page/index.html is stale: run page_build.py"


def test_embedded_weights_are_the_files():
    with open(PAGE) as f:
        html = f.read()
    start = html.index('<script type="application/json" id="nets">') + len('<script type="application/json" id="nets">')
    nets = json.loads(html[start:html.index("</script>", start)])
    assert set(nets) == set(P.NETS)
    for k, fname in P.NETS.items():
        ref = N.load(os.path.join(ROOT, "weights", fname))
        for f in ("encoding", "symmetry", "act", "layers"):
            assert nets[k][f] == ref[f], (k, f)
    assert N.size(nets["finisher"] | {"format": N.FORMAT})["nonzero"] == 141
    assert N.size(nets["converter"] | {"format": N.FORMAT})["nonzero"] == 204
    assert N.size(nets["optimal"] | {"format": N.FORMAT})["nonzero"] == 286
    assert N.size(nets["never"] | {"format": N.FORMAT})["nonzero"] == 39
    assert N.size(nets["xonly"] | {"format": N.FORMAT})["nonzero"] == 10
    # the page locks you to O for the X-only net and only for it
    assert [k for k in nets if nets[k].get("x_only")] == ["xonly"]


def test_selector_labels_are_the_counts():
    """Each net is named by total parameters first and nonzero weights second, as counted from the embedded net."""
    with open(PAGE) as f:
        html = f.read()
    start = html.index('<script type="application/json" id="copy">') + len('<script type="application/json" id="copy">')
    copy = json.loads(html[start:html.index("</script>", start)])
    start = html.index('<script type="application/json" id="nets">') + len('<script type="application/json" id="nets">')
    nets = json.loads(html[start:html.index("</script>", start)])
    for k, net in nets.items():
        s = N.size(net | {"format": N.FORMAT})
        assert copy[f"net.{k}"] == f"{s['params']} ({s['nonzero']})", k
    assert [copy[f"net.{k}"] for k in P.NETS] == ["47 (10)", "85 (39)", "275 (141)", "332 (204)", "457 (286)"]
    assert "net.name" not in copy


def test_page_nets_never_lose():
    for k, fname in P.NETS.items():
        ref = N.load(os.path.join(ROOT, "weights", fname))
        res = N.verify(ref)
        assert res["as_x"]["loss"] == 0, k
        assert ref.get("x_only") or res["as_o"]["loss"] == 0, k


def _faced(net, boards):
    """Positions the page can ask this net to move in: all of them, or only X-to-move for an X-only net."""
    return [b for b in boards if G.to_move(b) == G.X] if net.get("x_only") else list(boards)


def test_page_moves_match_python_on_every_position():
    node = _node()
    boards = G.decision_positions()
    assert len(boards) == 4520
    scratch = None
    nets = {k: N.load(os.path.join(ROOT, "weights", f)) for k, f in P.NETS.items()}
    faced = {k: _faced(net, boards) for k, net in nets.items()}
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        bp = os.path.join(d, "boards.json")
        with open(bp, "w") as f:
            json.dump({k: [list(b) for b in v] for k, v in faced.items()}, f)
        out = subprocess.run([node, os.path.join(HERE, "page_parity.mjs"), SERVED, bp],
                             check=True, capture_output=True, text=True).stdout
    got = json.loads(out)
    n_x = sum(1 for b in boards if G.to_move(b) == G.X)
    for k, net in nets.items():
        want = [N.choose_move(net, b) for b in faced[k]]
        bad = [(G.show(b), g, w) for b, g, w in zip(faced[k], got[k], want) if g != w]
        assert not bad, (k, len(bad), bad[:3])
        assert len(got[k]) == len(faced[k]) == (n_x if net.get("x_only") else 4520), k


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e)[:400])
    sys.exit(1 if fails else 0)

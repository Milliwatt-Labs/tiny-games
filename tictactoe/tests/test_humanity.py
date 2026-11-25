#!/usr/bin/env python3
"""The GPU player's CPU build (humanity/ttt_gpu.cu compiled with c++) against net.py / game.py: parity on all 4,520
to-move positions for finisher and satxo and on the 2,423 X-to-move positions for the X-only net, per-game traces against the pure-Python reference player for every arm, and the
counting path. Needs a C++ compiler; skipped without one.

    python3 tictactoe/tests/test_humanity.py     PASS/FAIL with the time, exits 1 on failure
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "humanity", "humanity.py")


def test_humanity_cpu_check():
    if not shutil.which("c++"):
        print("SKIP no c++")
        return
    root = tempfile.mkdtemp()
    env = dict(os.environ, TTT_HUMANITY_ROOT=os.path.join(root, "humanity-test"))
    r = subprocess.run([sys.executable, SCRIPT, "check", "--n", "32"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "CHECK PASS" in r.stdout, r.stdout + r.stderr


if __name__ == "__main__":
    t = time.time()
    try:
        test_humanity_cpu_check()
        print(f"PASS test_humanity_cpu_check {time.time() - t:.1f} s")
    except AssertionError as e:
        print(f"FAIL test_humanity_cpu_check {time.time() - t:.1f} s\n{e}")
        sys.exit(1)

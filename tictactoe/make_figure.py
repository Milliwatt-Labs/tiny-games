#!/usr/bin/env python3
"""Draw figures/x-only-diagram.png (Figure 1) from weights/smallest-x-only.json. Needs matplotlib.

    python3 make_figure.py

The wiring, the base scores and the formula for each hidden unit are read from the weight file; the two lines of
description under the diagram are derived from the signs, so they cannot disagree with the weights.
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TTT = HERE
GREEN, RED, GREY, INK = "#2e9e5b", "#d64545", "#777777", "#222222"


def formula(w, b):
    s = str(b)
    for i, v in enumerate(w):
        if v:
            s += f" {'+' if v > 0 else '−'} c{i}"
    return f"ReLU({s.replace('-', '−')})"


def cells(side, idx):
    if not idx:
        return []
    return [f"{side} at {', '.join(map(str, idx[:-1])) + ' or ' if len(idx) > 1 else ''}{idx[-1]}"]


def describe(j, w, W2):
    pos = [i for i, v in enumerate(w) if v > 0]
    neg = [i for i, v in enumerate(w) if v < 0]
    up, down = cells("X", pos) + cells("O", neg), cells("X", neg) + cells("O", pos)
    adds = [str(k) for k in range(9) if W2[k][j] > 0]
    subs = [str(k) for k in range(9) if W2[k][j] < 0]
    out = f"h{j + 1} rises with {' and with '.join(up)}, falls with {' and with '.join(down)}"
    if adds:
        out += f", adds to cell{'s' if len(adds) > 1 else ''} {', '.join(adds)}"
    if subs:
        out += f", subtracts from cell{'s' if len(subs) > 1 else ''} {', '.join(subs)}"
    return out + "."


def main():
    net = json.load(open(os.path.join(TTT, "weights", "smallest-x-only.json")))
    (L1, L2) = net["layers"]
    W1, b1, W2, b2 = L1["W"], L1["b"], L2["W"], L2["b"]
    H = len(b1)
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 5.6)
    ax.axis("off")
    cell = 0.62
    bx, by = 0.6, 1.2  # board lower-left
    ox = 7.5

    def centre(x0, i):
        r, c = divmod(i, 3)
        return x0 + (c + 0.5) * cell, by + (2 - r + 0.5) * cell

    for x0, title in ((bx, "Board in"), (ox, "Move scores")):
        for i in range(9):
            r, c = divmod(i, 3)
            ax.add_patch(plt.Rectangle((x0 + c * cell, by + (2 - r) * cell), cell, cell, fc="#f3f3f3", ec="#b0b0b0",
                                       lw=1.5))
        ax.text(x0 + 1.5 * cell, by + 3 * cell + 0.25, title, ha="center", fontsize=14, weight="bold", color=INK)
    hy = [3.75, 1.85][:H] if H == 2 else [3.75 - k * 1.9 / max(1, H - 1) for k in range(H)]
    hx = 5.0
    for j in range(H):
        for i, v in enumerate(W1[j]):
            if v:
                x, y = centre(bx, i)
                ax.plot([x, hx], [y, hy[j]], color=GREEN if v > 0 else RED, lw=2.6, alpha=0.85, zorder=1)
                ax.plot(x, y, "o", color=GREEN if v > 0 else RED, ms=7, zorder=3)
        for k in range(9):
            v = W2[k][j]
            if v:
                x, y = centre(ox, k)
                ax.plot([hx, x], [hy[j], y], color=GREEN if v > 0 else RED, lw=2.6, alpha=0.85, zorder=1)
        ax.add_patch(plt.Circle((hx, hy[j]), 0.3, fc="white", ec=INK, lw=2, zorder=4))
        ax.text(hx, hy[j], f"h{j + 1}", ha="center", va="center", fontsize=13, weight="bold", zorder=5)
        ax.text(hx, hy[j] + (0.45 if j == 0 else -0.55), formula(W1[j], b1[j]), ha="center", fontsize=10.5,
                color=GREY)
    for i in range(9):
        x, y = centre(bx, i)
        ax.text(x + cell / 2 - 0.08, y + cell / 2 - 0.1, str(i), ha="right", va="top", fontsize=9, color=GREY)
        x, y = centre(ox, i)
        ax.add_patch(plt.Circle((x, y), 0.19, fc="white", ec="none", zorder=4))
        ax.text(x, y, str(b2[i]), ha="center", va="center", fontsize=14, weight="bold", zorder=5, color=INK)
    ax.text(bx + 1.5 * cell, by - 0.3, "cell value: +1 X, −1 O, 0 empty", ha="center", fontsize=9.5, color=GREY)
    ax.text(ox + 1.5 * cell, by - 0.3, "output bias; lines add or subtract h1, h2", ha="center", fontsize=9.5,
            color=GREY)
    for j in range(H):
        ax.text(5.0, 0.35 - 0.35 * j, describe(j, W1[j], W2), ha="center", fontsize=10, color=INK)
    ax.plot([0.3, 0.7], [5.3, 5.3], color=GREEN, lw=3)
    ax.text(0.8, 5.3, "+1", va="center", fontsize=10, color=GREY)
    ax.plot([1.3, 1.7], [5.3, 5.3], color=RED, lw=3)
    ax.text(1.8, 5.3, "−1", va="center", fontsize=10, color=GREY)
    out = os.path.join(HERE, "figures", "x-only-diagram.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)
    for j in range(H):
        print(describe(j, W1[j], W2))


if __name__ == "__main__":
    main()

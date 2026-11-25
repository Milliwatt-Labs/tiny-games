"""Tic-tac-toe: exact solver over every legal position, symmetries and input encodings.

A board is a tuple of 9 ints, row-major (cell 0 = top left): 0 empty, 1 X, 2 O. X moves first, so the side to
move is fixed by the piece counts. Every position reachable from the empty board under legal play (stopping at a
win or a full board) is enumerated: 5,478 positions, 4,520 of them with a move to make.

Values are from the point of view of the side to move: +1 win, 0 draw, -1 loss under perfect play by both sides.
A move is *optimal* when it keeps the position's value. Pure Python, no dependencies.
"""
from __future__ import annotations

from functools import lru_cache

EMPTY, X, O = 0, 1, 2
LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))
START = (EMPTY,) * 9


def winner(b):
    """1 or 2 if that side has three in a row, else 0."""
    for i, j, k in LINES:
        if b[i] and b[i] == b[j] == b[k]:
            return b[i]
    return 0


def is_terminal(b):
    return winner(b) != 0 or all(b)


def to_move(b):
    nx, no = b.count(X), b.count(O)
    return X if nx == no else O


def legal(b):
    return [i for i in range(9) if b[i] == EMPTY]


def play(b, m):
    assert b[m] == EMPTY
    s = list(b)
    s[m] = to_move(b)
    return tuple(s)


@lru_cache(maxsize=None)
def value(b):
    """Minimax value for the side to move: +1 win, 0 draw, -1 loss."""
    w = winner(b)
    if w:
        return -1  # the previous mover just won
    if all(b):
        return 0
    return max(-value(play(b, m)) for m in legal(b))


def optimal_moves(b):
    """Every move that keeps the minimax value of a non-terminal position."""
    v = value(b)
    return [m for m in legal(b) if -value(play(b, m)) == v]


def winning_moves(b):
    """Empty cells that win at once for the side to move (three in a row with this move)."""
    return [m for m in legal(b) if winner(play(b, m))]


def all_positions():
    """Every position reachable from the empty board, in a fixed (DFS discovery) order."""
    seen = {}
    stack = [START]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen[b] = len(seen)
        if not is_terminal(b):
            for m in reversed(legal(b)):
                c = play(b, m)
                if c not in seen:
                    stack.append(c)
    return list(seen)


def decision_positions():
    """Non-terminal positions (someone has to move), sorted for a stable order."""
    return sorted(b for b in all_positions() if not is_terminal(b))


# ---- symmetries -------------------------------------------------------------------------------------------------
# A symmetry is a permutation p of the cells: the transformed board t has t[i] = b[p[i]], so a move at cell i of t is
# the move p[i] of b.
def _rot(p):  # rotate 90 degrees clockwise
    return tuple(p[6 - 3 * (i % 3) + i // 3] for i in range(9))


def _flip(p):  # mirror left-right
    return tuple(p[3 * (i // 3) + 2 - i % 3] for i in range(9))


def _symmetries():
    out = []
    p = tuple(range(9))
    for _ in range(4):
        out.append(p)
        out.append(_flip(p))
        p = _rot(p)
    assert len(set(out)) == 8
    return tuple(out)


SYMS = _symmetries()


def transform(b, p):
    return tuple(b[p[i]] for i in range(9))


def canonical(b):
    """(canonical board, permutation): the lexicographically smallest of the 8 images of b, and the p that makes it.

    A move m on the canonical board is the move p[m] on b."""
    best = None
    for p in SYMS:
        t = transform(b, p)
        if best is None or t < best[0]:
            best = (t, p)
    return best


# ---- input encodings --------------------------------------------------------------------------------------------
ENCODINGS = {"planes": 18, "trits": 9}


def encode(b, encoding):
    """Relative to the side to move, so one net plays both X and O.

    planes: 18 inputs in {0,1}: 9 cells holding the mover's pieces, then 9 holding the opponent's.
    trits:  9 inputs in {-1,0,+1}: +1 mover, -1 opponent, 0 empty."""
    me = to_move(b)
    if encoding == "planes":
        return [1 if c == me else 0 for c in b] + [1 if c and c != me else 0 for c in b]
    if encoding == "trits":
        return [0 if c == EMPTY else (1 if c == me else -1) for c in b]
    raise ValueError(f"unknown encoding {encoding!r}")


def show(b):
    ch = ".XO"
    return "\n".join("".join(ch[b[3 * r + c]] for c in range(3)) for r in range(3))

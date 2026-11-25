"""CNF encoding of "a 1-hidden-layer ternary ReLU net never loses" for a SAT solver (pysat, CaDiCaL 1.9.5).

Same model class, conditions and bias modes as minsat.py (read its docstring first); this is the fast engine.
The net's integer forward pass on a fixed board is a small circuit over the weight bits, written out as binary
arithmetic with full-equivalence Tseitin gates:

    W1[j][i], W2[k][j] in {-1, 0, +1}     two booleans p, n with not (p and n); value p - n
    b1 = l1 + B1, b2 = B2                  B1, B2 unsigned binary (b2's common shift fixed: min b2 = 0)
    z_j = sum_i x_i W1[j][i] + b1_j        x_i W = a_i - c_i with a, c the p/n literal picked by the sign of x_i;
                                           sum_i (a_i + not c_i) is a popcount, so z_j + n_occ - l1 = popcount + B1
    h_j = relu(z_j)                        compare with a constant, subtract it, mask
    Y_k = sum_j W2[k][j] h_j + b2_k        as  sum_j (p_kj and h_j) + B2_k + NOT(sum_j (n_kj and h_j)), a common
                                           offset for every k, so comparisons between cells are exact
    s beats m (s < m)                      Y_s >= Y_m (lowest index wins ties); m beats s is its negation
    argmax_s                               AND over the other legal m of "s beats m"

Condition `all`: OR over allowed s of argmax_s. Condition `reach`: reach_p -> OR over allowed s of argmax_s, and
reach_p and argmax_s -> reach_q for every net-to-move q one opponent reply later; roots are units.

Positions are added incrementally (CEGAR), so the solver keeps what it learned between rounds.
"""
from __future__ import annotations

import threading
import time

import net as N


class Circuit:
    def __init__(self, solver):
        self.s = solver
        self.top = 1
        self.T = 1
        self.s.add_clause([self.T])
        self.nclauses = 1

    def new(self):
        self.top += 1
        return self.top

    def add(self, cl):
        self.s.add_clause(cl)
        self.nclauses += 1

    def const(self, v):
        return self.T if v else -self.T

    def is_const(self, a):
        return abs(a) == self.T

    def val(self, a):
        return a == self.T

    # ---- gates (full equivalence) ----
    def AND(self, xs):
        xs = [x for x in xs if x != self.T]
        if any(x == -self.T for x in xs):
            return -self.T
        xs = list(dict.fromkeys(xs))
        if not xs:
            return self.T
        if len(xs) == 1:
            return xs[0]
        if any(-x in xs for x in xs):
            return -self.T
        o = self.new()
        for x in xs:
            self.add([-o, x])
        self.add([o] + [-x for x in xs])
        return o

    def OR(self, xs):
        return -self.AND([-x for x in xs])

    def XOR(self, a, b):
        if self.is_const(a):
            return b if not self.val(a) else -b
        if self.is_const(b):
            return a if not self.val(b) else -a
        if a == b:
            return -self.T
        if a == -b:
            return self.T
        o = self.new()
        self.add([-o, a, b])
        self.add([-o, -a, -b])
        self.add([o, -a, b])
        self.add([o, a, -b])
        return o

    def MAJ(self, a, b, c):
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            if self.is_const(x):
                return self.OR([y, z]) if self.val(x) else self.AND([y, z])
        if a == b or a == c:
            return a
        if b == c:
            return b
        if a == -b:
            return c
        if a == -c:
            return b
        if b == -c:
            return a
        o = self.new()
        self.add([-o, a, b])
        self.add([-o, a, c])
        self.add([-o, b, c])
        self.add([o, -a, -b])
        self.add([o, -a, -c])
        self.add([o, -b, -c])
        return o

    # ---- unsigned binary numbers, LSB first ----
    def num_const(self, v, w):
        return [self.const((v >> i) & 1) for i in range(w)]

    def trim(self, a):
        a = list(a)
        while a and a[-1] == -self.T:
            a.pop()
        return a

    def add_bin(self, a, b):
        a, b = self.trim(a), self.trim(b)
        w = max(len(a), len(b))
        a = a + [-self.T] * (w - len(a))
        b = b + [-self.T] * (w - len(b))
        out, c = [], -self.T
        for x, y in zip(a, b):
            out.append(self.XOR(self.XOR(x, y), c))
            c = self.MAJ(x, y, c)
        out.append(c)
        return self.trim(out)

    def sum_bin(self, nums):
        nums = [self.trim(n) for n in nums]
        nums = [n for n in nums if n]
        if not nums:
            return []
        while len(nums) > 1:
            nums.sort(key=len)
            a, b = nums.pop(0), nums.pop(0)
            nums.append(self.add_bin(a, b))
        return nums[0]

    def popcount(self, lits):
        return self.sum_bin([[l] for l in lits])

    def ge(self, a, b):
        """a >= b (unsigned): carry out of a + NOT b + 1."""
        a, b = self.trim(a), self.trim(b)
        w = max(len(a), len(b))
        a = a + [-self.T] * (w - len(a))
        b = b + [-self.T] * (w - len(b))
        c = self.T
        for x, y in zip(a, b):
            c = self.MAJ(x, -y, c)
        return c

    def ge_const(self, a, k):
        if k <= 0:
            return self.T
        a = self.trim(a)
        if k >= 1 << len(a):
            return -self.T
        return self.ge(a, self.num_const(k, len(a)))

    def sub_const(self, a, k, w):
        """(a - k) mod 2^w."""
        a = self.trim(a)[:w]
        a = a + [-self.T] * (w - len(a))
        return self.add_bin(a, self.num_const(((1 << w) - k) % (1 << w), w))[:w]

    def mask(self, a, e):
        return [self.AND([x, e]) for x in a]

    def NOT(self, a, w):
        a = self.trim(a)
        a = a + [-self.T] * (w - len(a))
        return [-x for x in a]

    def lex_ge(self, a, b):
        """a >= b with a, b MSB first."""
        return self.ge(list(reversed(a)), list(reversed(b)))


def bits_for(v):
    return max(1, int(v).bit_length())


def merge_sorted(C, a, b):
    """Totalizer merge of two sorted unary sequences (a[t] = "count >= t+1"), full equivalence both ways."""
    if not a:
        return list(b)
    if not b:
        return list(a)
    n, m = len(a), len(b)
    c = [C.new() for _ in range(n + m)]
    A = [C.T] + list(a) + [-C.T]
    B = [C.T] + list(b) + [-C.T]
    for i in range(n + 1):
        for j in range(m + 1):
            if i + j >= 1:  # a >= i and b >= j  ->  c >= i + j
                C.add([x for x in (-A[i] if i else None, -B[j] if j else None) if x is not None] + [c[i + j - 1]])
            if i + j < n + m:  # a < i + 1 and b < j + 1  ->  c < i + j + 1
                C.add([x for x in (A[i + 1] if i < n else None, B[j + 1] if j < m else None) if x is not None]
                      + [-c[i + j]])
    return c


def sort_unary(C, seqs):
    seqs = [list(q) for q in seqs if q]
    if not seqs:
        return []
    while len(seqs) > 1:
        seqs.sort(key=len)
        a, b = seqs.pop(0), seqs.pop(0)
        seqs.append(merge_sorted(C, a, b))
    return seqs[0]


class SatNet:
    """Incremental SAT model for one (problem, H, cond, bias) instance.

    enc "binary": hidden values and scores as binary numbers (adders, comparators); compact at any H.
    enc "unary": order encoding throughout (b1, b2 as monotone chains, hidden pre-activations and scores as sorted
    totalizer outputs, unary comparators); larger, but unit propagation sees the arithmetic. Same model class."""

    def __init__(self, P, H, cond="reach", bias="small", b1r=3, b2r=6, solver="cadical195", require_used=False,
                 enc="binary", sink=None, strong=False, tie_order=False):
        """sink: any object with add_clause(list) (e.g. cnfproof.Sink, to write DIMACS) instead of a pysat solver.
        strong: extra clauses every real net satisfies (see cnfproof.py): argmax at-least-one and transitivity of
        "beats" per position, W2 column in the unit-ordering key, and with require_used + general biases
        b1 >= -r + 1 (b1 = -r is a dead unit).
        tie_order: equal scores are broken by a free strict total order T on the 9 cells (36 free variables plus
        transitivity clauses) instead of the lowest index: s beats m iff Y_s > Y_m or (Y_s = Y_m and T_sm). This is
        the move rule of every net with real-valued output biases (the fractional parts, then the index, order the
        ties), so UNSAT here covers real b2 and any tie order."""
        self.P, self.H, self.cond, self.enc, self.strong, self.bias = P, H, cond, enc, strong, bias
        self.tie_order = tie_order
        self.T = None
        if sink is None:
            from pysat.solvers import Solver

            sink = Solver(name=solver)
        self.s = sink
        self.name = solver if not hasattr(sink, "is_sink") else "sink"
        C = self.C = Circuit(self.s)
        if bias == "small":
            self.l1, self.u1 = -b1r, b1r
            self.u2 = 2 * b2r
        else:
            self.l1, self.u1 = -9, 9
            self.u2 = 8 * (2 * 18 * H + 1)
        u1 = self.u1
        self.w1 = [[(C.new(), C.new()) for _ in range(9)] for _ in range(H)]
        self.w2 = [[(C.new(), C.new()) for _ in range(H)] for _ in range(9)]
        for row in self.w1 + self.w2:
            for p, n in row:
                C.add([-p, -n])
        if enc == "binary":
            self.wB1 = bits_for(u1 - self.l1)
            self.wB2 = bits_for(self.u2)
            self.B1 = [[C.new() for _ in range(self.wB1)] for _ in range(H)]
            self.B2 = [[C.new() for _ in range(self.wB2)] for _ in range(9)]
            for j in range(H):  # B1 <= u1 - l1
                C.add([C.ge(C.num_const(u1 - self.l1, self.wB1), self.B1[j])])
            for k in range(9):
                C.add([C.ge(C.num_const(self.u2, self.wB2), self.B2[k])])
            C.add([C.AND([-x for x in self.B2[k]]) for k in range(9)])  # min b2 = 0
            b1key = [list(reversed(self.B1[j])) for j in range(H)]
        else:  # order encoding: U[t] = "B >= t + 1", U[t+1] -> U[t]
            self.B1 = [[C.new() for _ in range(u1 - self.l1)] for _ in range(H)]
            self.B2 = [[C.new() for _ in range(self.u2)] for _ in range(9)]
            for U in self.B1 + self.B2:
                for t in range(len(U) - 1):
                    C.add([-U[t + 1], U[t]])
            C.add([-self.B2[k][0] for k in range(9) if self.B2[k]])  # min b2 = 0
            b1key = [list(self.B1[j]) for j in range(H)]  # lex on a monotone chain = numeric order
        lo_off = 1 if (strong and require_used and bias == "general") else 0  # b1 >= -r + lo_off
        if bias == "general":  # -r_j <= b1_j <= r_j, r_j = nonzero weights of row j
            for j in range(H):
                lits = [v for pn in self.w1[j] for v in pn]
                if enc == "binary":
                    r = C.popcount(lits)
                    # b1 = B1 + l1 ; b1 <= r  <=>  B1 <= r - l1 ; b1 >= -r  <=>  B1 + r >= -l1
                    C.add([C.ge(C.add_bin(r, C.num_const(-self.l1, bits_for(-self.l1))), self.B1[j])])
                    C.add([C.ge_const(C.add_bin(self.B1[j], r), -self.l1 + lo_off)])
                else:
                    R = sort_unary(C, [[v] for v in lits])  # R[t] = "r >= t + 1", r <= 9 (p, n exclusive)
                    rge = lambda t: C.T if t <= 0 else (R[t - 1] if t <= len(R) else -C.T)  # noqa: E731
                    for t in range(1, len(self.B1[j]) + 1):  # B1 >= t -> r >= t + l1 ; B1 < t -> r >= 1 - t - l1
                        C.add([-self.B1[j][t - 1], rge(t + self.l1)])
                        C.add([self.B1[j][t - 1], rge(1 - t - self.l1 + lo_off)])
        # hidden-unit symmetry breaking: key_j >= key_{j+1}, key = (b1, W1 row) with each trit as (p, not n)
        keys = [b1key[j] + [x for i in range(8, -1, -1) for x in (self.w1[j][i][0], -self.w1[j][i][1])]
                for j in range(H)]
        if strong:  # ties on (b1, W1 row) broken by the W2 column: the whole unit moves when units are permuted
            keys = [keys[j] + [x for k in range(9) for x in (self.w2[k][j][0], -self.w2[k][j][1])] for j in range(H)]
        for j in range(H - 1):
            C.add([C.lex_ge(keys[j], keys[j + 1])])
        if require_used:  # every unit has an input and an output weight (sound once width H-1 is UNSAT)
            for j in range(H):
                C.add([v for pn in self.w1[j] for v in pn])
                C.add([v for k in range(9) for v in self.w2[k][j]])
        self.rowsum = {}
        self.reach = {}
        self.argmax = {}
        self.added = set()
        if tie_order:  # T[(s, m)]: s is ahead of m in the tie order; a strict total order on the 9 cells
            C = self.C
            self.T = {}
            for s_ in range(9):
                for m_ in range(s_ + 1, 9):
                    v = C.new()
                    self.T[(s_, m_)], self.T[(m_, s_)] = v, -v
            for a in range(9):
                for b in range(9):
                    for c in range(9):
                        if len({a, b, c}) == 3:
                            C.add([-self.T[(a, b)], -self.T[(b, c)], self.T[(a, c)]])
        self.card = None
        self.act = {}  # (position, unit) -> literal "z > 0"
        self.require_used = require_used

    def _row(self, j, r, pat):
        key = (j, r, pat)
        if key not in self.rowsum:
            lits = []
            for c, xv in zip(range(3 * r, 3 * r + 3), pat):
                p, n = self.w1[j][c]
                if xv == 1:
                    lits += [p, -n]
                elif xv == -1:
                    lits += [n, -p]
            self.rowsum[key] = self.C.popcount(lits) if self.enc == "binary" else sort_unary(self.C, [[v] for v in lits])
        return self.rowsum[key]

    def _beats_binary(self, i):
        C, P, H = self.C, self.P, self.H
        x = P.x[i]
        nocc = sum(1 for v in x if v)
        T0 = nocc - self.l1  # z = Z' - T0 with Z' = popcount + B1
        hs = []
        for j in range(H):
            pc = C.sum_bin([self._row(j, r, tuple(x[3 * r:3 * r + 3])) for r in range(3)])
            Z = C.add_bin(pc, self.B1[j])
            act = C.ge_const(Z, T0 + 1)
            self.act[(i, j)] = act
            w = len(C.trim(Z)) or 1
            hs.append(C.trim(C.mask(C.sub_const(Z, T0, w), act)))
        hb = max([len(h) for h in hs] + [1]) + bits_for(H)  # width of a sum of H values
        Y = {}
        for k in P.legal[i]:
            pos = C.sum_bin([C.mask(hs[j], self.w2[k][j][0]) for j in range(H)])
            neg = C.sum_bin([C.mask(hs[j], self.w2[k][j][1]) for j in range(H)])
            Y[k] = C.add_bin(C.add_bin(pos, self.B2[k]), C.NOT(neg, hb))
        return lambda s, m: C.ge(Y[s], Y[m])

    def _beats_unary(self, i):
        C, P, H = self.C, self.P, self.H
        x = P.x[i]
        nocc = sum(1 for v in x if v)
        T0 = nocc - self.l1  # count = popcount + B1 = z + T0
        hm = nocc + self.u1  # largest possible h here
        hs = []
        for j in range(H):
            cnt = sort_unary(C, [self._row(j, r, tuple(x[3 * r:3 * r + 3])) for r in range(3)] + [self.B1[j]])
            h = [cnt[T0 + v - 1] if T0 + v - 1 < len(cnt) else -C.T for v in range(1, hm + 1)]  # h[v-1] = h >= v
            self.act[(i, j)] = h[0] if h else -C.T
            hs.append(h)
        Y = {}
        for k in P.legal[i]:
            seqs = []
            for j in range(H):
                p, n = self.w2[k][j]
                # t' = W2 h + hm in [0, 2 hm]: t' >= u is "not (n and h >= hm - u + 1)" for u <= hm, "p and h >= u - hm" above
                seqs.append([-C.AND([n, hs[j][hm - u]]) for u in range(1, hm + 1)]
                            + [C.AND([p, hs[j][u - hm - 1]]) for u in range(hm + 1, 2 * hm + 1)])
            seqs.append(self.B2[k])
            Y[k] = sort_unary(C, seqs)

        def ge(s, m):  # Y_s >= Y_m  <=>  for every t: Y_m >= t -> Y_s >= t
            a, b = Y[s], Y[m]
            return C.AND([C.OR([-b[t], a[t] if t < len(a) else -C.T]) for t in range(len(b))])

        return ge

    def add_position(self, i):
        if i in self.added:
            return
        self.added.add(i)
        C, P = self.C, self.P
        ge = self._beats_binary(i) if self.enc == "binary" else self._beats_unary(i)
        beats = {}
        L = P.legal[i]
        for a in range(len(L)):
            for b in range(a + 1, len(L)):
                s, m = L[a], L[b]
                if self.tie_order:  # Y_s > Y_m, or equal and s ahead of m in T
                    g = C.AND([ge(s, m), C.OR([-ge(m, s), self.T[(s, m)]])])
                else:
                    g = ge(s, m)
                beats[(s, m)], beats[(m, s)] = g, -g
        am = {s: C.AND([beats[(s, m)] for m in L if m != s]) for s in L}
        self.argmax[i] = am
        if self.strong and len(L) > 1:  # beats is a strict total order on L, so exactly one argmax
            for a in L:
                for b in L:
                    for c in L:
                        if len({a, b, c}) == 3:
                            C.add([-beats[(a, b)], -beats[(b, c)], beats[(a, c)]])
            C.add([am[s] for s in L])
        good = P.allowed[i]
        if self.cond == "all":
            if len(good) < len(L):
                C.add([am[s] for s in good])
            return
        r = self.reach.setdefault(i, C.new())
        if len(good) < len(L):
            C.add([-r] + [am[s] for s in good])
        if i in P.roots:
            C.add([r])
        for s in good:  # edges out of i
            for q in P.edges[(i, s)]:
                if q in self.added:
                    C.add([-r, -am[s], self.reach[q]])
        for p, s in self.in_edges(i):  # edges into i from positions already added
            C.add([-self.reach[p], -self.argmax[p][s], r])

    def finish_positions(self):
        """`strong`, once every position of the instance is in: each unit is inactive at some position (a unit
        active at all of them is linear there: lowering b1 by its minimum z and adding that times W2 to b2 keeps
        every move), and with require_used active at some position (else it is 0 there: a width H-1 net)."""
        if not self.strong:
            return
        for j in range(self.H):
            lits = [a for (i, jj), a in self.act.items() if jj == j]
            if self.bias == "general":  # the shift can leave the `small` box, so only for general biases
                self.C.add([-a for a in lits])
            if self.require_used:
                self.C.add(lits)

    def in_edges(self, i):
        if not hasattr(self, "_in"):
            self._in = {}
            for (p, s), qs in self.P.edges.items():
                if s in self.P.allowed[p]:
                    for q in qs:
                        self._in.setdefault(q, []).append((p, s))
        return [(p, s) for p, s in self._in.get(i, []) if p in self.added and p != i]

    def fix_net(self, W1, b1, W2, b2):
        """Assumption literals that pin the weights to a given net (b2 shifted so its minimum is 0)."""
        out = []
        for row, mrow in ((r, m) for R, M in ((self.w1, W1), (self.w2, W2)) for r, m in zip(R, M)):
            for (p, n), v in zip(row, mrow):
                out += [p if v == 1 else -p, n if v == -1 else -n]
        lo = min(b2)
        for bits, v in list(zip(self.B1, [b - self.l1 for b in b1])) + list(zip(self.B2, [b - lo for b in b2])):
            if self.enc == "binary":
                assert v < 1 << len(bits), "bias out of range"
                out += [b if (v >> t) & 1 else -b for t, b in enumerate(bits)]
            else:
                assert 0 <= v <= len(bits), "bias out of range"
                out += [b if t < v else -b for t, b in enumerate(bits)]
        return out

    def nonzero_lits(self):
        return [v for row in self.w1 + self.w2 for pn in row for v in pn]

    def set_max_nonzero(self, k):
        """Returns the assumption literals that enforce nonzero <= k (totalizer, built once)."""
        from pysat.card import ITotalizer

        lits = self.nonzero_lits()
        if self.card is None:
            self.card = ITotalizer(lits=lits, ubound=len(lits), top_id=self.C.top)
            for cl in self.card.cnf.clauses:
                self.s.add_clause(cl)
            self.C.top = self.card.top_id
        if k >= len(lits):
            return []
        return [-self.card.rhs[k]]

    def solve(self, assumptions=(), timeout=None):
        """True / False / None (timed out). pysat cannot interrupt CaDiCaL, so for it the time limit is the
        caller's process timeout (the runner wraps every job in `timeout`)."""
        if self.name.startswith(("cd", "cadical")) or not timeout:
            return self.s.solve(assumptions=list(assumptions))
        timer = threading.Timer(timeout, self.s.interrupt)
        timer.start()
        try:
            r = self.s.solve_limited(assumptions=list(assumptions), expect_interrupt=True)
        finally:
            timer.cancel()
            self.s.clear_interrupt()
        return r

    def weights(self):
        model = self.s.get_model()
        val = lambda v: model[v - 1] > 0  # noqa: E731  (only positive variables are read)
        if self.enc == "binary":
            num = lambda bits: sum(1 << t for t, b in enumerate(bits) if val(b))  # noqa: E731
        else:
            num = lambda U: sum(1 for b in U if val(b))  # noqa: E731
        W1 = [[int(val(p)) - int(val(n)) for p, n in row] for row in self.w1]
        W2 = [[int(val(p)) - int(val(n)) for p, n in row] for row in self.w2]
        b1 = [num(B) + self.l1 for B in self.B1]
        b2 = [num(B) for B in self.B2]
        return W1, b1, W2, b2


def initial_subset(P):
    s = set(P.roots)
    for i in list(P.roots):
        for sm in P.legal[i]:
            s |= set(P.edges[(i, sm)])
    return s


def solve(P, H, cond="reach", bias="small", b1r=3, b2r=6, timeout=3600.0, min_nonzero=False, max_nonzero=None,
          require_used=False, solver="cadical195", log=print, full=False, enc="binary"):
    """CEGAR loop. Returns {status: SAT/UNSAT/UNKNOWN, net?, nonzero?, rounds, subset, seconds, clauses}.
    With min_nonzero: after the first SAT, tighten nonzero <= best - 1 until UNSAT (proved minimum at this H)."""
    t0 = time.time()
    S = SatNet(P, H, cond, bias, b1r, b2r, solver, require_used, enc)
    for i in sorted(range(len(P.boards)) if full else initial_subset(P)):
        S.add_position(i)
    rounds, best, assume = 0, None, []
    if max_nonzero is not None:
        assume = S.set_max_nonzero(max_nonzero)

    def row(status, **kw):
        out = {"status": status, "rounds": rounds, "subset": len(S.added), "seconds": round(time.time() - t0, 1),
               "vars": S.C.top, "clauses": S.C.nclauses}
        out.update(kw)
        return out

    while True:
        rounds += 1
        left = timeout - (time.time() - t0)
        if left <= 1:
            return row("UNKNOWN", **({"best": best} if best else {}))
        r = S.solve(assume, left)
        log(f"H={H} {cond} {bias} round {rounds}: subset {len(S.added)}/{len(P.boards)} -> "
            f"{ {True: 'SAT', False: 'UNSAT', None: 'UNKNOWN'}[r] } ({time.time() - t0:.1f} s, {S.C.nclauses} clauses)")
        if r is None:
            return row("UNKNOWN", **({"best": best} if best else {}))
        if r is False:
            if best is not None:  # min-nonzero: best is proved minimal at this H
                return row("SAT", net=best["net"], nonzero=best["nonzero"], optimal=True)
            return row("UNSAT")
        W1, b1, W2, b2 = S.weights()
        fail = P.failures(W1, b1, W2, b2, cond)
        new = [i for i in fail if i not in S.added]
        if fail and not new:
            raise AssertionError(f"SAT model and reference disagree at {fail[:5]}")
        if new:
            for i in new:
                S.add_position(i)
            continue
        net = P.to_net(W1, b1, W2, b2)
        nz = N.size(net)["nonzero"]
        if not min_nonzero:
            return row("SAT", net=net, nonzero=nz)
        best = {"net": net, "nonzero": nz}
        if nz == 0:
            return row("SAT", net=net, nonzero=0, optimal=True)
        log(f"  valid net with {nz} nonzero weights; trying <= {nz - 1}")
        assume = S.set_max_nonzero(nz - 1)

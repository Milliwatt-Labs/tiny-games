# Tic-tac-toe: how small can a never-losing ternary net be?

Every number here comes from a file in this folder or in `../weights/` (each weight file carries its own check in
`"verified"`). The model card is [../README.md](../README.md).

## Result

**A 9-input, 4-hidden, 9-output ternary MLP with 39 nonzero weights and 85 parameters never loses at
tic-tac-toe, as X or as O, against every opponent line** (`weights/smallest-never-loses-h4.json`). It plays on the
folded board (the smallest of its 8 rotations and reflections). Game lines from the empty board, net vs every
opponent reply: as X 120 wins, 32 draws, 0 losses; as O 380 wins, 215 draws, 0 losses. Its move is optimal in
74.69 % of all 4,520 positions with a move to make; a draw is safe, so it does not have to be.

To our knowledge it is the smallest published neural network verified to never lose tic-tac-toe as X and O
without search. It is the smallest we found, not a proven minimum.

What the guarantee covers:

- **From the start of a game.** The check plays out every opponent line from the empty board, with the net as X
  and as O. From a position the net would not reach by its own play it can lose (see arm C below).
- **Fixed rules outside the parameter count:** the symmetry folding, the legal-move mask and the lowest-index tie
  rule. With the folding, an X-only player needs no weights at all (below).

| saved net (`weights/`) | input | hidden | params | nonzero weights | weight slots | biases (nonzero) | never loses | optimal moves, all 4,520 |
|---|---|---|---|---|---|---|---|---|
| `smallest-never-loses-h4.json` (SAT search, pruned) | 9 trits, symmetry | 4 | **85** | **39** | 72 | 13 (11) | yes | 74.69 % |
| `smallest-never-loses-h5.json` (SAT search, pruned) | 9 trits, symmetry | 5 | 104 | 46 | 90 | 14 (5) | yes | 74.07 % |
| `smallest-never-loses.json` (trained) | 9 trits, symmetry | 10 | 199 | 63 | 180 | 19 (7) | yes | 79.65 % |
| `smallest-never-loses-params.json` (trained) | 9 trits, symmetry | 9 | 180 | 73 | 162 | 18 (9) | yes | 79.42 % |
| `smallest-optimal.json` (optimal everywhere) | 18 planes, symmetry | 16 | 457 | 286 | 432 | 25 (11) | yes | 100 % |
| `smallest-never-loses-nosym.json` (no folding) | 9 trits | 29 | 560 | 186 | 522 | 38 (11) | yes | 76.37 % |
| `smallest-optimal-nosym.json` (no folding) | 9 trits | 123 | 2,346 | 1,340 | 2,214 | 132 (40) | yes | 100 % |
| `smallest-never-loses-step.json` (binary hidden units) | 18 planes, symmetry | 15 | 429 | 114 | 405 | 24 (16) | yes | 77.52 % |

`smallest-sat-xo.json` is the same net as `smallest-never-loses-h5.json`; the page embeds it under that name.

## Nets that finish the game

The small nets never lose, but they do not always take a win. The 199-parameter net (63 nonzero weights), holding 4 and 6 as X, plays 5
instead of completing the 2-4-6 diagonal (`tests/test_ttt.py::test_game_2_4_6`). Two stricter levels, checked
over all 4,520 positions:

- **finisher**: never loses, and takes an immediate win in all 2,358 positions that have one.
- **converter**: a finisher that also keeps the win in all 2,836 positions that are won for the side to move,
  against every reply (`net.converts_by_play` plays every line).

| saved net (`weights/`) | input | hidden | params | nonzero weights | immediate wins taken | won positions kept | optimal moves |
|---|---|---|---|---|---|---|---|
| `smallest-finisher.json` | 9 trits, symmetry | 14 | 275 | **141** | 2,358 / 2,358 | 2,684 / 2,836 | 92.65 % |
| `smallest-converter.json` | 9 trits, symmetry | 17 | 332 | **204** | 2,358 / 2,358 | 2,836 / 2,836 | 96.11 % |
| `smallest-optimal.json` | 18 planes, symmetry | 16 | 457 | 286 | 2,126 / 2,358 | 2,836 / 2,836 | 100 % |
| `smallest-never-loses.json` | 9 trits, symmetry | 10 | 199 | 63 | 1,454 / 2,358 | 2,068 / 2,836 | 79.65 % |

The optimal net keeps every won position won but sometimes takes a slower win.

## The model

- Input: the board relative to the side to move, so one net plays both sides. `trits`: 9 inputs, +1 mover,
  -1 opponent, 0 empty. `planes`: 18 inputs in {0, 1}.
- Symmetry: the board is replaced by its canonical image (lexicographically smallest of the 8) before the net sees
  it, and the chosen cell is mapped back. No parameters.
- Layers: `h = relu(W1 x + b1)`, `z = W2 h + b2`, move = argmax of `z` over the empty cells, lowest index on ties.
  W in {-1, 0, +1}, b integers.
- Per-layer scales are all 1. A real scale on the hidden layer folds into the integer output bias, and argmax
  ignores the output scale. The integer net is the model; nothing is rounded at export.

## Method

1. **Exact solver** (`game.py`): minimax over all 5,478 legal positions (4,520 with a move to make, 627 up to
   symmetry). The root is a draw. Each position gets its set of optimal moves.
2. **Ternary QAT** (`train.py`): full-batch Adam on all positions (or the 627 canonical ones), forward with
   `clamp(round(W), -1, 1)` and `round(b)`, straight-through gradients, a float blend ramped to fully ternary over
   the first 60 % of steps, loss = -log P(optimal move set) + 0.3 x margin hinge.
3. **Discrete repair**: hill-climbing on single weights and biases to cut the positions with a wrong move,
   weighted towards positions in the net's own game tree.
4. **Pruning**: greedy removal of hidden units, then weights, then biases, keeping the net's level; re-pruned from
   12 random orders for the final picks.
5. **Exact search** (`minsat.py`, `satnet.py`): the never-lose question for width H as a SAT problem (below).
6. **Verification** (`net.py`): all 4,520 positions for the optimal-move share, and the full game tree from the
   empty board with the net as X and as O against every opponent reply.

The training sweep covered encodings {planes, trits} x symmetry {off, on} x widths 0 to 128, two-layer nets and
binary-step units (`sweep.jsonl`, `tables.md`); the finisher and converter grids are in `sweep-finish.jsonl`,
`shrink-finish.jsonl` and `tables-finish.md`. Training never found a never-losing net narrower than 8 hidden
units. The exact search found one at 4.

**Symmetry folding shrinks the net about 3x.** Without it the smallest never-losing net found is 29 hidden,
186 nonzero, 560 parameters; with it, 10 hidden and 63 nonzero from training, 4 hidden and 39 nonzero from the
exact search. For optimal play everywhere the gap is 1,340 against 286 nonzero weights.

## Exact search

Model class: exactly the saved nets' class, one hidden layer, trits input, with or without the folding. The
never-lose condition is exact: each net-to-move position gets a "reached" flag; the roots are reached; if a
reached position's move is s, every net-to-move position one opponent reply later is reached; at every reached
position the move must keep the mover's value >= 0. That is the same test as `net.game_tree`, so SAT means a net
that never loses and UNSAT means none of that width exists.

Bias ranges. `small`: b1 in [-3, 3], b2 spread at most 12. `general`: no loss of generality for integer biases.
A unit with r nonzero input weights has W1 x in [-r, r], so b1 below -r is a dead unit and b1 above r a linear one
that folds into b2; any b2 can then be squeezed to [0, 8 (36 H + 1)] without changing a comparison. The argument
is in the `cnfproof.py` docstring and `normalise`, and `tests/test_cnfproof.py` checks it on saved and random nets.

The circuit: popcounts for each pre-activation, ReLU, the masked output sums, a comparator per cell pair and an
AND per argmax, in binary or unary arithmetic; hidden units sorted to break their permutation symmetry. Solver:
CaDiCaL 1.9.5 through python-sat, incremental, with positions added as counterexamples (CEGAR). Every SAT net is
re-checked with `net.verify` before it is written. `tests/test_minsat.py` checks that the CNF accepts the saved
nets and rejects losing variants of them.

Both sides, with folding (rows in `solver-runs.jsonl` and `proofs.jsonl`):

| hidden | result |
|---|---|
| 1 | **UNSAT for any integer biases**; LRAT proof checked by lrat-check (CaDiCaL, 59.5 s); kissat with a DRAT proof checked by drat-trim agrees on the CNF without the positions step |
| 2 | **UNSAT for small biases**: three encodings agree (CEGAR 10,915 s, all 627 positions at once 2,377 s, unary 4,918 s), no proof file. General biases: running |
| 3 | open; general biases running as cube-and-conquer with a checked proof per cube |
| 4 | SAT, 47 nonzero; pruned to **39 nonzero, 85 parameters** |
| 5 | SAT, 54 nonzero; pruned to 46 nonzero, 104 parameters |
| 6 | SAT, 70 nonzero |

Without folding, width 1 is UNSAT for any integer biases; widths 8 to 16 did not finish, so 560 parameters is an
upper bound only. No nonzero-weight minimum is proved at any width: searches for at most 45 nonzero weights at
widths 5 to 7 ran out of time.

**What a checked proof certifies.** drat-trim and lrat-check confirm that the CNF file is unsatisfiable. That the
CNF encodes tic-tac-toe correctly rests on the encoder and its tests. `cnfproof.py build` writes the same file
byte for byte on any machine (width 1: sha256 `40f146598702bf42...`, in `proofs.jsonl`); CaDiCaL is deterministic,
so the proof can be regenerated rather than downloaded.

## X only

| net (`weights/`) | symmetry | hidden | params | nonzero weights | as X W/D/L | as O losses |
|---|---|---|---|---|---|---|
| `xonly-sym-bias-only.json` | yes | 1 (dead) | 28 | **0** | 156 / 48 / 0 | 462 |
| `smallest-x-only.json` | no | 2 | 47 | **10** | 109 / 38 / 0 | 197 |
| `xonly-raw-H1.json` | no | 1 | 28 | 11 | 103 / 20 / 0 | 178 |
| `xonly-raw-trained.json` | no | 2 | 47 | 11 | 157 / 46 / 0 | 117 |

- **With folding, X needs no weights.** A net whose hidden unit is dead plays a fixed order of cells set by its
  output biases. `priority.py` checked all 9! = 362,880 orders (`priority.json`): on the folded board 146 never
  lose as X; without folding none does, and none never loses as O either way. The folding is itself most of an X
  strategy, which is worth keeping in mind when reading every "with symmetry" number here.
- **Without folding**, one hidden unit with 11 nonzero weights is enough, and two units with general biases need
  only 10 ([x-only-diagram.png](x-only-diagram.png)). 10 is not proved minimal.

## Comparison

| player | size | sides | checked on every line | notes |
|---|---|---|---|---|
| this work | 85 params (39 nonzero ternary) | X and O | yes | no search; folding and mask fixed |
| this work, no folding | 560 params | X and O | yes | no search; mask fixed |
| Garrido-Merchán 2026 | 15,781 params | X and O | yes | hand-set network, no training |
| Fogel 1993 | 9-H-9, H <= 10; at most 199 real params | X | no | evolved; fitness from sampled games |
| Oltean, MEP F2 | 13-symbol real formula | X | yes | value function with 1-ply lookahead |
| Bhatt et al. 2008 | tables over 765 states | X and O | yes | evolved tables, not networks |
| Lomont 2024 | 22-entry table | X, centre opening | yes | SMT-proved minimal table |
| MENACE (Michie) | 304 matchboxes | X | no | bead counts, sampled play |

Units differ across rows (symbols, table entries, weights), so sizes are only compared like with like: total
parameters against total parameters.

- E. C. Garrido-Merchán, "Zero-Shot Weight-Space Programming Across a Difficulty-Graded Ladder of Two-Player
  Games", arXiv:2611.09271 (2026).
- D. B. Fogel, "Using Evolutionary Programming to Create Neural Networks that are Capable of Playing
  Tic-Tac-Toe", IEEE International Conference on Neural Networks (1993).
- M. Oltean, "New Evolutionary Computation Models and their Applications to Machine Learning", PhD thesis (2004),
  section 6.1.3; arXiv:2110.00468.
- A. Bhatt, P. Varshney, K. Deb, "In Search of No-loss Strategies for the Game of Tic-Tac-Toe using a Customized
  Genetic Algorithm", GECCO (2008).
- C. Lomont, "Minimal Size Tic Tac Toe Trees", https://www.lomont.org/posts/2024/minimaltictactoe/ (2024).
- D. Michie, "Experiments on the Mechanization of Game-Learning. Part I", The Computer Journal (1963).

## Game play: a stress test of the GPU player

The guarantees come from the exhaustive verifier; this run tests a second implementation of the players at scale
and adds nothing to them. The games covered 2,391 distinct complete games of the 255,168 possible, because the nets
are deterministic. Three nets played 2,590,000,000,000 games on one GPU
(`humanity/ttt_gpu.cu`, one whole game per thread): the finisher (275 parameters, 141 nonzero weights) and the
104-parameter net (46 nonzero) as X and O, and the 47-parameter X-only net (10 nonzero) as X. Against an optimal opponent, one that picks uniformly among value-keeping moves (1.3 trillion games), every game was a draw; against random,
"kid" and mixed opponents (1.26 trillion games) the nets lost none. At the start of every GPU session the GPU's
move matched `net.choose_move` on every position, and per-game traces matched a pure-Python player using the same
random streams. Every count is exact. The 85-parameter net was found after this run. Full tables:
[humanity.md](humanity.md), data: `humanity.json`.

Arm C starts each game with 1 to 3 random moves and then lets the net play both sides. It is outside the
guarantee and is reported so the guarantee is not over-read: of 3 x 10^10 games, 7,857,112,915 were lost by a side
whose position was at least a draw after the opening.

## Limits

- "Smallest" means the smallest found. Width 3 is open, and no nonzero-weight minimum is proved.
- The lower bounds hold for this model class: one hidden layer, ternary weights, integer biases, ReLU,
  side-relative trits. The width-2 bound is for bounded biases. Other inputs, depths or activations could be
  smaller.
- The never-lose guarantee is on-policy: it holds from the empty board, not from every position.
- The symmetry folding, the legal mask and the tie rule are outside every parameter count; their per-move cost is
  in `cost.json`. With the symmetry folding, 146 of the 362,880 fixed cell orders never lose as X with no weights.
- The hardware figures in `cost.json` are estimates under stated assumptions, not measurements.
- Repair and pruning are discrete search after QAT, not QAT alone.

## Files here

| file | what |
|---|---|
| `tables.md`, `tables-finish.md` | the training sweep tables |
| `sweep.jsonl`, `sweep-finish.jsonl`, `shrink-finish.jsonl`, `sweep-xonly.jsonl` | one row per training run |
| `solver-runs.jsonl`, `minsat.jsonl` | exact-search rows |
| `minsat-jobs*.json` | the exact-search job lists (`prove.py --jobs`) |
| `proofs.jsonl` | the checked width-1 proof: CNF and proof sha256, solver, check time |
| `priority.json` | the 9! fixed-order players |
| `humanity.md`, `humanity.json` | the 2.59 trillion game run |
| `x-only-diagram.png` | the 47-parameter X-only net (10 nonzero weights) |

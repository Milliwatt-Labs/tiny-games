# tiny-games

This repository holds small neural networks that play games, together with the code that trained them, searched for them and checked them. Each network has weights restricted to −1, 0 or +1, small integer biases and one hidden layer, and it chooses a move with one forward pass and no search. Each guarantee was checked against every line of play, without sampling.

| game | contents |
|---|---|
| [Tic-tac-toe](tictactoe/) | Networks that never lose tic-tac-toe as X or O from the start of a game, checked over the full game tree: a 123-parameter ternary model (every weight is −1, 0 or +1) with 6 hidden units and 67 nonzero weights that needs no symmetry step, and an 85-parameter ternary model with 4 hidden units and 39 nonzero weights that plays on the symmetry-reduced board; and larger ternary models that also take every immediate win (the 275-parameter ternary model, 141 nonzero weights), keep every won position won (the 332-parameter ternary model, 204 nonzero weights) or play optimally in every position (the 457-parameter ternary model, 286 nonzero weights). |

## Licence

The code is licensed under MIT ([LICENSE](LICENSE)). The weights, results data and figures in each game's `weights/` and `results/` folders, the model cards and the accompanying paper are licensed under CC BY 4.0 ([LICENSE-WEIGHTS](LICENSE-WEIGHTS)), with the attribution "Tic-tac-toe nets by Milliwatt Labs (milliwattlabs.com)".

This repository is maintained by [Milliwatt Labs](https://milliwattlabs.com/).

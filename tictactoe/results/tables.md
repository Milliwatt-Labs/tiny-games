# Sweep tables (432 runs)

| inputs / symmetry / act | hidden | runs | QAT perfect | QAT best optimal % | repair perfect | repair best optimal % | smallest perfect (nonzero / params) | smallest all-optimal (nonzero / params) |
|---|---|---|---|---|---|---|---|---|
| planes / raw / relu | 32x32 | 4 | 0 | 97.68 | 1 | 99.60 | 382 / 1403 | — |
| planes / raw / relu | 32x16 | 4 | 0 | 96.00 | 0 | 99.07 | — | — |
| planes / raw / relu | 16x16 | 4 | 0 | 92.43 | 0 | 96.04 | — | — |
| planes / raw / relu | 128 | 4 | 4 | 99.76 | 4 | 100.00 | 463 / 1941 | 2007 / 3565 |
| planes / raw / relu | 96 | 4 | 4 | 99.62 | 4 | 100.00 | 434 / 1661 | 1582 / 2641 |
| planes / raw / relu | 64 | 4 | 2 | 99.16 | 3 | 99.98 | 301 / 1129 | — |
| planes / raw / relu | 48 | 4 | 0 | 98.50 | 3 | 99.87 | 330 / 905 | — |
| planes / raw / relu | 32 | 4 | 0 | 97.30 | 0 | 99.20 | — | — |
| planes / raw / relu | 24 | 4 | 0 | 96.59 | 0 | 98.52 | — | — |
| planes / raw / relu | 16 | 4 | 0 | 95.35 | 0 | 97.46 | — | — |
| planes / raw / relu | 12 | 4 | 0 | 93.45 | 0 | 95.46 | — | — |
| planes / raw / relu | 0 | 2 | 0 | 73.39 | 0 | 76.99 | — | — |
| planes / sym / relu | 8x8 | 6 | 0 | 93.47 | 0 | 97.10 | — | — |
| planes / sym / relu | 8x6 | 6 | 0 | 91.95 | 0 | 96.28 | — | — |
| planes / sym / relu | 6x6 | 6 | 0 | 90.18 | 0 | 93.36 | — | — |
| planes / sym / relu | 6x4 | 6 | 0 | 88.32 | 0 | 91.59 | — | — |
| planes / sym / relu | 4x4 | 6 | 0 | 84.11 | 0 | 87.96 | — | — |
| planes / sym / relu | 16 | 6 | 3 | 97.79 | 5 | 100.00 | 126 / 401 | 293 / 457 |
| planes / sym / relu | 12 | 6 | 0 | 96.79 | 5 | 99.47 | 100 / 289 | — |
| planes / sym / relu | 10 | 6 | 1 | 95.44 | 2 | 98.23 | 85 / 233 | — |
| planes / sym / relu | 8 | 6 | 0 | 92.83 | 1 | 96.88 | 83 / 233 | — |
| planes / sym / relu | 7 | 6 | 0 | 92.83 | 0 | 96.19 | — | — |
| planes / sym / relu | 6 | 6 | 0 | 90.66 | 0 | 93.27 | — | — |
| planes / sym / relu | 5 | 6 | 0 | 88.81 | 0 | 91.24 | — | — |
| planes / sym / relu | 4 | 6 | 0 | 84.47 | 0 | 87.52 | — | — |
| planes / sym / relu | 0 | 2 | 0 | 77.70 | 0 | 84.16 | — | — |
| planes / sym / relu / tree repair | 10 | 12 | 0 | 96.02 | 8 | 97.35 | 104 / 261 | — |
| planes / sym / relu / tree repair | 9 | 12 | 0 | 94.82 | 7 | 96.02 | 91 / 233 | — |
| planes / sym / relu / tree repair | 8 | 12 | 0 | 93.52 | 3 | 95.49 | 93 / 233 | — |
| planes / sym / relu / tree repair | 7 | 12 | 0 | 92.26 | 0 | 94.16 | — | — |
| planes / sym / relu / tree repair | 6 | 12 | 0 | 90.97 | 0 | 92.21 | — | — |
| planes / sym / relu / tree repair | 5 | 12 | 0 | 89.03 | 0 | 89.03 | — | — |
| planes / sym / relu / tree repair | 4 | 12 | 0 | 86.15 | 0 | 84.87 | — | — |
| planes / sym / step | 8x8 | 3 | 0 | 89.51 | 0 | 94.96 | — | — |
| planes / sym / step | 16 | 3 | 0 | 94.82 | 1 | 99.82 | 145 / 429 | — |
| planes / sym / step | 12 | 3 | 0 | 93.36 | 0 | 98.76 | — | — |
| planes / sym / step | 8 | 3 | 0 | 92.39 | 0 | 96.02 | — | — |
| trits / raw / relu | 32x32 | 4 | 0 | 97.63 | 1 | 99.69 | 278 / 767 | — |
| trits / raw / relu | 32x16 | 4 | 0 | 96.70 | 0 | 99.09 | — | — |
| trits / raw / relu | 16x16 | 4 | 0 | 94.25 | 0 | 97.26 | — | — |
| trits / raw / relu | 128 | 4 | 4 | 99.69 | 4 | 100.00 | 405 / 1130 | 1340 / 2308 |
| trits / raw / relu | 96 | 4 | 4 | 99.16 | 4 | 99.98 | 335 / 845 | — |
| trits / raw / relu | 64 | 4 | 1 | 98.72 | 2 | 99.78 | 270 / 674 | — |
| trits / raw / relu | 48 | 4 | 0 | 98.03 | 0 | 99.42 | — | — |
| trits / raw / relu | 32 | 4 | 0 | 96.97 | 0 | 98.45 | — | — |
| trits / raw / relu | 24 | 4 | 0 | 96.57 | 0 | 98.21 | — | — |
| trits / raw / relu | 16 | 4 | 0 | 95.88 | 0 | 97.21 | — | — |
| trits / raw / relu | 12 | 4 | 0 | 92.77 | 0 | 94.60 | — | — |
| trits / raw / relu | 0 | 2 | 0 | 71.06 | 0 | 75.40 | — | — |
| trits / sym / relu | 8x8 | 6 | 0 | 93.05 | 0 | 96.55 | — | — |
| trits / sym / relu | 8x6 | 6 | 0 | 92.12 | 0 | 95.31 | — | — |
| trits / sym / relu | 6x6 | 6 | 0 | 89.20 | 0 | 91.99 | — | — |
| trits / sym / relu | 6x4 | 6 | 0 | 88.14 | 0 | 91.59 | — | — |
| trits / sym / relu | 4x4 | 6 | 0 | 85.31 | 0 | 86.50 | — | — |
| trits / sym / relu | 16 | 6 | 1 | 98.01 | 4 | 99.56 | 90 / 256 | — |
| trits / sym / relu | 12 | 6 | 0 | 96.28 | 0 | 98.41 | — | — |
| trits / sym / relu | 10 | 6 | 0 | 94.87 | 1 | 97.17 | 101 / 199 | — |
| trits / sym / relu | 8 | 6 | 0 | 92.65 | 0 | 95.13 | — | — |
| trits / sym / relu | 7 | 6 | 0 | 91.79 | 0 | 94.67 | — | — |
| trits / sym / relu | 6 | 6 | 0 | 90.97 | 0 | 93.10 | — | — |
| trits / sym / relu | 5 | 6 | 0 | 90.44 | 0 | 91.33 | — | — |
| trits / sym / relu | 4 | 6 | 0 | 87.04 | 0 | 88.58 | — | — |
| trits / sym / relu | 0 | 2 | 0 | 77.12 | 0 | 82.74 | — | — |
| trits / sym / relu / tree repair | 10 | 12 | 0 | 96.35 | 4 | 96.81 | 73 / 199 | — |
| trits / sym / relu / tree repair | 9 | 12 | 0 | 93.87 | 1 | 94.96 | 84 / 180 | — |
| trits / sym / relu / tree repair | 8 | 12 | 0 | 92.88 | 0 | 93.50 | — | — |
| trits / sym / relu / tree repair | 7 | 12 | 0 | 91.15 | 0 | 91.59 | — | — |
| trits / sym / relu / tree repair | 6 | 12 | 0 | 89.60 | 0 | 89.67 | — | — |
| trits / sym / relu / tree repair | 5 | 12 | 0 | 88.41 | 0 | 86.02 | — | — |
| trits / sym / relu / tree repair | 4 | 12 | 0 | 84.56 | 0 | 85.04 | — | — |

| pick | from run | hidden | nonzero weights | weights | biases (nonzero) | total params | optimal % (all 4,520) | optimal % (in its game tree) | lines as X W/D/L | lines as O W/D/L |
|---|---|---|---|---|---|---|---|---|---|---|
| smallest-never-loses.json | push-trits-sym-relu-10-s8 | 10 | 63 | 180 | 19 (7) | 199 | 79.65 | 87.33 | 128/28/0 | 334/293/0 |
| smallest-never-loses-params.json | push-trits-sym-relu-10-s10 | 9 | 73 | 162 | 18 (9) | 180 | 79.42 | 87.63 | 191/48/0 | 375/286/0 |
| smallest-optimal.json | planes-sym-relu-16-s0 | 16 | 286 | 432 | 25 (11) | 457 | 100.00 | 100.00 | 108/4/0 | 404/111/0 |
| smallest-never-loses-nosym.json | trits-raw-relu-64-s0 | 29 | 186 | 522 | 38 (11) | 560 | 76.37 | 90.28 | 111/33/0 | 353/230/0 |
| smallest-optimal-nosym.json | trits-raw-relu-128-s2 | 123 | 1340 | 2214 | 132 (40) | 2346 | 100.00 | 100.00 | 107/2/0 | 449/114/0 |
| smallest-never-loses-step.json | planes-sym-step-16-s0 | 15 | 114 | 405 | 24 (16) | 429 | 77.52 | 88.60 | 192/44/0 | 438/275/0 |

Canonical decision positions: 627 (338 with X to move, 304 of those with more than one empty cell).

# Tic-tac-toe: more games than humanity

Updated 2026-11-22T11:21:09-0500. Code: `tictactoe/humanity/`. Data: `results/humanity.json`.

Humanity's games of tic-tac-toe, order of magnitude: 1e+12. Games played here: **2,590,000,000,000** (2.59x), 21,706,026,226,715 moves, 0.333 GPU-hours on one NVIDIA GB10, 0.029 kWh measured at the GPU (0.025 kWh above idle). Distinct complete games seen: 2,391 of 255,168.

| group | games | net wins | draws | net losses |
|---|---:|---:|---:|---:|
| A | 1,300,000,000,000 | 0 | 1,300,000,000,000 | 0 |
| B | 1,260,000,000,000 | 621,013,194,812 | 638,986,805,188 | 0 |

| net | plays | games | wins | draws | losses | GPU-h | kWh |
|---|---|---:|---:|---:|---:|---:|---:|
| finisher (`smallest-finisher.json`) | XO | 1,024,000,000,000 | 305,746,936,979 | 718,253,063,021 | 0 | 0.170 | 0.0150 |
| satxo (`smallest-sat-xo.json`) | XO | 1,024,000,000,000 | 183,215,928,026 | 840,784,071,974 | 0 | 0.131 | 0.0115 |
| xonly (`smallest-x-only.json`) | X | 512,000,000,000 | 132,050,329,807 | 379,949,670,193 | 0 | 0.025 | 0.0021 |

Every count is exact. At the start of every GPU session the GPU move function equalled `net.choose_move` on all 4,520 to-move positions for finisher and satxo, and on all 2,423 X-to-move positions for xonly (no symmetry, plays X only), and per-game traces equalled a pure-Python reference player (same Philox streams) for every arm.

## A: net vs a perfect opponent (uniform over value-keeping moves)

| arm | games | wins | draws | losses | distinct games | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| A-perfect-finisher-x | 260,000,000,000 | 0 | 260,000,000,000 | 0 | 4 | 0.040 |
| A-perfect-finisher-o | 260,000,000,000 | 0 | 260,000,000,000 | 0 | 79 | 0.043 |
| A-perfect-satxo-x | 260,000,000,000 | 0 | 260,000,000,000 | 0 | 8 | 0.033 |
| A-perfect-satxo-o | 260,000,000,000 | 0 | 260,000,000,000 | 0 | 127 | 0.031 |
| A-perfect-xonly-x | 260,000,000,000 | 0 | 260,000,000,000 | 0 | 8 | 0.011 |

## B: net vs human-like opponents

random = uniform over empty cells; kidP = takes a win with probability P, else blocks with probability P, else random; mixQ = a perfect move with probability Q, else a random one.

| arm | games | wins | draws | losses | distinct games | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| B-random-finisher-x | 36,000,000,000 | 35,625,008,803 | 374,991,197 | 0 | 76 | 0.005 |
| B-kid50-finisher-x | 36,000,000,000 | 31,078,062,966 | 4,921,937,034 | 0 | 76 | 0.007 |
| B-kid90-finisher-x | 36,000,000,000 | 21,500,606,082 | 14,499,393,918 | 0 | 76 | 0.006 |
| B-mix25-finisher-x | 36,000,000,000 | 33,692,926,911 | 2,307,073,089 | 0 | 76 | 0.006 |
| B-mix50-finisher-x | 36,000,000,000 | 28,617,204,931 | 7,382,795,069 | 0 | 76 | 0.006 |
| B-mix75-finisher-x | 36,000,000,000 | 18,271,004,783 | 17,728,995,217 | 0 | 76 | 0.006 |
| B-mix90-finisher-x | 36,000,000,000 | 8,451,143,357 | 27,548,856,643 | 0 | 76 | 0.006 |
| B-random-finisher-o | 36,000,000,000 | 32,990,444,783 | 3,009,555,217 | 0 | 457 | 0.006 |
| B-kid50-finisher-o | 36,000,000,000 | 21,866,572,812 | 14,133,427,188 | 0 | 457 | 0.006 |
| B-kid90-finisher-o | 36,000,000,000 | 7,043,097,144 | 28,956,902,856 | 0 | 457 | 0.006 |
| B-mix25-finisher-o | 36,000,000,000 | 28,299,952,201 | 7,700,047,799 | 0 | 457 | 0.006 |
| B-mix50-finisher-o | 36,000,000,000 | 21,314,293,114 | 14,685,706,886 | 0 | 457 | 0.006 |
| B-mix75-finisher-o | 36,000,000,000 | 11,918,876,016 | 24,081,123,984 | 0 | 457 | 0.006 |
| B-mix90-finisher-o | 36,000,000,000 | 5,077,743,076 | 30,922,256,924 | 0 | 457 | 0.006 |
| B-random-satxo-x | 36,000,000,000 | 31,124,922,914 | 4,875,077,086 | 0 | 236 | 0.005 |
| B-kid50-satxo-x | 36,000,000,000 | 18,046,901,489 | 17,953,098,511 | 0 | 236 | 0.005 |
| B-kid90-satxo-x | 36,000,000,000 | 3,858,417,392 | 32,141,582,608 | 0 | 236 | 0.005 |
| B-mix25-satxo-x | 36,000,000,000 | 26,815,366,810 | 9,184,633,190 | 0 | 236 | 0.005 |
| B-mix50-satxo-x | 36,000,000,000 | 20,109,415,102 | 15,890,584,898 | 0 | 236 | 0.005 |
| B-mix75-satxo-x | 36,000,000,000 | 11,042,050,900 | 24,957,949,100 | 0 | 236 | 0.005 |
| B-mix90-satxo-x | 36,000,000,000 | 4,614,265,065 | 31,385,734,935 | 0 | 236 | 0.005 |
| B-random-satxo-o | 36,000,000,000 | 19,619,075,205 | 16,380,924,795 | 0 | 783 | 0.004 |
| B-kid50-satxo-o | 36,000,000,000 | 10,571,459,949 | 25,428,540,051 | 0 | 783 | 0.005 |
| B-kid90-satxo-o | 36,000,000,000 | 3,059,005,745 | 32,940,994,255 | 0 | 783 | 0.005 |
| B-mix25-satxo-o | 36,000,000,000 | 15,477,455,994 | 20,522,544,006 | 0 | 783 | 0.005 |
| B-mix50-satxo-o | 36,000,000,000 | 10,842,822,812 | 25,157,177,188 | 0 | 783 | 0.005 |
| B-mix75-satxo-o | 36,000,000,000 | 5,691,685,389 | 30,308,314,611 | 0 | 783 | 0.005 |
| B-mix90-satxo-o | 36,000,000,000 | 2,343,083,260 | 33,656,916,740 | 0 | 783 | 0.005 |
| B-random-xonly-x | 36,000,000,000 | 32,437,581,198 | 3,562,418,802 | 0 | 147 | 0.002 |
| B-kid50-xonly-x | 36,000,000,000 | 20,671,768,425 | 15,328,231,575 | 0 | 147 | 0.002 |
| B-kid90-xonly-x | 36,000,000,000 | 4,927,773,466 | 31,072,226,534 | 0 | 147 | 0.002 |
| B-mix25-xonly-x | 36,000,000,000 | 29,260,917,907 | 6,739,082,093 | 0 | 147 | 0.002 |
| B-mix50-xonly-x | 36,000,000,000 | 23,718,840,939 | 12,281,159,061 | 0 | 147 | 0.002 |
| B-mix75-xonly-x | 36,000,000,000 | 14,492,470,400 | 21,507,529,600 | 0 | 147 | 0.002 |
| B-mix90-xonly-x | 36,000,000,000 | 6,540,977,472 | 29,459,022,528 | 0 | 147 | 0.002 |

## C: random opening, then the net plays both sides

Not part of the zero-loss claim. 'opening value' is the minimax value after the random moves. 'lost from a draw or better' counts games where the losing side's position was at least a draw after the opening; 'won, not converted' counts won openings that ended in a draw.

| arm | games | X wins | draws | O wins | lost from a draw or better | won, not converted | distinct |
|---|---:|---:|---:|---:|---:|---:|---:|
| C-open1-finisher | 5,000,000,000 | 0 | 2,777,780,330 | 2,222,219,670 | 2,222,219,670 | 0 | 9 |
| C-open2-finisher | 5,000,000,000 | 2,777,725,136 | 1,111,164,906 | 1,111,109,958 | 1,388,897,316 | 0 | 72 |
| C-open3-finisher | 5,000,000,000 | 1,825,352,428 | 1,230,187,816 | 1,944,459,756 | 1,984,105,867 | 0 | 504 |
| C-open1-satxo | 5,000,000,000 | 0 | 5,000,000,000 | 0 | 0 | 0 | 9 |
| C-open2-satxo | 5,000,000,000 | 1,944,434,292 | 2,222,200,028 | 833,365,680 | 833,365,680 | 833,327,053 | 72 |
| C-open3-satxo | 5,000,000,000 | 873,022,681 | 2,261,900,132 | 1,865,077,187 | 1,428,524,382 | 555,573,057 | 504 |

By opening value (X wins / draws / O wins):

- C-open1-finisher: opening x wins 0 -> 0 / 0 / 0; opening draw 5,000,000,000 -> 0 / 2,777,780,330 / 2,222,219,670; opening o wins 0 -> 0 / 0 / 0
- C-open2-finisher: opening x wins 3,333,280,825 -> 2,499,937,778 / 0 / 833,343,047; opening draw 1,666,719,175 -> 277,787,358 / 1,111,164,906 / 277,766,911; opening o wins 0 -> 0 / 0 / 0
- C-open3-finisher: opening x wins 1,269,799,337 -> 1,111,065,463 / 0 / 158,733,874; opening draw 2,738,088,348 -> 396,815,504 / 1,230,187,816 / 1,111,085,028; opening o wins 992,112,315 -> 317,471,461 / 0 / 674,640,854
- C-open1-satxo: opening x wins 0 -> 0 / 0 / 0; opening draw 5,000,000,000 -> 0 / 5,000,000,000 / 0; opening o wins 0 -> 0 / 0 / 0
- C-open2-satxo: opening x wins 3,333,308,717 -> 1,944,434,292 / 833,327,053 / 555,547,372; opening draw 1,666,691,283 -> 0 / 1,388,872,975 / 277,818,308; opening o wins 0 -> 0 / 0 / 0
- C-open3-satxo: opening x wins 1,269,836,749 -> 555,571,543 / 476,208,518 / 238,056,688; opening draw 2,738,087,702 -> 158,744,071 / 1,706,327,075 / 873,016,556; opening o wins 992,075,549 -> 158,707,067 / 79,364,539 / 754,003,943

RNG: Philox4x32-10, counter (game lo, game hi, arm_id, block), key (seed0, seed1); seeds 20261122, 1213549889; arm ids in the JSON. Energy: nvidia-smi power.draw every 500 ms while the binary ran (the whole GPU, resident services included); 'above idle' subtracts the mean draw measured for 6 s before each session.

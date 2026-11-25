# Finish sweep (96 runs)

QAT on the finisher target, repair, greedy prune (sweep.py --grid finish):

| inputs | hidden | runs | repaired to finisher | smallest finisher (nonzero / params) | converter | smallest converter (nonzero / params) |
|---|---|---|---|---|---|---|
| planes / sym | 12 | 8 | 1 | 233 / 345 | 0 | — |
| planes / sym | 14 | 8 | 2 | 235 / 401 | 0 | — |
| planes / sym | 16 | 8 | 6 | 233 / 429 | 1 | 285 / 457 |
| planes / sym | 20 | 8 | 8 | 290 / 541 | 7 | 320 / 541 |
| planes / sym | 24 | 8 | 8 | 305 / 625 | 7 | 346 / 653 |
| planes / sym | 32 | 8 | 8 | 408 / 821 | 8 | 436 / 849 |
| trits / sym | 12 | 8 | 0 | — | 0 | — |
| trits / sym | 14 | 8 | 0 | — | 0 | — |
| trits / sym | 16 | 8 | 1 | 168 / 313 | 0 | — |
| trits / sym | 20 | 8 | 7 | 220 / 389 | 1 | 249 / 389 |
| trits / sym | 24 | 8 | 7 | 228 / 427 | 4 | 276 / 446 |
| trits / sym | 32 | 8 | 8 | 302 / 560 | 7 | 325 / 598 |

Drop-a-unit-and-repair (train.shrink) on the smallest of each kind:

| goal | from run | before (hidden / nonzero) | after (hidden / nonzero / params) | seconds |
|---|---|---|---|---|
| converter | finish-trits-sym-relu-24-s2 | 23 / 280 | 17 / 204 / 332 | 510 |
| converter | finish-trits-sym-relu-24-s0 | 24 / 285 | 20 / 220 / 389 | 428 |
| converter | finish-trits-sym-relu-20-s7 | 20 / 249 | 17 / 224 / 332 | 248 |
| converter | finish-planes-sym-relu-16-s0 | 16 / 285 | 14 / 252 / 401 | 181 |
| converter | finish-trits-sym-relu-24-s1 | 23 / 276 | 21 / 253 / 408 | 241 |
| converter | finish-trits-sym-relu-24-s3 | 24 / 300 | 23 / 274 / 446 | 301 |
| finisher | finish-trits-sym-relu-20-s4 | 20 / 220 | 14 / 141 / 275 | 264 |
| finisher | finish-trits-sym-relu-16-s2 | 16 / 168 | 16 / 161 / 313 | 112 |
| finisher | finish-trits-sym-relu-24-s4 | 23 / 228 | 17 / 170 / 332 | 418 |
| finisher | finish-trits-sym-relu-20-s3 | 20 / 225 | 16 / 177 / 313 | 301 |
| finisher | finish-trits-sym-relu-20-s6 | 20 / 226 | 16 / 179 / 313 | 375 |
| finisher | finish-trits-sym-relu-20-s0 | 20 / 225 | 17 / 192 / 332 | 294 |

| pick | from run | hidden | nonzero weights | weights | biases (nonzero) | total params | immediate wins taken | won positions kept | optimal % (all 4,520) | lines as X W/D/L | lines as O W/D/L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smallest-finisher.json | shrink-finisher-finish-trits-sym-relu-20-s4 | 14 | 141 | 252 | 23 (17) | 275 | 2358/2358 | 2684/2836 | 92.65 | 72/4/0 | 378/79/0 |
| smallest-converter.json | shrink-converter-finish-trits-sym-relu-24-s2 | 17 | 204 | 306 | 26 (13) | 332 | 2358/2358 | 2836/2836 | 96.11 | 72/4/0 | 362/95/0 |

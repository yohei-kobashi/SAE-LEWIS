# B1 LinguaLens-clamp baseline (LinguaLens)

pairs scored: 499; rewriter google/gemma-2-2b-it; intervention=clamp on layers[12]; value/alpha sweep [5.0, 10.0, 20.0]; conditioning identical to the EF probe

| condition | mode | exact | sim_target | copy |
|---|---|---|---|---|
| true | clamp10 | 0.1743 | 0.5276 | 0.0942 |
| true | clamp20 | 0.0301 | 0.2125 | 0.0080 |
| true | clamp5 | 0.0641 | 0.5189 | 0.2585 |
| true | clampZ | 0.1483 | 0.5198 | 0.1563 |
| empty | raw | 0.0601 | 0.5490 | 0.0661 |
| empty | recon | 0.0100 | 0.5413 | 0.4770 |
| random | clamp10 | 0.0180 | 0.5287 | 0.4188 |

## Multi-site breakdown (condition = true)

| n_ops | pairs | clamp10 exact | clamp20 exact | clamp5 exact | clampZ exact | clamp10 sim | clamp20 sim | clamp5 sim | clampZ sim |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 99 | 0.1212 | 0.0202 | 0.1010 | 0.1515 | 0.6527 | 0.2493 | 0.6791 | 0.7453 |
| 2-3 | 210 | 0.2619 | 0.0429 | 0.1000 | 0.2238 | 0.6097 | 0.2262 | 0.6072 | 0.6020 |
| 4-8 | 177 | 0.1130 | 0.0226 | 0.0056 | 0.0621 | 0.3763 | 0.1774 | 0.3375 | 0.3142 |
| 9+ | 13 | 0.0000 | 0.0000 | 0.0000 | 0.0769 | 0.3078 | 0.1886 | 0.3437 | 0.2752 |

# B2 prompt-rewrite baseline (LinguaLens)

pairs scored: 499; rewriter google/gemma-2-2b-it; n_desc [8, 16]; conditioning identical to the EF probe (local, blocklist, k=64/64)

| condition | mode | exact | sim_target | copy |
|---|---|---|---|---|
| true | prompt16 | 0.1242 | 0.6118 | 0.0822 |
| true | prompt8 | 0.1242 | 0.6058 | 0.0681 |
| empty | prompt8 | 0.0341 | 0.5526 | 0.4770 |
| random | prompt8 | 0.0100 | 0.4996 | 0.2745 |

## Multi-site breakdown (condition = true)

| n_ops | pairs | prompt16 exact | prompt8 exact | prompt16 sim | prompt8 sim |
|---|---|---|---|---|---|
| 1 | 99 | 0.0909 | 0.0707 | 0.6637 | 0.6381 |
| 2-3 | 210 | 0.1667 | 0.1667 | 0.6814 | 0.6769 |
| 4-8 | 177 | 0.0904 | 0.1073 | 0.5141 | 0.5170 |
| 9+ | 13 | 0.1538 | 0.0769 | 0.4219 | 0.4197 |

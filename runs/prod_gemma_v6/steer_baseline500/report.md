# B3 steering-vector baseline (LinguaLens)

pairs scored: 997; rewriter google/gemma-2-2b-it; intervention=steer on layers[12]; value/alpha sweep [0.5]; conditioning identical to the EF probe

| condition | mode | exact | sim_target | copy |
|---|---|---|---|---|
| true | steer0.5 | 0.2337 | 0.6016 | 0.0802 |
| true | steer1 | 0.1222 | 0.3511 | 0.1002 |
| true | steer2 | 0.0120 | 0.0854 | 0.0802 |
| true | steer4 | 0.0000 | 0.0257 | 0.0441 |
| empty | raw | 0.0542 | 0.5507 | 0.0863 |
| random | steer0.5 | 0.0321 | 0.4576 | 0.1948 |
| random | steer1 | 0.0200 | 0.4127 | 0.3407 |

## Multi-site breakdown (condition = true)

| n_ops | pairs | steer0.5 exact | steer1 exact | steer2 exact | steer4 exact | steer0.5 sim | steer1 sim | steer2 sim | steer4 sim |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 227 | 0.0881 | 0.1717 | 0.0606 | 0.0000 | 0.5714 | 0.6754 | 0.1879 | 0.0083 |
| 2-3 | 381 | 0.3675 | 0.2000 | 0.0000 | 0.0000 | 0.7113 | 0.4512 | 0.0792 | 0.0417 |
| 4-8 | 360 | 0.1889 | 0.0056 | 0.0000 | 0.0000 | 0.5172 | 0.0694 | 0.0417 | 0.0181 |
| 9+ | 29 | 0.1724 | 0.0769 | 0.0000 | 0.0000 | 0.4457 | 0.1003 | 0.0000 | 0.0022 |

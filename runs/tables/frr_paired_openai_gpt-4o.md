# Paired FRR tests — openai_gpt-4o

Exact McNemar over pairs where both systems have a scorable gold direction. Discordant pairs (one system realized the feature, the other did not) carry the signal; judge noise is priced into them, and it attenuates gaps toward chance rather than fabricating them, so each Δ reads as a conservative estimate of the true gap.

| A vs B | n | FRR A | FRR B | Δ | A only | B only | p (exact) |
|---|---|---|---|---|---|---|---|
| ef32 vs routed | 980 | 0.8735 | 0.8306 | +0.0429 | 54 | 12 | 1.69e-07 *** |
| ef32 vs steer | 980 | 0.8735 | 0.7327 | +0.1408 | 205 | 67 | 1.93e-17 *** |
| routed vs steer | 980 | 0.8306 | 0.7327 | +0.0980 | 155 | 59 | 3.94e-11 *** |

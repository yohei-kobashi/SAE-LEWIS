# Paired FRR tests — hf_google_gemma-2-9b-it

Exact McNemar over pairs where both systems have a scorable gold direction. Discordant pairs (one system realized the feature, the other did not) carry the signal; judge noise is priced into them, and it attenuates gaps toward chance rather than fabricating them, so each Δ reads as a conservative estimate of the true gap.

| A vs B | n | FRR A | FRR B | Δ | A only | B only | p (exact) |
|---|---|---|---|---|---|---|---|
| ef32 vs routed | 983 | 0.7894 | 0.7691 | +0.0203 | 49 | 29 | 3.08e-02 * |
| ef32 vs steer | 983 | 0.7894 | 0.7406 | +0.0488 | 167 | 119 | 5.36e-03 ** |
| routed vs steer | 983 | 0.7691 | 0.7406 | +0.0285 | 118 | 90 | 6.09e-02 n.s. |

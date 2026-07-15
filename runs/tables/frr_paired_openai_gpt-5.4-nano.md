# Paired FRR tests — openai_gpt-5.4-nano

Exact McNemar over pairs where both systems have a scorable gold direction. Discordant pairs (one system realized the feature, the other did not) carry the signal; judge noise is priced into them, and it attenuates gaps toward chance rather than fabricating them, so each Δ reads as a conservative estimate of the true gap.

| A vs B | n | FRR A | FRR B | Δ | A only | B only | p (exact) |
|---|---|---|---|---|---|---|---|
| ef32 vs routed | 978 | 0.7679 | 0.7331 | +0.0348 | 67 | 33 | 8.74e-04 *** |
| ef32 vs steer | 978 | 0.7679 | 0.6922 | +0.0757 | 180 | 106 | 1.43e-05 *** |
| routed vs steer | 978 | 0.7331 | 0.6922 | +0.0409 | 135 | 95 | 9.97e-03 ** |

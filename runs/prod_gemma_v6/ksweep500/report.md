# Conditioning k-sweep (P-A frontier + P-C selection)

pairs: 997; decode thr0.1; ckpt ./runs/prod_gemma_v6/editflow_s3/editflow-final.pt

## P-A — exact/sim vs k (top-|dz| order)

| k | exact | sim | copy | mean gain |
|---|---|---|---|---|
| 32 | 0.2237 | 0.5809 | 0.0572 | 0.0176 |

r95 (smallest k reaching 95% of k=32 exact 0.2237): **32**

## Oracle per-pair minimal k (analysis only — label-peeking, not a method)

| min k | pairs |
|---|---|
| 32 | 223 |
| never | 774 |

## P-C — deployable selection: smallest k with sae_gain >= tau (else largest)

| tau | exact | sim | mean chosen k |
|---|---|---|---|
| 0.3 | 0.2237 | 0.5809 | 32.0 |
| 0.5 | 0.2237 | 0.5809 | 32.0 |
| 0.7 | 0.2237 | 0.5809 | 32.0 |

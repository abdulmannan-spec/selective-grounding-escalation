# Results (served system = gpt_cid, n = 420 queries)

No abstention: Exact@30s = 0.343, MAE = 743.4 s.

## A. Fixed disagreement threshold: accept only if |Mistral-7B(Chunk-ID) - GPT-5.4(Chunk-ID)| <= tau

| tau (s) | coverage | escalated | Exact@30s (accepted) [95% CI] | median AE (accepted) | share >5 min (accepted) | wrong answers sent to human | right answers sent to human |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 0.505 | 0.495 | 0.415 [0.344, 0.484] | 100 | 0.434 | 0.551 | 0.389 |
| 10 | 0.612 | 0.388 | 0.405 [0.346, 0.464] | 105 | 0.420 | 0.446 | 0.278 |
| 30 | 0.726 | 0.274 | 0.397 [0.342, 0.451] | 105 | 0.413 | 0.333 | 0.160 |
| 60 | 0.812 | 0.188 | 0.384 [0.332, 0.437] | 120 | 0.431 | 0.239 | 0.090 |
| 120 | 0.852 | 0.148 | 0.377 [0.328, 0.426] | 128 | 0.430 | 0.192 | 0.062 |
| 300 | 0.900 | 0.100 | 0.362 [0.314, 0.411] | 155 | 0.434 | 0.127 | 0.049 |

## B. Risk-coverage: area under the curve (AURC, lower = better) and Exact@30s at fixed coverage

| policy | AURC | Exact@30s @50% | @70% | @90% |
|---|---:|---:|---:|---:|
| random abstention | 0.656 | 0.343 | 0.357 | 0.349 |
| rule: earlier candidate index | 0.585 | 0.370 | 0.387 | 0.362 |
| rule: Mistral vs GPT-5.4 agreement | 0.599 | 0.415 | 0.401 | 0.362 |
| rule: votes (other systems within 30 s) | 0.505 | 0.500 | 0.426 | 0.372 |
| learned (agreement features only) | 0.493 | 0.529 | 0.449 | 0.373 |
| learned (position in meeting only) | 0.464 | 0.490 | 0.446 | 0.378 |
| learned logistic head (all 9 features) | 0.413 | 0.571 | 0.459 | 0.376 |
| learned MLP head (all 9 features) | 0.433 | 0.533 | 0.449 | 0.370 |
| oracle (upper bound) | 0.291 | 0.686 | 0.490 | 0.381 |

Paired bootstrap of AURC difference vs. the disagreement rule (negative = better), 95% CI:

- learned_logreg: -0.183 [-0.227, -0.142]
- learned_mlp: -0.162 [-0.212, -0.116]
- learned_logreg_agreement_only: -0.102 [-0.146, -0.057]
- learned_logreg_time_only: -0.133 [-0.179, -0.083]
- votes_within_30s: -0.092 [-0.131, -0.055]

Paired bootstrap of AURC difference vs. the votes rule (negative = better), 95% CI:

- learned_logreg: -0.092 [-0.123, -0.059]
- learned_logreg_time_only: -0.042 [-0.088, +0.008]
- learned_logreg_agreement_only: -0.012 [-0.040, +0.016]

## C. Exact@30s by predicted position in the meeting (context for the 'position' feature)

| predicted time bin (s) | n | Exact@30s |
|---|---:|---:|
| 0 - 88 | 105 | 0.600 |
| 88 - 352 | 105 | 0.381 |
| 352 - 1296 | 105 | 0.324 |
| 1296 - 9554 | 105 | 0.067 |

Gold timestamp quartiles (s): [121.0, 394.0, 943.0]

## D. Single-feature AUC for 'served answer is within 30 s' (0.5 = no signal; <0.5 means larger value -> less likely correct)

| feature | AUC |
|---|---:|
| log1p_absdiff_vs_mis_cid | 0.392 |
| parsed_mis_cid | 0.502 |
| log1p_absdiff_vs_mis_free | 0.264 |
| parsed_mis_free | 0.548 |
| log1p_absdiff_vs_gpt_free | 0.263 |
| parsed_gpt_free | 0.504 |
| votes_within_30s | 0.729 |
| selected_candidate_index/15 | 0.415 |
| log1p_predicted_time_sec | 0.254 |

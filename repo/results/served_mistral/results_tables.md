# Results (served system = mis_cid, n = 420 queries)

No abstention: Exact@30s = 0.301, MAE = 761.0 s.

## A. Fixed disagreement threshold: accept only if |Mistral-7B(Chunk-ID) - GPT-5.4(Chunk-ID)| <= tau

| tau (s) | coverage | escalated | Exact@30s (accepted) [95% CI] | median AE (accepted) | share >5 min (accepted) | wrong answers sent to human | right answers sent to human |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 0.505 | 0.495 | 0.415 [0.344, 0.484] | 100 | 0.434 | 0.578 | 0.302 |
| 10 | 0.612 | 0.388 | 0.405 [0.346, 0.464] | 104 | 0.420 | 0.480 | 0.175 |
| 30 | 0.726 | 0.274 | 0.393 [0.338, 0.448] | 104 | 0.416 | 0.371 | 0.048 |
| 60 | 0.812 | 0.188 | 0.352 [0.303, 0.404] | 130 | 0.434 | 0.248 | 0.048 |
| 120 | 0.852 | 0.148 | 0.338 [0.290, 0.386] | 135 | 0.439 | 0.194 | 0.040 |
| 300 | 0.900 | 0.100 | 0.325 [0.279, 0.372] | 168 | 0.450 | 0.133 | 0.024 |

## B. Risk-coverage: area under the curve (AURC, lower = better) and Exact@30s at fixed coverage

| policy | AURC | Exact@30s @50% | @70% | @90% |
|---|---:|---:|---:|---:|
| random abstention | 0.707 | 0.290 | 0.310 | 0.302 |
| rule: earlier candidate index | 0.628 | 0.337 | 0.338 | 0.323 |
| rule: Mistral vs GPT-5.4 agreement | 0.608 | 0.415 | 0.398 | 0.325 |
| rule: votes (other systems within 30 s) | 0.529 | 0.458 | 0.389 | 0.328 |
| learned (agreement features only) | 0.524 | 0.471 | 0.388 | 0.325 |
| learned (position in meeting only) | 0.530 | 0.438 | 0.391 | 0.331 |
| learned logistic head (all 9 features) | 0.464 | 0.505 | 0.395 | 0.331 |
| learned MLP head (all 9 features) | 0.483 | 0.495 | 0.398 | 0.325 |
| oracle (upper bound) | 0.339 | 0.600 | 0.429 | 0.333 |

Paired bootstrap of AURC difference vs. the disagreement rule (negative = better), 95% CI:

- learned_logreg: -0.142 [-0.180, -0.109]
- learned_mlp: -0.122 [-0.164, -0.083]
- learned_logreg_agreement_only: -0.082 [-0.122, -0.043]
- learned_logreg_time_only: -0.076 [-0.117, -0.028]
- votes_within_30s: -0.077 [-0.111, -0.041]

Paired bootstrap of AURC difference vs. the votes rule (negative = better), 95% CI:

- learned_logreg: -0.066 [-0.093, -0.034]
- learned_logreg_time_only: +0.000 [-0.045, +0.055]
- learned_logreg_agreement_only: -0.006 [-0.033, +0.019]

## C. Exact@30s by predicted position in the meeting (context for the 'position' feature)

| predicted time bin (s) | n | Exact@30s |
|---|---:|---:|
| 0 - 90 | 105 | 0.524 |
| 90 - 314 | 104 | 0.346 |
| 314 - 1240 | 105 | 0.276 |
| 1240 - 10557 | 105 | 0.057 |

Gold timestamp quartiles (s): [121.0, 394.0, 943.0]

## D. Single-feature AUC for 'served answer is within 30 s' (0.5 = no signal; <0.5 means larger value -> less likely correct)

| feature | AUC |
|---|---:|
| log1p_absdiff_vs_gpt_cid | 0.321 |
| parsed_gpt_cid | 0.500 |
| log1p_absdiff_vs_mis_free | 0.279 |
| parsed_mis_free | 0.529 |
| log1p_absdiff_vs_gpt_free | 0.256 |
| parsed_gpt_free | 0.512 |
| votes_within_30s | 0.764 |
| selected_candidate_index/15 | 0.411 |
| log1p_predicted_time_sec | 0.267 |

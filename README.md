# Selective grounding: when should a constrained-selection pipeline escalate to a human?

A small, fully reproducible follow-up to *Topic-to-Timestamp Alignment by Constrained Evidence Selection*
(Yılbırt, Litvak, Färber, arXiv:2606.20890, 2026). The paper replaces free-form timestamp generation by
**selecting one retrieved candidate**, which reduces invalid outputs but leaves many boundary errors (Exact@30s = 34.3 % for GPT-5.4 on 420 queries).
The question here:

> If the pipeline may **abstain and escalate to a human**, which inference-time signals tell us when to do so,
> and what is the trade-off between **coverage** (answers given automatically) and **error** among the answers it does give?

This mirrors the "logic first, memory when ambiguous, human when no reliable decision is possible" design of verifiable
neuro-symbolic systems, but it is **only an analogy**: this repo studies one retrieval-and-selection task.

## What was done (and what was not)

- **Data.** Only the *sanitized per-query outputs released by the upstream authors* are used: gold timestamp, predicted timestamp and
  selected candidate ID for 420 queries (196 recordings) from four systems — Mistral-7B and GPT-5.4 with Chunk-ID selection (hybrid RRF retrieval,
  90 s/45 s windows), and Mistral-7B and GPT-5.4 with free-form timestamp generation (dense retrieval, 30 s/15 s windows).
- **Not done.** The upstream release contains no transcripts, audio or retrieval scores (annotation layer only), so the original pipeline was **not re-run**
  and no new model outputs were generated. There are no model confidence scores (logits) in the release; signals are therefore built from agreement between systems,
  the selected candidate index and the predicted position in the meeting.
- **Step 0 check.** `results/reproduction_of_upstream_numbers.json` re-derives Exact@30s, MAE, median AE and parsed counts from the per-query files;
  they match the paper (e.g. GPT-5.4 Chunk-ID: 34.3 % Exact@30s, MAE 743.4 s; Mistral-7B Chunk-ID: 30.0 %, MAE 761.0 s, 419/420 parsed).
- **Policies compared** (answer = the served system's Chunk-ID prediction; an answer is *correct* if within 30 s of gold, as in the paper):
  1. random abstention (chance reference);
  2. rules: Mistral-vs-GPT-5.4 disagreement, number of other systems that agree within 30 s ("votes"), earlier candidate index;
  3. a **learned deferral head in PyTorch** (logistic regression and a 16-unit MLP) predicting P(correct) from 9 inference-time features, evaluated
     **out-of-fold** (5 folds grouped by recording, 10 repeats), plus feature-group ablations;
  4. an oracle (accept all correct answers first) as an upper bound.
- **Metrics.** Risk–coverage curves, area under the curve (AURC, lower is better), Exact@30s among accepted answers, share of accepted answers with error > 5 min,
  share of wrong answers that reach a human, share of right answers needlessly escalated, bootstrap 95 % CIs.

## Headline results (served system: GPT-5.4 Chunk-ID, n = 420)

| policy | AURC ↓ | Exact@30s at 50 % coverage | at 70 % | at 90 % |
|---|---:|---:|---:|---:|
| no abstention (100 % coverage) | – | 0.343 | 0.343 | 0.343 |
| random abstention | 0.656 | 0.343 | 0.357 | 0.349 |
| rule: Mistral/GPT-5.4 agreement | 0.599 | 0.415 | 0.401 | 0.362 |
| rule: votes (others within 30 s) | 0.505 | 0.500 | 0.426 | 0.372 |
| learned head, agreement features only | 0.493 | 0.529 | 0.449 | 0.373 |
| learned head, position in meeting only | 0.464 | 0.490 | 0.446 | 0.378 |
| learned logistic head, all 9 features | **0.413** | **0.571** | 0.459 | 0.376 |
| learned MLP head, all 9 features | 0.433 | 0.533 | 0.449 | 0.370 |
| oracle (upper bound) | 0.291 | 0.686 | 0.490 | 0.381 |

Full tables and the figure: `results/results_tables.md`, `results/risk_coverage.png`. A robustness run with Mistral-7B as the served system is in `results/served_mistral/`
and gives the same overall picture (all-features logistic head best, AURC 0.464 vs. 0.707 for random and 0.339 for the oracle); the relative order of the single-signal policies differs between the two runs.

What the numbers say, and what they do not:

- Agreement between two selectors helps only modestly: accepting only exact agreement (coverage 50.5 %) raises Exact@30s from 34.3 % to 41.5 % (95 % CI 34.4–48.4 %).
  Even then 43 % of accepted answers are more than 5 minutes off, i.e. both selectors often make the **same** wrong choice.
- The **position of the predicted time in the meeting** carries as much signal as the agreement features (AUC 0.75 on its own): answers predicted in the first
  ~90 s are right 60 % of the time, those after ~21 min only 6.7 % (`results/results_tables.md`, section C). This is a property of this benchmark and may not transfer.
- The learned logistic head combines both and closes about two thirds of the gap between random (AURC 0.656) and oracle (0.291) abstention, at 0.413. The small MLP is not better than logistic regression.
- Against the simple votes rule, only the combined head is clearly better (ΔAURC −0.092, 95 % CI −0.123 to −0.059). The agreement-only head (−0.012 [−0.040, +0.016]) and the position-only head (−0.042 [−0.088, +0.008]) cannot be distinguished from it. The same holds in the Mistral run.
- A label-permutation sanity check gives AUC ≈ 0.50 against the permuted labels, so the cross-validation pipeline itself does not leak.

## Limitations (please read)

- Post-hoc analysis on the **same 420 queries** the authors report; there is no separate test set. Folds are grouped by recording, but the feature sets and ablations
  were chosen after looking at the data (see `RUNLOG.md`), so the learned-head numbers are optimistic.
- 420 queries, one benchmark, single-annotator gold labels (upstream limitation), municipal meetings only.
- Using a second model for agreement means **two or more neural calls per query**; this study measures escalation quality, not savings in model calls.
- Features exclude anything that needs transcript text or retrieval scores (not released), which probably leaves the strongest signals unused.
- Correct = within 30 s of the annotated start; other tolerances were not studied.

## Reproduce

```bash
bash run.sh        # venv + pinned requirements + clones upstream at commit 63ffb7a + both runs + tables + figure + tests (about 5 min on CPU, including the first PyTorch install)
```

Requires Python ≥ 3.10, `git` and internet access to PyPI and GitHub. To reuse an existing clone: `UPSTREAM_DIR=/path/to/clone bash run.sh`.
Output: `results/` (`summary.json`, `results_tables.md`, `risk_coverage.png`, `runlog.jsonl`, `single_feature_auc.json`, `reproduction_of_upstream_numbers.json`).

```
src/data.py           load + align the four per-query result files; re-derive errors and cross-check released MAE
src/features.py       inference-time features (gold is used only for labels)
src/metrics.py        risk-coverage, AURC, oracle
src/deferral.py       PyTorch logistic / MLP deferral head, grouped CV
src/run_experiment.py full experiment;  src/make_report.py tables + figure
tests/                unit tests for the metrics
RUNLOG.md             plain log of what was tried, incl. what failed
NOTICE.md             attribution and licences
```

## Attribution

Built on the artifact of Yılbırt, Litvak and Färber (code MIT, annotations/evaluation metadata CC BY 4.0). See `NOTICE.md`. Not affiliated with or endorsed by the upstream authors.

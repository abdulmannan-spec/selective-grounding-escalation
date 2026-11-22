# Run log (plain, chronological, including what failed)

Date of all work: 2026-11-22. Machine: 2 CPU cores, 7 GB RAM, no GPU, Python 3.13.

## 1. Reading
- Read the job ad and the published abstract/limitations of arXiv:2606.20890. The paper itself was read through a **fetch-tool summary of the arXiv PDF**
  (the sandbox shell cannot reach arxiv.org), so details quoted in the note come from that summary plus the upstream README/SUPPLEMENTARY, which were read in full.
- Cloned the upstream artifact (commit `63ffb7a`). Read README, SUPPLEMENTARY, prompts, configs, `eval.py` (candidate construction, prompts), result summaries.

## 2. What could not be done
- **Re-running the paper's pipeline: not possible here.** The release contains no transcripts, audio, chunks or vector DB (annotation layer only); the models
  (BAAI/bge-large-en-v1.5, Mistral-7B) need huggingface.co, which returned 403 from the sandbox; there is no GPU and no OpenAI access. Rebuilding transcripts would also mean downloading
  200 third-party recordings and running Whisper + pyannote, which the upstream README leaves to the user.
- Consequence: no new model outputs and no confidence scores (logits). The experiment was redesigned as a **post-hoc selective-prediction study on the released per-query outputs**.
- GitHub from the sandbox: `gh` token invalid; nothing was pushed from here.

## 3. Runs
| # | what | outcome |
|---|---|---|
| 0 | `pip install torch ...` (CPU box; CUDA wheels installed but unused) | OK, 2 min 17 s |
| 1 | `src/data.py`: load and align 4 result files; re-derive errors | all files aligned (420 queries, 196 recordings); re-derived MAE equals released `mae_sec`; Exact@30s/MAE/median/parsed **match the paper** (e.g. GPT-5.4 Chunk-ID 0.343 / 743.4 s; Mistral 0.300 / 761.0 s) |
| 2 | quick exploration of Mistral-vs-GPT disagreement thresholds | accepting exact agreement: coverage 0.505, Exact@30s 0.415 |
| 3 | first full run (rules + logistic/MLP head, 10 grouped-CV repeats) | learned logistic AURC 0.4155, MLP 0.4381, votes rule 0.505, random 0.656, oracle 0.291. **Sanity check looked broken**: a label-shuffled head scored AURC 0.82 (worse than chance 0.657) |
| 4 | diagnosis of run 3 sanity check | not a leak: the check scored the shuffled-label model against the *true* labels. A model fit on noise picks an arbitrary sign on the dominant feature (single-feature AUC of predicted time = 0.254), so against true labels it is strongly anti-correlated or correlated by luck. Fixed to score against the *permuted* labels: AUC 0.505 / 0.502 / 0.500 (chance) |
| 5 | **added after seeing run 3**: feature-group ablations (agreement-only, + candidate index, position-only) and single-feature AUCs | position alone (AURC 0.464) beat agreement features alone (0.481); combined best (0.413). This analysis choice was made post hoc |
| 6 | robustness: same pipeline with Mistral-7B Chunk-ID as the served system | all-features head best (0.464; random 0.707; oracle 0.339); order of single-signal policies differs from the GPT run (e.g. position-only 0.530 vs agreement-only 0.524) |
| 7 | clean-copy run of `run.sh` (fresh copy of the repo, existing upstream clone) | identical AURCs to the development run (determinism check), tests pass |
| 8 | **added after seeing runs 3–6**: paired bootstrap of the heads against the votes rule | only the combined head is clearly better than votes (ΔAURC −0.092 [−0.123, −0.059] GPT run; −0.066 [−0.093, −0.034] Mistral run); agreement-only and position-only heads are indistinguishable from it |
| 9 | final full `run.sh` (both served systems, tables, figure, 4 unit tests) | passed in about 2 min on CPU |

## 4. Things that went wrong or were cosmetic
- Run 3 sanity-check misreading (run 4).
- First figure had its legend on top of the oracle curve; moved below the panels.
- A first README draft said the Mistral run "shows the same ordering"; this was wrong (see run 6) and was corrected.

## 5. Known weaknesses (not fixed)
- Feature sets and ablations were chosen after looking at the same 420 queries; folds are grouped by recording but there is no untouched test set.
- No retrieval scores, transcript text or logits were available, so the strongest signals are probably unused.
- The position feature reflects this benchmark (early topics are easier) and may not transfer.

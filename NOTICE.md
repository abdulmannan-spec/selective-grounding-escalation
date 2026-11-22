# Attribution

This repository is an independent, post-hoc analysis of the artifact released with
*Topic-to-Timestamp Alignment by Constrained Evidence Selection* (Yılbırt, Litvak, Färber; arXiv:2606.20890, 2026).

- Upstream artifact: https://github.com/faerber-lab/topic-timestamp-alignment (pinned commit `63ffb7a`)
- Upstream code licence: MIT. Upstream data/annotation/evaluation-metadata licence: CC BY 4.0 (authors' original annotations only).
- This repository does **not** copy upstream files. `run.sh` clones the upstream repository at the pinned commit and reads only
  its sanitized per-query result summaries (`results/summaries/*.json`). Derived numbers and figures in `results/` are produced from
  those files; please credit the upstream authors when reusing them.
- No transcripts, audio or video are used or redistributed. This work is not affiliated with or endorsed by the upstream authors.

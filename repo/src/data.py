"""Load per-query outputs released with the upstream artifact
(faerber-lab/topic-timestamp-alignment) and align them by query.

Only the *sanitized per-query result summaries* are used: they contain the gold
timestamp, the model's predicted timestamp and the selected candidate ID, but no
transcript text. See README for licences and attribution.
"""
import json, os
import numpy as np

UPSTREAM_COMMIT = "63ffb7a"  # commit of the upstream repo these results were taken from

FILES = {
    "mis_cid":  "eval_ablation_90_45_mistral_chunk_id_final_sanitized.json",       # Mistral-7B, hybrid RRF, Chunk-ID
    "gpt_cid":  "eval_time_90_45_gpt54_hybrid_bm25_rrf_chunk_id_strict_sanitized.json",  # GPT-5.4, hybrid RRF, Chunk-ID
    "mis_free": "eval_time_30_15_mistral_baseline_dense_sanitized.json",          # Mistral-7B, dense, free-form timestamp
    "gpt_free": "eval_time_30_15_gpt54_baseline_dense_sanitized.json",            # GPT-5.4, dense, free-form timestamp
}

def hms(s):
    if s is None or s == "":
        return np.nan
    p = [int(x) for x in str(s).split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return p[0] * 3600 + p[1] * 60 + p[2]

def cid_index(c):
    """'C05' -> 5 ; None -> nan"""
    if c is None:
        return np.nan
    return float(int(str(c)[1:]))

def load(upstream_dir):
    R = os.path.join(upstream_dir, "results", "summaries")
    raw = {k: json.load(open(os.path.join(R, v)))["detailed_results_sanitized"] for k, v in FILES.items()}
    n = len(raw["mis_cid"])
    # alignment check: identical query order, gold and source in every file
    for k, rows in raw.items():
        assert len(rows) == n, (k, len(rows), n)
        for a, b in zip(raw["mis_cid"], rows):
            assert a["case_index"] == b["case_index"] and a["expected"] == b["expected"] and a["source_id"] == b["source_id"], k
    out = {
        "case_index": np.array([r["case_index"] for r in raw["mis_cid"]]),
        "source_id": np.array([r["source_id"] for r in raw["mis_cid"]]),
        "gold": np.array([hms(r["expected"]) for r in raw["mis_cid"]], dtype=float),
    }
    for k, rows in raw.items():
        out[f"pred_{k}"] = np.array([hms(r["predicted"]) for r in rows], dtype=float)
        out[f"sel_{k}"] = np.array([cid_index(r["selected_id"]) for r in rows], dtype=float)
        out[f"in_cand_{k}"] = np.array([r["timestamp_in_candidates"] if r["timestamp_in_candidates"] is not None else np.nan for r in rows], dtype=float)
        # re-derive the absolute error ourselves (and cross-check with the released mae_sec)
        err = np.abs(out[f"pred_{k}"] - out["gold"])
        rel = np.array([np.nan if r["mae_sec"] is None else r["mae_sec"] for r in rows], dtype=float)
        ok = ~np.isnan(err) & ~np.isnan(rel)
        assert np.allclose(err[ok], rel[ok]), f"mae mismatch in {k}"
        out[f"err_{k}"] = err
    return out

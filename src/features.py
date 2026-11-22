"""Inference-time features for deciding whether to accept a served answer or escalate.

Every feature is computed from system outputs only (predicted timestamps, selected
candidate indices, parse success). The gold timestamp is used ONLY to build the label.
"""
import numpy as np

SYSTEMS = ["mis_cid", "gpt_cid", "mis_free", "gpt_free"]
BIG = np.log1p(7200.0)  # stand-in for 'no prediction to compare with' (2 h)

def _ld(a, b):
    d = np.abs(a - b)
    return np.where(np.isnan(d), BIG, np.log1p(d))

def build(d, served):
    """Return X (N x F), feature names for the served system ('gpt_cid' or 'mis_cid')."""
    others = [s for s in SYSTEMS if s != served]
    A = d[f"pred_{served}"]
    cols, names = [], []
    for o in others:
        cols.append(_ld(A, d[f"pred_{o}"])); names.append(f"log1p_absdiff_vs_{o}")
        cols.append((~np.isnan(d[f"pred_{o}"])).astype(float)); names.append(f"parsed_{o}")
    votes = np.zeros_like(A)
    for o in others:
        diff = np.abs(A - d[f"pred_{o}"])
        votes += np.where(np.isnan(diff), 0.0, (diff <= 30.0).astype(float))
    cols.append(votes); names.append("votes_within_30s")
    cols.append(np.nan_to_num(d[f"sel_{served}"], nan=0.0) / 15.0); names.append("selected_candidate_index/15")
    cols.append(np.log1p(np.nan_to_num(A, nan=0.0))); names.append("log1p_predicted_time_sec")
    return np.stack(cols, axis=1).astype(np.float32), names

def votes(d, served):
    X, n = build(d, served)
    return X[:, n.index("votes_within_30s")]

def disagreement_cid(d):
    """|Mistral(Chunk-ID) - GPT-5.4(Chunk-ID)| in seconds (NaN if either is unparsed)."""
    return np.abs(d["pred_mis_cid"] - d["pred_gpt_cid"])

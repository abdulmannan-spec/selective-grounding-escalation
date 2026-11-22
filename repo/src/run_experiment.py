"""Run the full selective-grounding / escalation experiment.

    python -m src.run_experiment --upstream <path to upstream clone>

Steps (each logged to results/runlog.jsonl):
  0. reproduce the upstream aggregate numbers from the released per-query outputs
  1. rule-based escalation (disagreement thresholds, vote count, candidate index)
  2. learned deferral head (PyTorch logistic regression / small MLP), grouped CV
  3. risk-coverage tables, bootstrap CIs, figure
"""
import argparse, json, os, time, platform
import numpy as np, torch
from . import data, features as F, metrics as M, deferral as D

p = argparse.ArgumentParser()
p.add_argument("--upstream", required=True)
p.add_argument("--out", default="results")
p.add_argument("--served", default="gpt_cid", choices=["gpt_cid", "mis_cid"])
p.add_argument("--repeats", type=int, default=10)
p.add_argument("--boot", type=int, default=2000)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
LOG = open(os.path.join(args.out, "runlog.jsonl"), "w")
def log(step, **kw):
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, **kw}
    LOG.write(json.dumps(rec, default=float) + "\n"); LOG.flush()
    print(step, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()})

d = data.load(args.upstream)
N = len(d["gold"])
log("env", python=platform.python_version(), torch=torch.__version__, cuda=torch.cuda.is_available(), n_queries=N,
    n_sources=int(len(set(d["source_id"]))), upstream_commit=data.UPSTREAM_COMMIT, served=args.served)

# ---- 0. reproduce upstream aggregates --------------------------------------------------------
repro = {}
for k in F.SYSTEMS:
    e = d[f"err_{k}"]
    repro[k] = dict(parsed=int((~np.isnan(e)).sum()), exact30=float(M.correct(e).mean()), mae=float(np.nanmean(e)),
                    median_ae=float(np.nanmedian(e)))
    log("reproduce", system=k, **repro[k])
json.dump(repro, open(os.path.join(args.out, "reproduction_of_upstream_numbers.json"), "w"), indent=1)

served = args.served
err = d[f"err_{served}"]; parsed = ~np.isnan(err)
y = M.correct(err).astype(np.float32)
groups = d["source_id"]
base_acc = float(y[parsed].mean()); base_risk = 1 - base_acc

# ---- 1. rule-based escalation ----------------------------------------------------------------
rules = {}
dis = F.disagreement_cid(d)
Xs, names = F.build(d, served)
rules["agree_mis_vs_gpt_chunkid"] = np.where(np.isnan(dis), -1e9, -dis)           # smaller disagreement = more confident
rules["votes_within_30s"] = F.votes(d, served) + 0.0                              # 0..3 other systems within 30 s
rules["selected_candidate_index"] = -np.nan_to_num(d[f"sel_{served}"], nan=99)     # earlier candidate = more confident (the prompt prefers early ones)
rng = np.random.default_rng(0)
rules["random"] = rng.random(N)                                                    # chance-level reference

curves = {}
for name, s in rules.items():
    cov, risk, mae, sev = M.risk_coverage(s, err, parsed)
    curves[name] = (cov, risk, mae, sev)
    log("rule", name=name, aurc=M.aurc(cov, risk), risk_at_50=M.at_coverage(cov, risk, .5), risk_at_70=M.at_coverage(cov, risk, .7))

# fixed thresholds on the disagreement rule (the simplest deployable policy)
thr_rows = []
for tau in [0, 10, 30, 60, 120, 300]:
    acc_mask = parsed & ~np.isnan(dis) & (dis <= tau)
    row = dict(tau_sec=tau, coverage=float(acc_mask.mean()), escalation_rate=float(1 - acc_mask.mean()),
               exact30_accepted=float(y[acc_mask].mean()), mae_accepted=float(np.mean(err[acc_mask])),
               median_ae_accepted=float(np.median(err[acc_mask])), severe_rate_accepted=float((err[acc_mask] > M.SEVERE).mean()),
               wrong_caught=float(((y == 0) & ~acc_mask).sum() / max((y == 0).sum(), 1)),      # share of wrong answers sent to a human
               right_escalated=float(((y == 1) & ~acc_mask).sum() / max((y == 1).sum(), 1)))   # share of right answers needlessly escalated
    # bootstrap CI over queries
    brng = np.random.default_rng(1); accs, covs = [], []
    for _ in range(args.boot):
        i = brng.integers(0, N, N); m = acc_mask[i]
        covs.append(m.mean()); accs.append(y[i][m].mean() if m.any() else np.nan)
    row["exact30_accepted_ci95"] = [float(np.nanpercentile(accs, 2.5)), float(np.nanpercentile(accs, 97.5))]
    row["coverage_ci95"] = [float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))]
    thr_rows.append(row); log("threshold", **{k: v for k, v in row.items() if not isinstance(v, list)})

# ---- 2. learned deferral head (PyTorch) + feature ablations -----------------------------------
from sklearn.metrics import roc_auc_score
feat_auc = {n: float(roc_auc_score(y, Xs[:, i])) for i, n in enumerate(names)}   # single-feature AUC for "answer is correct"
json.dump(feat_auc, open(os.path.join(args.out, "single_feature_auc.json"), "w"), indent=1)
log("single_feature_auc", **feat_auc)
TIME = "log1p_predicted_time_sec"; CAND = "selected_candidate_index/15"
subsets = {
    "all": names,
    "agreement_only": [n for n in names if n not in (TIME, CAND)],
    "agreement_plus_candidate_index": [n for n in names if n != TIME],
    "time_only": [TIME],
}
oof = {}
runs = [("logreg", "all"), ("mlp", "all"), ("logreg", "agreement_only"), ("logreg", "agreement_plus_candidate_index"), ("logreg", "time_only")]
for kind, sub in runs:
    t0 = time.time()
    cols = [names.index(n) for n in subsets[sub]]
    sc = D.oof_scores(Xs[:, cols], y, groups, kind, seeds=range(args.repeats))
    key = f"learned_{kind}" if sub == "all" else f"learned_{kind}_{sub}"
    oof[key] = sc
    per_rep = [M.aurc(*M.risk_coverage(sc[r], err, parsed, n_shuffle=1)[:2]) for r in range(sc.shape[0])]
    auc_rep = [float(roc_auc_score(y, sc[r])) for r in range(sc.shape[0])]
    log("learned_deferral", model=kind, subset=sub, n_features=len(cols), aurc_mean=float(np.mean(per_rep)), aurc_std=float(np.std(per_rep)),
        auc_mean=float(np.mean(auc_rep)), auc_std=float(np.std(auc_rep)), seconds=time.time() - t0)
    curves[key] = M.risk_coverage(sc.mean(0), err, parsed)
# pipeline sanity: train on label-permuted data and score against the SAME permuted labels -> AUC must be ~0.5
yp = np.random.default_rng(5).permutation(y).astype(np.float32)
shuf = D.oof_scores(Xs, yp, groups, "logreg", seeds=range(3))
log("sanity_label_shuffle", auc_vs_permuted_labels=[float(roc_auc_score(yp, s)) for s in shuf], expected="~0.5")

oc = M.oracle_curve(err, parsed); curves["oracle"] = oc

# ---- 3. summary tables -----------------------------------------------------------------------
summary = dict(served=served, n=N, accuracy_exact30=base_acc, risk_no_abstention=base_risk,
               mae_no_abstention=float(np.nanmean(err)), thresholds=thr_rows, aurc={}, at_coverage={})
for name, (cov, risk, mae, sev) in curves.items():
    summary["aurc"][name] = M.aurc(cov, risk)
    summary["at_coverage"][name] = {str(c): dict(risk=M.at_coverage(cov, risk, c), exact30=1 - M.at_coverage(cov, risk, c),
                                                 mae=M.at_coverage(cov, mae, c), severe=M.at_coverage(cov, sev, c))
                                    for c in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] if c <= cov.max() + 1e-9}
# paired bootstrap of AURC differences vs. the best rule
def aurc_of(score, idx):
    return M.aurc(*M.risk_coverage(score[idx], err[idx], parsed[idx], n_shuffle=20, seed=3)[:2])
brng = np.random.default_rng(2); diffs = {"learned_logreg": [], "learned_mlp": [], "learned_logreg_agreement_only": [], "learned_logreg_time_only": [], "votes_within_30s": []}
sc_ref = rules["agree_mis_vs_gpt_chunkid"]
for _ in range(min(args.boot, 500)):
    i = brng.integers(0, N, N)
    a_ref = aurc_of(sc_ref, i)
    diffs["learned_logreg"].append(aurc_of(oof["learned_logreg"].mean(0), i) - a_ref)
    diffs["learned_mlp"].append(aurc_of(oof["learned_mlp"].mean(0), i) - a_ref)
    diffs["learned_logreg_agreement_only"].append(aurc_of(oof["learned_logreg_agreement_only"].mean(0), i) - a_ref)
    diffs["learned_logreg_time_only"].append(aurc_of(oof["learned_logreg_time_only"].mean(0), i) - a_ref)
    diffs["votes_within_30s"].append(aurc_of(rules["votes_within_30s"].astype(float), i) - a_ref)
# added after the first full run (see RUNLOG.md): paired bootstrap of the strongest policies against the simple votes rule
brng = np.random.default_rng(4); d2 = {"learned_logreg": [], "learned_logreg_time_only": [], "learned_logreg_agreement_only": []}
sc_votes = rules["votes_within_30s"].astype(float)
for _ in range(min(args.boot, 500)):
    i = brng.integers(0, N, N); a_ref = aurc_of(sc_votes, i)
    for k in d2: d2[k].append(aurc_of(oof[k].mean(0), i) - a_ref)
summary["aurc_diff_vs_votes_rule"] = {k: dict(mean=float(np.mean(v)), ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]) for k, v in d2.items()}
summary["aurc_diff_vs_disagreement_rule"] = {k: dict(mean=float(np.mean(v)), ci95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]) for k, v in diffs.items()}
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
np.savez(os.path.join(args.out, "curves.npz"), **{f"{k}__{j}": curves[k][i] for k in curves for i, j in enumerate(["cov", "risk", "mae", "severe"])})
log("done", aurc={k: round(v, 4) for k, v in summary["aurc"].items()}, diff_vs_rule=summary["aurc_diff_vs_disagreement_rule"])

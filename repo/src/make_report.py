"""Build results/results_tables.md and results/risk_coverage.png from a finished run."""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import data, features as F, metrics as M

p = argparse.ArgumentParser(); p.add_argument("--upstream", required=True); p.add_argument("--out", default="results")
p.add_argument("--served", default="gpt_cid"); a = p.parse_args()
S = json.load(open(os.path.join(a.out, "summary.json")))
C = np.load(os.path.join(a.out, "curves.npz"))
cur = lambda k: tuple(C[f"{k}__{j}"] for j in ["cov", "risk", "mae", "severe"])
LABEL = {"random": "random abstention", "agree_mis_vs_gpt_chunkid": "rule: Mistral vs GPT-5.4 agreement",
         "votes_within_30s": "rule: votes (other systems within 30 s)", "selected_candidate_index": "rule: earlier candidate index",
         "learned_logreg_agreement_only": "learned (agreement features only)", "learned_logreg_time_only": "learned (position in meeting only)",
         "learned_logreg": "learned logistic head (all 9 features)", "learned_mlp": "learned MLP head (all 9 features)", "oracle": "oracle (upper bound)"}
order = ["random", "selected_candidate_index", "agree_mis_vs_gpt_chunkid", "votes_within_30s", "learned_logreg_agreement_only",
         "learned_logreg_time_only", "learned_logreg", "learned_mlp", "oracle"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sty = {"random": ("0.55", ":"), "oracle": ("k", "--")}
for k in order:
    cov, risk, mae, sev = cur(k); col, ls = sty.get(k, (None, "-"))
    ax[0].plot(cov, 1 - risk, label=LABEL[k], color=col, ls=ls, lw=1.6)
    ax[1].plot(cov, sev, label=LABEL[k], color=col, ls=ls, lw=1.6)
ax[0].set_xlabel("coverage (share of queries answered automatically)"); ax[0].set_ylabel("Exact@30s among answered")
ax[1].set_xlabel("coverage"); ax[1].set_ylabel("share of answered with error > 5 min")
ax[0].set_title("Accuracy of accepted answers"); ax[1].set_title("Severe errors among accepted answers")
for x in ax: x.grid(alpha=.25)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
plt.suptitle(f"Selective answering vs. escalation to a human (served system: {S['served']}, n={S['n']})", fontsize=10)
plt.tight_layout(rect=(0, 0.12, 1, 0.95)); plt.savefig(os.path.join(a.out, "risk_coverage.png"), dpi=170); plt.close()

L = []
L.append(f"# Results (served system = {S['served']}, n = {S['n']} queries)\n")
L.append(f"No abstention: Exact@30s = {S['accuracy_exact30']:.3f}, MAE = {S['mae_no_abstention']:.1f} s.\n")
L.append("## A. Fixed disagreement threshold: accept only if |Mistral-7B(Chunk-ID) - GPT-5.4(Chunk-ID)| <= tau\n")
L.append("| tau (s) | coverage | escalated | Exact@30s (accepted) [95% CI] | median AE (accepted) | share >5 min (accepted) | wrong answers sent to human | right answers sent to human |")
L.append("|---:|---:|---:|---|---:|---:|---:|---:|")
for r in S["thresholds"]:
    L.append(f"| {r['tau_sec']} | {r['coverage']:.3f} | {r['escalation_rate']:.3f} | {r['exact30_accepted']:.3f} [{r['exact30_accepted_ci95'][0]:.3f}, {r['exact30_accepted_ci95'][1]:.3f}] | {r['median_ae_accepted']:.0f} | {r['severe_rate_accepted']:.3f} | {r['wrong_caught']:.3f} | {r['right_escalated']:.3f} |")
L.append("\n## B. Risk-coverage: area under the curve (AURC, lower = better) and Exact@30s at fixed coverage\n")
L.append("| policy | AURC | Exact@30s @50% | @70% | @90% |")
L.append("|---|---:|---:|---:|---:|")
for k in order:
    ac = S["at_coverage"][k]
    g = lambda c: f"{ac[c]['exact30']:.3f}" if c in ac else "-"
    L.append(f"| {LABEL[k]} | {S['aurc'][k]:.3f} | {g('0.5')} | {g('0.7')} | {g('0.9')} |")
L.append("\nPaired bootstrap of AURC difference vs. the disagreement rule (negative = better), 95% CI:\n")
for k, v in S["aurc_diff_vs_disagreement_rule"].items():
    L.append(f"- {k}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")

L.append("\nPaired bootstrap of AURC difference vs. the votes rule (negative = better), 95% CI:\n")
for k, v in S["aurc_diff_vs_votes_rule"].items():
    L.append(f"- {k}: {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")

d = data.load(a.upstream)
t = d[f"pred_{a.served}"]; ok = ~np.isnan(t); y = M.correct(d[f"err_{a.served}"])
qs = np.nanpercentile(t, [0, 25, 50, 75, 100])
L.append("\n## C. Exact@30s by predicted position in the meeting (context for the 'position' feature)\n")
L.append("| predicted time bin (s) | n | Exact@30s |"); L.append("|---|---:|---:|")
for lo, hi in zip(qs[:-1], qs[1:]):
    m = ok & (t >= lo) & (t <= hi if hi == qs[-1] else t < hi)
    L.append(f"| {lo:.0f} - {hi:.0f} | {int(m.sum())} | {y[m].mean():.3f} |")
L.append(f"\nGold timestamp quartiles (s): {np.percentile(d['gold'], [25, 50, 75]).round(0).tolist()}")
fa = json.load(open(os.path.join(a.out, "single_feature_auc.json")))
L.append("\n## D. Single-feature AUC for 'served answer is within 30 s' (0.5 = no signal; <0.5 means larger value -> less likely correct)\n")
L.append("| feature | AUC |"); L.append("|---|---:|")
for k, v in fa.items(): L.append(f"| {k} | {v:.3f} |")
open(os.path.join(a.out, "results_tables.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

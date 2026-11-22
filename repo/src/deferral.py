"""Learned deferral head in PyTorch: predict P(answer within 30 s of gold) from the
inference-time features, evaluated out-of-fold with folds grouped by recording."""
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold

torch.set_num_threads(2)

def make_model(kind, n_in, hidden=16):
    if kind == "logreg":
        return nn.Linear(n_in, 1)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
    raise ValueError(kind)

def fit_predict(Xtr, ytr, Xte, kind, seed, epochs=300, lr=0.02, wd=1e-3):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_t = torch.tensor((Xtr - mu) / sd); Xte_t = torch.tensor((Xte - mu) / sd)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).unsqueeze(1)
    m = make_model(kind, Xtr.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    # class weighting: correct answers are the minority (~1/3)
    pos = float(ytr.mean()); w = torch.tensor([(1 - pos) / max(pos, 1e-6)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=w)
    m.train()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(m(Xtr_t), ytr_t); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(Xte_t)).squeeze(1).numpy()

def oof_scores(X, y, groups, kind, seeds=range(10), n_splits=5):
    """One out-of-fold score per query per repeat; returns array (n_repeats, N)."""
    out = np.zeros((len(list(seeds)), len(y)))
    for r, s in enumerate(seeds):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s)
        for tr, te in cv.split(X, y, groups):
            out[r, te] = fit_predict(X[tr], y[tr], X[te], kind, seed=s)
    return out

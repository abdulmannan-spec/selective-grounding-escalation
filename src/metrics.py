import numpy as np

TOL = 30.0      # "correct" = absolute timestamp error <= 30 s (the paper's Exact@30s)
SEVERE = 300.0  # "severe error" = more than 5 minutes off (our own, reported separately)

def correct(err, tol=TOL):
    """err may contain NaN (unparsed output) -> counted as not correct."""
    return np.where(np.isnan(err), False, err <= tol)

def risk_coverage(score, err, parsed, n_shuffle=200, seed=0):
    """Selective-prediction curve with random tie-breaking.

    score  : higher = more confident (answer accepted first)
    err    : absolute error in seconds (NaN where the output could not be parsed)
    parsed : False -> the answer can never be accepted (always escalated to a human)
    Returns coverage[k], risk[k] (= error rate among accepted, 1-Exact@30s),
            mae[k], severe[k] for k = 1..N accepted answers; averaged over tie-break shuffles.
    Unparsed outputs are placed last so they are only 'accepted' once every parsed answer is.
    """
    rng = np.random.default_rng(seed)
    N = len(score)
    wrong = (~correct(err)).astype(float)
    e = np.where(np.isnan(err), 0.0, err)
    sev = np.where(np.isnan(err), 0.0, (err > SEVERE).astype(float))
    s = np.where(parsed, score, -np.inf)
    risk = np.zeros(N); mae = np.zeros(N); sevr = np.zeros(N)
    k = np.arange(1, N + 1)
    for _ in range(n_shuffle):
        tie = rng.random(N)
        order = np.lexsort((tie, -s))        # primary: -score ascending, then random tie-break
        risk += np.cumsum(wrong[order]) / k
        mae += np.cumsum(e[order]) / k
        sevr += np.cumsum(sev[order]) / k
    n_parsed = int(parsed.sum())
    cov = k / N
    # coverage beyond the number of parsed answers is not meaningful: mask it
    valid = k <= n_parsed
    return cov[valid], (risk / n_shuffle)[valid], (mae / n_shuffle)[valid], (sevr / n_shuffle)[valid]

def aurc(cov, risk):
    """Area under the risk-coverage curve (mean selective risk over k = 1..K accepted)."""
    return float(np.mean(risk))

def at_coverage(cov, arr, c):
    i = int(np.argmin(np.abs(cov - c)))
    return float(arr[i])

def oracle_curve(err, parsed):
    """Best possible ordering: accept every correct answer first (risk 0 until all are used)."""
    wrong = (~correct(err)).astype(float)
    s = np.where(parsed, -wrong, -np.inf)
    return risk_coverage(s, err, parsed, n_shuffle=1)

import numpy as np
from src import metrics as M

def test_correct_counts_nan_as_wrong():
    e = np.array([0, 30, 31, np.nan])
    assert M.correct(e).tolist() == [True, True, False, False]

def test_oracle_has_zero_risk_until_accuracy():
    err = np.array([5, 500, 10, 900, 20.0]); parsed = np.ones(5, bool)
    cov, risk, *_ = M.oracle_curve(err, parsed)
    assert np.allclose(risk[:3], 0) and risk[-1] == 2 / 5

def test_random_scores_give_base_risk():
    rng = np.random.default_rng(0); err = np.where(rng.random(2000) < .3, 5.0, 500.0)
    cov, risk, *_ = M.risk_coverage(rng.random(2000), err, np.ones(2000, bool))
    assert abs(M.aurc(cov, risk) - 0.7) < 0.03

def test_unparsed_never_accepted_early():
    err = np.array([5.0, np.nan, 5.0]); parsed = np.array([True, False, True])
    cov, risk, *_ = M.risk_coverage(np.array([1., 9., 1.]), err, parsed)
    assert len(cov) == 2 and np.allclose(risk, 0)

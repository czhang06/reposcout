from reposcout.evals.compare import paired_bootstrap


def test_clear_win_has_interval_above_zero():
    r = paired_bootstrap([0.0] * 40, [1.0] * 30 + [0.0] * 10)
    assert r["ci95"][0] > 0 and r["wins"] == 30 and r["losses"] == 0


def test_noise_interval_spans_zero():
    a = [1, 0] * 20
    b = [0, 1] * 20
    r = paired_bootstrap(a, b)
    assert r["ci95"][0] < 0 < r["ci95"][1]

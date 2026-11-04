import statistics

import pytest

import limno as lm


def test_mean_matches_statistics():
    values = [1.0, -2.5, 4.0, 0.5, 7.25]
    tracker = lm.tracker.Mean()
    assert tracker.get() == 0.0
    for value in values:
        tracker.update(value)
    assert tracker.get() == pytest.approx(statistics.fmean(values))


def test_ewmean_starts_without_bias():
    tracker = lm.tracker.EWMean(alpha=0.001)
    assert tracker.get() == 0.0
    tracker.update(5.0)
    assert tracker.get() == pytest.approx(5.0)
    for _ in range(10):
        tracker.update(5.0)
    assert tracker.get() == pytest.approx(5.0)


def test_ewmean_is_close_to_a_plain_mean_at_the_start():
    tracker = lm.tracker.EWMean(alpha=0.001)
    tracker.update(0.0)
    tracker.update(10.0)
    assert tracker.get() == pytest.approx(5.0, abs=0.01)


def test_ewmean_becomes_the_usual_update():
    alpha = 0.1
    tracker = lm.tracker.EWMean(alpha=alpha)
    usual = 0.0
    for i in range(500):
        value = float(i % 7)
        tracker.update(value)
        usual = (1 - alpha) * usual + alpha * value
    assert tracker.get() == pytest.approx(usual)


def test_ewmean_follows_a_change_and_mean_does_not():
    ewmean, mean = lm.tracker.EWMean(alpha=0.01), lm.tracker.Mean()
    for value in [0.0] * 2000 + [1.0] * 1000:
        ewmean.update(value)
        mean.update(value)
    assert ewmean.get() == pytest.approx(1.0, abs=0.01)
    assert mean.get() == pytest.approx(1 / 3)


@pytest.mark.parametrize("alpha", [0, -0.1, 1.5])
def test_ewmean_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha must be above 0 and at most 1"):
        lm.tracker.EWMean(alpha=alpha)

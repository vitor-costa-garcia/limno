import random

import pytest

import limno as lm


class AbsoluteError:
    """A minimal River-style metric."""

    bigger_is_better = False

    def __init__(self):
        self.total = 0.0
        self.n = 0

    def update(self, y_true, y_pred):
        self.total += abs(y_true - y_pred)
        self.n += 1

    def revert(self, y_true, y_pred):
        self.total -= abs(y_true - y_pred)
        self.n -= 1

    def get(self):
        return self.total / self.n

    def clone(self):
        return AbsoluteError()


class Correct(AbsoluteError):
    """A metric for which bigger is better."""

    bigger_is_better = True

    def get(self):
        return 1 - super().get()

    def clone(self):
        return Correct()


class Fixed:
    """A model that only uses `a`, and so has nothing to learn."""

    def predict_one(self, x):
        return 2 * x.get("a", 0.0)


def make_stream(n=20000, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        x = {"a": rng.random(), "b": rng.random()}
        yield x, 2 * x["a"]


def static_pfi(metric=None, **kwargs):
    return lm.importance.PFI(
        Fixed(),
        metric or AbsoluteError(),
        imputer=lm.imputer.Marginal(lm.storage.UniformReservoir(size=500, seed=1), seed=2),
        tracker=lm.tracker.Mean(),
        **kwargs,
    )


def test_matches_the_true_value():
    # The loss is 0 with `a`, and 2 |a - a'| without it, whose mean is 2 / 3 for uniform values
    explainer = static_pfi()
    for x, y in make_stream():
        explainer.learn_one(x, y)
    assert explainer.importances["a"] == pytest.approx(2 / 3, abs=0.02)
    assert explainer.importances["b"] == 0.0


def test_bigger_is_better_metric_keeps_the_sign():
    explainer = static_pfi(metric=Correct())
    for x, y in make_stream(2000):
        explainer.learn_one(x, y)
    assert explainer.importances["a"] > 0.5


def test_more_samples_give_less_noise():
    def error(n_samples):
        errors = []
        for seed in range(30):
            explainer = lm.importance.PFI(
                Fixed(),
                AbsoluteError(),
                imputer=lm.imputer.Marginal(
                    lm.storage.UniformReservoir(size=100, seed=seed), seed=seed
                ),
                tracker=lm.tracker.Mean(),
                n_samples=n_samples,
            )
            for x, y in make_stream(150, seed=seed):
                explainer.learn_one(x, y)
            errors.append(abs(explainer.importances["a"] - 2 / 3))
        return sum(errors) / len(errors)

    assert error(n_samples=10) < error(n_samples=1)


def test_number_of_predictions():
    class Counting(Fixed):
        n = 0

        def predict_one(self, x):
            self.n += 1
            return super().predict_one(x)

    model = Counting()
    explainer = lm.importance.PFI(model, AbsoluteError(), n_samples=3, seed=0)
    explainer.learn_one({"a": 1.0, "b": 1.0}, 2.0)
    assert model.n == 0  # no past instance yet
    explainer.learn_one({"a": 1.0, "b": 1.0}, 2.0)
    assert model.n == 1 + 2 * 3


def test_no_importances_before_a_past_instance():
    explainer = lm.importance.PFI(Fixed(), AbsoluteError(), seed=0)
    assert explainer.importances == {}
    explainer.learn_one({"a": 1.0}, 2.0)
    assert explainer.importances == {}
    explainer.learn_one({"a": 1.0}, 2.0)
    assert set(explainer.importances) == {"a"}


def test_instance_is_stored_after_it_is_explained():
    explainer = static_pfi()
    explainer.learn_one({"a": 1.0, "b": 0.0}, 2.0)
    explainer.learn_one({"a": 1.0, "b": 0.0}, 2.0)
    assert explainer.importances["a"] == 0.0
    assert len(explainer.imputer.storage) == 2


def test_features_can_appear_and_disappear():
    explainer = lm.importance.PFI(Fixed(), AbsoluteError(), seed=0)
    explainer.learn_one({"a": 1.0}, 2.0)
    explainer.learn_one({"a": 0.0, "new": 1.0}, 0.0)
    assert set(explainer.importances) == {"a", "new"}
    explainer.learn_one({"new": 1.0}, 0.0)
    assert set(explainer.importances) == {"a", "new"}


@pytest.mark.parametrize("value", [None, {}])
def test_model_that_cannot_predict_yet_is_skipped(value):
    class Untrained:
        def predict_one(self, x):
            return value

    explainer = lm.importance.PFI(Untrained(), AbsoluteError(), seed=0)
    for _ in range(3):
        explainer.learn_one({"a": 1.0}, 2.0)
    assert explainer.importances == {}
    assert len(explainer.imputer.storage) == 3


def test_metric_given_is_left_alone():
    metric = AbsoluteError()
    metric.update(0.0, 10.0)
    explainer = lm.importance.PFI(Fixed(), metric, seed=0)
    for x, y in make_stream(10):
        explainer.learn_one(x, y)
    assert metric.n == 1
    assert explainer.metric.n == 0


def test_seed_makes_it_reproducible():
    def run(seed):
        explainer = lm.importance.PFI(Fixed(), AbsoluteError(), seed=seed)
        for x, y in make_stream(500):
            explainer.learn_one(x, y)
        return explainer.importances

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_seed_with_own_imputer_is_rejected():
    with pytest.raises(ValueError, match="seed only applies to the default imputer"):
        lm.importance.PFI(Fixed(), AbsoluteError(), imputer=lm.imputer.Marginal(), seed=1)


def test_bad_n_samples():
    with pytest.raises(ValueError, match="n_samples must be at least 1, got 0"):
        lm.importance.PFI(Fixed(), AbsoluteError(), n_samples=0)


def test_bad_metric():
    with pytest.raises(TypeError, match="metric must have.*got object"):
        lm.importance.PFI(Fixed(), object())


def test_metric_that_needs_probabilities_needs_predict_proba_one():
    class Probabilities(AbsoluteError):
        requires_labels = False

    with pytest.raises(TypeError, match="Probabilities needs probabilities.*Fixed"):
        lm.importance.PFI(Fixed(), Probabilities())

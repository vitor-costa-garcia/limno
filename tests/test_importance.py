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


def test_feature_is_hidden_after_max_age():
    explainer = lm.importance.PFI(Fixed(), AbsoluteError(), max_age=2, seed=0)
    for _ in range(2):
        explainer.learn_one({"a": 1.0, "old": 1.0}, 2.0)
    explainer.learn_one({"a": 1.0}, 2.0)
    assert set(explainer.importances) == {"a", "old"}
    explainer.learn_one({"a": 1.0}, 2.0)
    assert set(explainer.importances) == {"a"}


def test_hidden_feature_comes_back_with_its_old_value():
    explainer = lm.importance.PFI(
        Fixed(),
        AbsoluteError(),
        imputer=lm.imputer.Marginal(lm.storage.GeometricReservoir(size=1, seed=0), seed=0),
        tracker=lm.tracker.Mean(),
        max_age=2,
    )
    explainer.learn_one({"a": 0.0}, 0.0)
    explainer.learn_one({"a": 1.0}, 2.0)
    assert explainer.importances["a"] == 2.0
    for _ in range(2):
        explainer.learn_one({"b": 1.0}, 0.0)
    assert "a" not in explainer.importances

    # The past instance has no `a`, so the prediction is 0 and this value is 1
    explainer.learn_one({"a": 0.5}, 1.0)
    assert explainer.importances["a"] == 1.5


def test_default_max_age():
    explainer = lm.importance.PFI(Fixed(), AbsoluteError(), seed=0)
    for _ in range(2):
        explainer.learn_one({"old": 1.0}, 0.0)
    for _ in range(1999):
        explainer.learn_one({"a": 1.0}, 2.0)
    assert "old" in explainer.importances
    explainer.learn_one({"a": 1.0}, 2.0)
    assert "old" not in explainer.importances


def test_no_max_age_never_hides_a_feature():
    explainer = lm.importance.PFI(Fixed(), AbsoluteError(), max_age=None, seed=0)
    for _ in range(2):
        explainer.learn_one({"old": 1.0}, 0.0)
    for _ in range(3000):
        explainer.learn_one({"a": 1.0}, 2.0)
    assert "old" in explainer.importances


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


def test_bad_max_age():
    with pytest.raises(ValueError, match="max_age must be at least 1 or None, got 0"):
        lm.importance.PFI(Fixed(), AbsoluteError(), max_age=0)


def test_bad_metric():
    with pytest.raises(TypeError, match="metric must have.*got object"):
        lm.importance.PFI(Fixed(), object())


def test_metric_that_needs_probabilities_needs_predict_proba_one():
    class Probabilities(AbsoluteError):
        requires_labels = False

    with pytest.raises(TypeError, match="Probabilities needs probabilities.*Fixed"):
        lm.importance.PFI(Fixed(), Probabilities())


def test_group_of_one_feature_matches_the_feature():
    plain, grouped = static_pfi(), static_pfi(groups={"only_a": ["a"]})
    for x, y in make_stream(2000):
        plain.learn_one(x, y)
        grouped.learn_one(x, y)
    assert grouped.importances == {"only_a": plain.importances["a"], "b": plain.importances["b"]}


def test_group_is_replaced_from_one_past_instance():
    # `a` and `b` are always equal, so the model is exact as long as they come from one instance
    class Difference:
        def predict_one(self, x):
            return x["a"] - x["b"]

    def run(groups):
        explainer = lm.importance.PFI(
            Difference(), AbsoluteError(), tracker=lm.tracker.Mean(), groups=groups, seed=0
        )
        rng = random.Random(0)
        for _ in range(5000):
            value = rng.random()
            explainer.learn_one({"a": value, "b": value}, 0.0)
        return explainer.importances

    alone = run(None)
    assert alone["a"] == pytest.approx(1 / 3, abs=0.02)
    assert alone["b"] == pytest.approx(1 / 3, abs=0.02)
    assert run({"ab": ["a", "b"]}) == {"ab": 0.0}


def test_group_number_of_predictions():
    class Counting(Fixed):
        n = 0

        def predict_one(self, x):
            self.n += 1
            return super().predict_one(x)

    model = Counting()
    explainer = lm.importance.PFI(
        model, AbsoluteError(), n_samples=3, groups={"ab": ["a", "b"]}, seed=0
    )
    x = {"a": 1.0, "b": 1.0, "c": 1.0}
    explainer.learn_one(x, 2.0)
    explainer.learn_one(x, 2.0)
    assert model.n == 1 + 2 * 3


def test_group_with_some_of_its_features():
    class Recording(lm.imputer.Marginal):
        calls = ()

        def impute_one(self, x, features):
            self.calls = (*self.calls, list(features))
            return super().impute_one(x, features)

    imputer = Recording(seed=0)
    explainer = lm.importance.PFI(
        Fixed(), AbsoluteError(), imputer=imputer, groups={"ab": ["a", "b"]}
    )
    explainer.learn_one({"a": 1.0, "c": 1.0}, 2.0)
    explainer.learn_one({"a": 1.0, "c": 1.0}, 2.0)
    assert imputer.calls == (["a", "b"], ["c"])
    assert set(explainer.importances) == {"ab", "c"}


def test_group_is_seen_when_any_of_its_features_is():
    explainer = lm.importance.PFI(
        Fixed(), AbsoluteError(), max_age=2, groups={"ab": ["a", "b"]}, seed=0
    )
    for x in [{"a": 1.0, "b": 1.0}, {"a": 1.0}, {"b": 1.0}, {"b": 1.0}]:
        explainer.learn_one(x, 2.0)
    assert set(explainer.importances) == {"ab"}


@pytest.mark.parametrize(
    "groups, message",
    [
        ({"x": ["a"], "y": ["a", "b"]}, "feature 'a' is in more than one group"),
        ({"x": []}, "group 'x' has no features"),
    ],
)
def test_bad_groups(groups, message):
    with pytest.raises(ValueError, match=message):
        lm.importance.PFI(Fixed(), AbsoluteError(), groups=groups)


def test_group_named_like_a_feature_in_no_group():
    explainer = lm.importance.PFI(Fixed(), AbsoluteError(), groups={"b": ["a"]})
    with pytest.raises(ValueError, match="'b' is both a feature and the name of a group"):
        explainer.learn_one({"a": 1.0, "b": 1.0}, 2.0)


def test_sparse_group_is_replaced_as_a_whole():
    # As with one-hot encoding, an instance only has the active column of the group
    class Recording:
        seen = ()

        def predict_one(self, x):
            self.seen = (*self.seen, dict(x))
            return 0.0

    model = Recording()
    explainer = lm.importance.PFI(model, AbsoluteError(), groups={"color": ["red", "blue"]}, seed=0)
    explainer.learn_one({"blue": 1.0}, 0.0)
    explainer.learn_one({"red": 1.0}, 0.0)
    assert model.seen == ({"red": 1.0}, {"blue": 1.0})


class SquaredError(AbsoluteError):
    """A minimal River-style metric with a squared loss."""

    def update(self, y_true, y_pred):
        self.total += (y_true - y_pred) ** 2
        self.n += 1

    def revert(self, y_true, y_pred):
        self.total -= (y_true - y_pred) ** 2
        self.n -= 1

    def clone(self):
        return SquaredError()


class Additive:
    """A model that uses `a` and `b`, but not `c`."""

    def predict_one(self, x):
        return 3 * x.get("a", 0.0) + x.get("b", 0.0)


def make_additive_stream(n, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        x = {"a": rng.random(), "b": rng.random(), "c": rng.random()}
        yield x, 3 * x["a"] + x["b"]


def static_sage(metric=None, model=None, **kwargs):
    return lm.importance.SAGE(
        model or Additive(),
        metric or SquaredError(),
        imputer=lm.imputer.Marginal(lm.storage.UniformReservoir(size=500, seed=1), seed=2),
        tracker=lm.tracker.Mean(),
        **kwargs,
    )


def test_sage_matches_the_true_values():
    # With a squared loss and independent features, the value is the variance from the feature:
    # 9 / 12 for `a`, 1 / 12 for `b`, and 0 for `c`
    explainer = static_sage(n_samples=50, seed=3)
    for x, y in make_additive_stream(4000):
        explainer.learn_one(x, y)
    assert explainer.importances["a"] == pytest.approx(9 / 12, abs=0.03)
    assert explainer.importances["b"] == pytest.approx(1 / 12, abs=0.03)
    assert explainer.importances["c"] == pytest.approx(0.0, abs=0.03)


@pytest.mark.parametrize(
    "make_tracker", [lm.tracker.Mean, lambda: lm.tracker.EWMean(alpha=0.01)], ids=["mean", "ewmean"]
)
@pytest.mark.parametrize("metric", [AbsoluteError, SquaredError, Correct])
def test_sage_values_add_up_to_the_drop_in_loss(make_tracker, metric):
    explainer = lm.importance.SAGE(
        Additive(), metric(), tracker=make_tracker(), n_samples=2, seed=0
    )
    mean, drop = make_tracker(), make_tracker()
    sign = -1 if metric.bigger_is_better else 1

    def loss(y, y_pred):
        one = metric()
        one.update(y, y_pred)
        return sign * one.get()

    for i, (x, y) in enumerate(make_additive_stream(1000)):
        explainer.learn_one(x, y)
        if i > 0:  # no past instance for the first one
            y_pred = Additive().predict_one(x)
            mean.update(y_pred)
            drop.update(loss(y, mean.get()) - loss(y, y_pred))

    assert sum(explainer.importances.values()) == pytest.approx(drop.get(), abs=1e-9)


def test_sage_few_samples_keep_the_sum_but_move_values_below_zero():
    def importances(n_samples):
        explainer = static_sage(n_samples=n_samples, seed=3)
        for x, y in make_additive_stream(4000):
            explainer.learn_one(x, y)
        return explainer.importances

    few, many = importances(1), importances(50)
    assert few["c"] < many["c"] - 0.1
    assert few["c"] < -0.1
    assert sum(few.values()) == pytest.approx(sum(many.values()), abs=0.01)


def test_sage_number_of_predictions():
    class Counting(Additive):
        n = 0

        def predict_one(self, x):
            self.n += 1
            return super().predict_one(x)

    model = Counting()
    explainer = lm.importance.SAGE(model, AbsoluteError(), n_samples=4, seed=0)
    x = {"a": 1.0, "b": 1.0, "c": 1.0}
    explainer.learn_one(x, 4.0)
    assert model.n == 0  # no past instance yet
    explainer.learn_one(x, 4.0)
    assert model.n == 1 + (3 - 1) * 4


def test_sage_default_number_of_samples():
    assert lm.importance.SAGE(Additive(), AbsoluteError()).n_samples == 5


class Recording(lm.imputer.Marginal):
    """An imputer that keeps the features it was asked to replace."""

    calls = ()

    def impute_one(self, x, features):
        self.calls = (*self.calls, tuple(features))
        return super().impute_one(x, features)


def test_sage_reveals_one_more_feature_at_each_step():
    imputer = Recording(seed=0)
    explainer = lm.importance.SAGE(
        Additive(), AbsoluteError(), imputer=imputer, n_samples=2, seed=0
    )
    x = {"a": 1.0, "b": 1.0, "c": 1.0}
    explainer.learn_one(x, 4.0)
    explainer.learn_one(x, 4.0)

    first, second, third, fourth = imputer.calls
    assert first == second
    assert third == fourth
    assert len(first) == 2
    assert len(third) == 1
    assert set(third) < set(first)
    assert set(explainer.importances) == {"a", "b", "c"}


def test_sage_order_changes_between_instances():
    imputer = Recording(seed=0)
    explainer = lm.importance.SAGE(
        Additive(), AbsoluteError(), imputer=imputer, n_samples=1, seed=0
    )
    for x, y in make_additive_stream(50):
        explainer.learn_one(x, y)
    assert len(set(imputer.calls)) > 3


def test_sage_averages_predictions_before_the_loss():
    # The samples give -1 and 1, whose mean is exact, while the mean of their losses is 1
    class Script(lm.base.Imputer):
        values = (-1.0, 1.0)
        i = 0

        def impute_one(self, x, features):
            self.i += 1
            return {**x, "v": self.values[self.i % 2]}

    class Value:
        def predict_one(self, x):
            return x["v"]

    explainer = lm.importance.SAGE(
        Value(), AbsoluteError(), imputer=Script(), tracker=lm.tracker.Mean(), n_samples=2, seed=0
    )
    explainer.learn_one({"v": 0.0, "other": 0.0}, 0.0)
    assert explainer.importances == {"v": 0.0, "other": 0.0}


def test_average_of_probabilities_counts_a_missing_class_as_zero():
    average = lm.importance._average([{True: 0.8}, {True: 0.2, False: 0.8}])
    assert average == pytest.approx({True: 0.5, False: 0.4})


def test_sage_mean_prediction_of_a_new_class_counts_earlier_instances_as_zero():
    explainer = lm.importance.SAGE(Additive(), AbsoluteError(), tracker=lm.tracker.Mean())
    assert explainer._mean_prediction({"a": 1.0}) == {"a": 1.0}
    assert explainer._mean_prediction({"b": 1.0}) == pytest.approx({"a": 0.5, "b": 0.5})
    assert explainer._mean_prediction({"a": 1.0}) == pytest.approx({"a": 2 / 3, "b": 1 / 3})


def test_sage_no_importances_before_a_past_instance():
    explainer = lm.importance.SAGE(Additive(), AbsoluteError(), seed=0)
    assert explainer.importances == {}
    explainer.learn_one({"a": 1.0}, 3.0)
    assert explainer.importances == {}
    explainer.learn_one({"a": 1.0}, 3.0)
    assert set(explainer.importances) == {"a"}


def test_sage_features_can_appear_and_disappear():
    explainer = lm.importance.SAGE(Additive(), AbsoluteError(), seed=0)
    explainer.learn_one({"a": 1.0}, 3.0)
    explainer.learn_one({"a": 0.0, "new": 1.0}, 0.0)
    assert set(explainer.importances) == {"a", "new"}
    explainer.learn_one({"new": 1.0}, 0.0)
    assert set(explainer.importances) == {"a", "new"}


@pytest.mark.parametrize("value", [None, {}])
def test_sage_model_that_cannot_predict_yet_is_skipped(value):
    class Untrained:
        def predict_one(self, x):
            return value

    explainer = lm.importance.SAGE(Untrained(), AbsoluteError(), seed=0)
    for _ in range(3):
        explainer.learn_one({"a": 1.0}, 2.0)
    assert explainer.importances == {}
    assert len(explainer.imputer.storage) == 3
    assert explainer._means == {}


@pytest.mark.parametrize("label", [True, "yes"])
def test_sage_rejects_predictions_that_cannot_be_averaged(label):
    class Labels:
        def predict_one(self, x):
            return label

    explainer = lm.importance.SAGE(Labels(), AbsoluteError(), seed=0)
    explainer.learn_one({"a": 1.0}, True)
    with pytest.raises(TypeError, match="must be numbers or probabilities"):
        explainer.learn_one({"a": 1.0}, True)


def test_sage_seed_makes_it_reproducible():
    def run(seed):
        explainer = lm.importance.SAGE(Additive(), AbsoluteError(), seed=seed)
        for x, y in make_additive_stream(300):
            explainer.learn_one(x, y)
        return explainer.importances

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_sage_seed_applies_to_the_order_with_own_imputer():
    def run(seed):
        explainer = static_sage(model=Additive(), seed=seed, n_samples=1)
        for x, y in make_additive_stream(300):
            explainer.learn_one(x, y)
        return explainer.importances

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_sage_importances_follow_the_order_of_the_features():
    explainer = lm.importance.SAGE(Additive(), AbsoluteError(), seed=3)
    for x, y in make_additive_stream(5):
        explainer.learn_one(x, y)
    assert list(explainer.importances) == ["a", "b", "c"]


def test_sage_group_of_one_feature_matches_the_feature():
    plain = static_sage(n_samples=2, seed=3)
    grouped = static_sage(n_samples=2, seed=3, groups={"only_a": ["a"]})
    for x, y in make_additive_stream(500):
        plain.learn_one(x, y)
        grouped.learn_one(x, y)
    values = plain.importances
    assert grouped.importances == {"only_a": values["a"], "b": values["b"], "c": values["c"]}


def test_sage_group_matches_the_true_value():
    # With a squared loss, the group gets the variance of `3a + b`, which is 10 / 12
    explainer = static_sage(n_samples=50, seed=3, groups={"ab": ["a", "b"]})
    for x, y in make_additive_stream(4000):
        explainer.learn_one(x, y)
    assert explainer.importances["ab"] == pytest.approx(10 / 12, abs=0.03)
    assert explainer.importances["c"] == pytest.approx(0.0, abs=0.03)


def test_sage_group_number_of_predictions():
    class Counting(Additive):
        n = 0

        def predict_one(self, x):
            self.n += 1
            return super().predict_one(x)

    model = Counting()
    explainer = lm.importance.SAGE(
        model, AbsoluteError(), n_samples=4, groups={"ab": ["a", "b"]}, seed=0
    )
    x = {"a": 1.0, "b": 1.0, "c": 1.0}
    explainer.learn_one(x, 4.0)
    explainer.learn_one(x, 4.0)
    assert model.n == 1 + (2 - 1) * 4


def test_sage_sparse_group_is_hidden_as_a_whole():
    # As with one-hot encoding, an instance only has the active column of the group
    class Recording:
        seen = ()

        def predict_one(self, x):
            self.seen = (*self.seen, dict(x))
            return 0.0

    model = Recording()
    explainer = lm.importance.SAGE(
        model, AbsoluteError(), n_samples=1, groups={"color": ["red", "blue"]}, seed=0
    )
    explainer.learn_one({"blue": 1.0, "c": 0.0}, 0.0)
    for _ in range(10):
        explainer.learn_one({"red": 1.0, "c": 1.0}, 0.0)
    hidden = [x for x in model.seen if x["c"] == 1.0 and "blue" in x]
    assert hidden
    assert all("red" not in x for x in hidden)
    assert set(explainer.importances) == {"color", "c"}

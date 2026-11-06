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


class TwoOutputs:
    """A model whose output `y1` only uses `a`, and whose output `y2` only uses `b`."""

    def predict_one(self, x):
        return {"y1": 2 * x.get("a", 0.0), "y2": 2 * x.get("b", 0.0)}


def make_stream(n=20000, seed=0):
    rng = random.Random(seed)
    for _ in range(n):
        x = {"a": rng.random(), "b": rng.random()}
        yield x, {"y1": 2 * x["a"], "y2": 2 * x["b"]}


def pfi(model, **kwargs):
    return lm.importance.PFI(model, AbsoluteError(), **kwargs)


def per_output(model=None):
    def make_explainer(output):
        return lm.importance.PFI(
            output,
            AbsoluteError(),
            imputer=lm.imputer.Marginal(lm.storage.UniformReservoir(size=500, seed=1), seed=2),
            tracker=lm.tracker.Mean(),
        )

    return lm.multioutput.PerOutput(model or TwoOutputs(), make_explainer)


def test_each_output_matches_the_true_value():
    # The loss is 0 with the feature, and 2 |a - a'| without it, whose mean is 2 / 3
    explainer = per_output()
    for x, y in make_stream():
        explainer.learn_one(x, y)
    importances = explainer.importances

    assert set(importances) == {"y1", "y2"}
    assert importances["y1"]["a"] == pytest.approx(2 / 3, abs=0.02)
    assert importances["y1"]["b"] == 0.0
    assert importances["y2"]["b"] == pytest.approx(2 / 3, abs=0.02)
    assert importances["y2"]["a"] == 0.0


def test_no_importances_before_an_output_appears():
    explainer = per_output()
    assert explainer.importances == {}


def test_only_outputs_in_the_target_are_updated():
    explainer = per_output()
    for _ in range(2):
        explainer.learn_one({"a": 1.0, "b": 1.0}, {"y1": 2.0})
    assert set(explainer.importances) == {"y1"}

    explainer.learn_one({"a": 1.0, "b": 1.0}, {"y2": 2.0})
    assert set(explainer.importances) == {"y1", "y2"}
    assert explainer.importances["y2"] == {}


def test_each_output_has_its_own_explainer():
    made = []

    def make_explainer(output):
        made.append(output.output)
        return pfi(output, seed=0)

    explainer = lm.multioutput.PerOutput(TwoOutputs(), make_explainer)
    for x, y in make_stream(3):
        explainer.learn_one(x, y)
    assert made == ["y1", "y2"]
    assert explainer.explainers["y1"] is not explainer.explainers["y2"]
    assert explainer.explainers["y1"].imputer is not explainer.explainers["y2"].imputer


def test_number_of_predictions():
    class Counting(TwoOutputs):
        n = 0

        def predict_one(self, x):
            self.n += 1
            return super().predict_one(x)

    model = Counting()
    explainer = lm.multioutput.PerOutput(model, lambda output: pfi(output, n_samples=3, seed=0))
    x, y = {"a": 1.0, "b": 1.0}, {"y1": 2.0, "y2": 2.0}
    explainer.learn_one(x, y)
    assert model.n == 0  # no past instance yet
    explainer.learn_one(x, y)
    assert model.n == 2 * (1 + 2 * 3)


def test_model_that_cannot_predict_yet_is_skipped():
    class Untrained:
        def predict_one(self, x):
            return {}

    explainer = lm.multioutput.PerOutput(Untrained(), lambda output: pfi(output, seed=0))
    for _ in range(3):
        explainer.learn_one({"a": 1.0}, {"y1": 2.0})
    assert explainer.importances == {"y1": {}}


def test_metric_that_needs_probabilities_gets_those_of_the_output():
    class Probabilities(TwoOutputs):
        def predict_proba_one(self, x):
            return {"y1": x["a"], "y2": x["b"]}

    class NeedsProbabilities(AbsoluteError):
        requires_labels = False

        def clone(self):
            return NeedsProbabilities()

    explainer = lm.multioutput.PerOutput(
        Probabilities(), lambda output: lm.importance.PFI(output, NeedsProbabilities(), seed=0)
    )
    for _ in range(2):
        explainer.learn_one({"a": 0.5, "b": 0.5}, {"y1": 0.5, "y2": 0.5})
    assert explainer.importances == {"y1": {"a": 0.0, "b": 0.0}, "y2": {"a": 0.0, "b": 0.0}}


def test_metric_that_needs_probabilities_needs_predict_proba_one():
    class NeedsProbabilities(AbsoluteError):
        requires_labels = False

    explainer = lm.multioutput.PerOutput(
        TwoOutputs(), lambda output: lm.importance.PFI(output, NeedsProbabilities())
    )
    with pytest.raises(TypeError, match="needs probabilities"):
        explainer.learn_one({"a": 1.0}, {"y1": 2.0})


def test_bad_model():
    with pytest.raises(TypeError, match="model must have a predict_one method, got object"):
        lm.multioutput.PerOutput(object(), pfi)


def test_make_explainer_must_give_an_importance_explainer():
    explainer = lm.multioutput.PerOutput(TwoOutputs(), lambda output: object())
    with pytest.raises(TypeError, match="make_explainer must give.*got object"):
        explainer.learn_one({"a": 1.0}, {"y1": 2.0})

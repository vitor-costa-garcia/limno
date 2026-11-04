import random

import pytest

import limno as lm

river = pytest.importorskip("river")
from river import datasets, linear_model, metrics, preprocessing, stats, tree


class DropFeature(lm.base.Explainer):
    def explain_one(self, x):
        y_pred = self.model.predict_one(x)
        return {
            f: y_pred - self.model.predict_one({k: v for k, v in x.items() if k != f}) for f in x
        }


def regressor_pipeline():
    return preprocessing.StandardScaler() | linear_model.LinearRegression()


def classifier_pipeline():
    return preprocessing.StandardScaler() | linear_model.LogisticRegression()


def run(explainer, model, stream):
    for x, y in stream:
        explainer.learn_one(x, y)
        model.learn_one(x, y)
    return explainer.importances


@pytest.mark.parametrize(
    "model",
    [
        linear_model.LinearRegression(),
        linear_model.LogisticRegression(),
        tree.HoeffdingTreeRegressor(),
        tree.HoeffdingTreeClassifier(),
        regressor_pipeline(),
    ],
    ids=lambda model: type(model).__name__,
)
def test_river_models_satisfy_protocol(model):
    assert isinstance(model, lm.base.Model)


@pytest.mark.parametrize(
    "metric",
    [metrics.MAE(), metrics.MSE(), metrics.Accuracy(), metrics.LogLoss(), metrics.CrossEntropy()],
    ids=lambda metric: type(metric).__name__,
)
def test_river_metrics_satisfy_protocol(metric):
    assert isinstance(metric, lm.base.Metric)


def test_local_explainer_on_a_river_pipeline():
    model = regressor_pipeline()
    explainer = DropFeature(model)

    for x, y in datasets.synth.Friedman(seed=42).take(200):
        attributions = explainer.explain_one(x)
        assert set(attributions) == set(x)
        model.learn_one(x, y)
        explainer.learn_one(x, y)

    assert any(attributions.values())


def test_pfi_regression_finds_the_features_in_the_target():
    # In Friedman's stream, features 5 to 9 are not used to build the target
    model = regressor_pipeline()
    explainer = lm.importance.PFI(model, metrics.MAE(), seed=42)
    importances = run(explainer, model, datasets.synth.Friedman(seed=42).take(3000))

    assert set(importances) == set(range(10))
    assert min(importances[f] for f in (0, 1, 3, 4)) > max(importances[f] for f in range(5, 10))


@pytest.mark.parametrize(
    "metric", [metrics.LogLoss(), metrics.Accuracy()], ids=lambda metric: type(metric).__name__
)
def test_pfi_classification_finds_what_the_model_uses(metric):
    # With classification function 0, the Agrawal class only depends on age
    stream = datasets.synth.Agrawal(classification_function=0, seed=42).take(3000)
    model = tree.HoeffdingTreeClassifier(grace_period=100)
    importances = run(lm.importance.PFI(model, metric, seed=42), model, stream)

    assert max(importances, key=importances.get) == "age"
    assert importances["age"] > 0


def test_pfi_picks_the_prediction_method_from_the_metric():
    model = classifier_pipeline()
    assert lm.importance.PFI(model, metrics.LogLoss())._predict == model.predict_proba_one
    assert lm.importance.PFI(model, metrics.Accuracy())._predict == model.predict_one
    regressor = regressor_pipeline()
    assert lm.importance.PFI(regressor, metrics.MAE())._predict == regressor.predict_one


@pytest.mark.parametrize(
    "model, metric, names",
    [
        (regressor_pipeline(), metrics.LogLoss(), "LogLoss does not work with Pipeline"),
        (classifier_pipeline(), metrics.MAE(), "MAE does not work with Pipeline"),
        (linear_model.LinearRegression(), metrics.Accuracy(), "Accuracy.*LinearRegression"),
    ],
)
def test_pfi_rejects_metric_that_does_not_work_with_the_model(model, metric, names):
    with pytest.raises(ValueError, match=names):
        lm.importance.PFI(model, metric)


def test_river_metric_with_a_model_from_elsewhere():
    class Fixed:
        def predict_one(self, x):
            return 2 * x["a"]

    explainer = lm.importance.PFI(Fixed(), metrics.MAE(), seed=0)
    explainer.learn_one({"a": 0.0}, 0.0)
    explainer.learn_one({"a": 1.0}, 2.0)
    assert explainer.importances["a"] == pytest.approx(2.0)


def test_untrained_classifier_is_skipped():
    # A new Hoeffding tree gives {} as its probabilities
    model = tree.HoeffdingTreeClassifier()
    assert model.predict_proba_one({"a": 1.0}) == {}
    explainer = lm.importance.PFI(model, metrics.LogLoss(), seed=0)
    explainer.learn_one({"a": 1.0}, True)
    explainer.learn_one({"a": 2.0}, False)
    assert explainer.importances == {}


def test_metric_stays_empty_along_the_stream():
    model = classifier_pipeline()
    explainer = lm.importance.PFI(model, metrics.LogLoss(), seed=0)
    run(explainer, model, datasets.Phishing().take(100))
    assert explainer.metric.get() == 0.0


def test_river_stats_as_trackers():
    stream = list(datasets.synth.Friedman(seed=42).take(300))

    def importances(tracker):
        model = regressor_pipeline()
        explainer = lm.importance.PFI(
            model,
            metrics.MAE(),
            imputer=lm.imputer.Marginal(lm.storage.UniformReservoir(seed=1), seed=2),
            tracker=tracker,
        )
        return run(explainer, model, stream)

    assert isinstance(stats.Mean(), lm.base.Tracker)
    assert importances(stats.Mean()) == pytest.approx(importances(lm.tracker.Mean()))


def test_pfi_follows_a_drift():
    # The target moves from depending on `a` to depending on `b`
    def stream():
        rng = random.Random(0)
        for i in range(12000):
            x = {"a": rng.random(), "b": rng.random()}
            yield x, 3 * x["a"] if i < 6000 else 3 * x["b"]

    model = linear_model.LinearRegression()
    explainer = lm.importance.PFI(
        model,
        metrics.MAE(),
        imputer=lm.imputer.Marginal(lm.storage.GeometricReservoir(seed=0), seed=0),
        tracker=lm.tracker.EWMean(alpha=0.01),
    )
    before = None
    for i, (x, y) in enumerate(stream()):
        explainer.learn_one(x, y)
        model.learn_one(x, y)
        if i == 5999:
            before = explainer.importances
    after = explainer.importances

    assert before["a"] > before["b"] + 0.2
    assert after["b"] > after["a"] + 0.2

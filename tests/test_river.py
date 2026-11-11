import random

import pytest

import limno as lm

river = pytest.importorskip("river")
from river import datasets, linear_model, metrics, multioutput, preprocessing, stats, tree


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


def regressor_chain():
    return multioutput.RegressorChain(linear_model.LinearRegression(), order=["y1", "y2"])


def classifier_chain():
    return multioutput.ClassifierChain(linear_model.LogisticRegression(), order=["y1", "y2"])


def multi_output_stream(target, n=4000):
    # `y1` only depends on `a`, `y2` only depends on `b`, and `c` is noise
    rng = random.Random(0)
    for _ in range(n):
        x = {"a": rng.random(), "b": rng.random(), "c": rng.random()}
        yield x, {"y1": target(x["a"]), "y2": target(x["b"])}


def run_per_output(model, metric, stream):
    explainer = lm.multioutput.PerOutput(
        model, lambda output: lm.importance.PFI(output, metric, seed=42)
    )
    for x, y in stream:
        explainer.learn_one(x, y)
        model.learn_one(x, y)
    return explainer.importances


@pytest.mark.parametrize(
    "model",
    [regressor_chain(), preprocessing.StandardScaler() | regressor_chain()],
    ids=["chain", "pipeline"],
)
def test_per_output_regression_finds_the_feature_of_each_output(model):
    importances = run_per_output(model, metrics.MAE(), multi_output_stream(lambda v: 3 * v))

    assert max(importances["y1"], key=importances["y1"].get) == "a"
    assert max(importances["y2"], key=importances["y2"].get) == "b"
    assert importances["y1"]["a"] > 0.5
    assert importances["y2"]["b"] > 0.5
    assert abs(importances["y1"]["c"]) < 0.05
    assert abs(importances["y2"]["c"]) < 0.05


@pytest.mark.parametrize(
    "metric", [metrics.LogLoss(), metrics.Accuracy()], ids=lambda metric: type(metric).__name__
)
def test_per_output_classification_finds_the_feature_of_each_output(metric):
    stream = multi_output_stream(lambda v: v > 0.5)
    importances = run_per_output(classifier_chain(), metric, stream)

    assert max(importances["y1"], key=importances["y1"].get) == "a"
    assert max(importances["y2"], key=importances["y2"].get) == "b"


def test_per_output_includes_the_effect_through_earlier_outputs():
    # With no direct weight, `a` only reaches `y2` through the prediction of `y1`
    model = regressor_chain()
    x = {"a": 1.0}
    for _ in range(100):
        model.learn_one(x, {"y1": 3.0, "y2": 3.0})
    model["y2"].weights["a"] = 0.0
    model["y2"].weights["y1"] = 1.0
    model["y2"].intercept = 0.0

    explainer = lm.multioutput.PerOutput(
        model, lambda output: lm.importance.PFI(output, metrics.MAE(), seed=42)
    )
    y_y1 = model.predict_one(x)["y1"]
    explainer.learn_one({"a": 0.0}, {"y2": y_y1})
    explainer.learn_one(x, {"y2": y_y1})
    assert explainer.importances["y2"]["a"] > 0.5


@pytest.mark.parametrize(
    "model, metric, names",
    [
        (
            regressor_chain(),
            metrics.LogLoss(),
            "LogLoss does not work with one output of Regressor",
        ),
        (classifier_chain(), metrics.MAE(), "MAE does not work with one output of ClassifierChain"),
        (
            regressor_chain(),
            metrics.multioutput.MicroAverage(metrics.MAE()),
            "MicroAverage does not work with one output of RegressorChain",
        ),
        (
            preprocessing.StandardScaler() | classifier_chain(),
            metrics.MAE(),
            "MAE does not work with one output of Pipeline",
        ),
    ],
    ids=["regressor-logloss", "classifier-mae", "micro-average", "pipeline"],
)
def test_per_output_rejects_metric_that_does_not_work_with_one_output(model, metric, names):
    explainer = lm.multioutput.PerOutput(model, lambda output: lm.importance.PFI(output, metric))
    with pytest.raises(ValueError, match=names):
        explainer.learn_one({"a": 1.0}, {"y1": 1.0})


@pytest.mark.parametrize(
    "model",
    [
        linear_model.LinearRegression(),
        preprocessing.StandardScaler() | linear_model.LinearRegression(),
    ],
    ids=["model", "pipeline"],
)
def test_per_output_rejects_a_model_with_one_output(model):
    with pytest.raises(TypeError, match="model must have many outputs"):
        lm.multioutput.PerOutput(model, lambda output: lm.importance.PFI(output, metrics.MAE()))


def test_pfi_with_a_multi_output_metric_gives_one_value_per_feature():
    model = regressor_chain()
    explainer = lm.importance.PFI(model, metrics.multioutput.MicroAverage(metrics.MAE()), seed=42)
    importances = run(explainer, model, multi_output_stream(lambda v: 3 * v))

    assert set(importances) == {"a", "b", "c"}
    assert min(importances["a"], importances["b"]) > 0.25
    assert abs(importances["c"]) < 0.05


def test_pfi_groups_on_a_river_pipeline():
    # In Friedman's stream, features 0 to 4 build the target, and 5 to 9 are noise
    model = regressor_pipeline()
    groups = {"used": [0, 1, 2, 3, 4], "noise": [5, 6, 7, 8, 9]}
    explainer = lm.importance.PFI(model, metrics.MAE(), groups=groups, seed=42)
    importances = run(explainer, model, datasets.synth.Friedman(seed=42).take(3000))

    assert set(importances) == {"used", "noise"}
    assert importances["used"] > 1
    assert abs(importances["noise"]) < 0.1


def test_per_output_with_groups():
    model = regressor_chain()
    explainer = lm.multioutput.PerOutput(
        model,
        lambda output: lm.importance.PFI(output, metrics.MAE(), groups={"ab": ["a", "b"]}, seed=42),
    )
    for x, y in multi_output_stream(lambda v: 3 * v):
        explainer.learn_one(x, y)
        model.learn_one(x, y)
    importances = explainer.importances

    assert set(importances["y1"]) == {"ab", "c"}
    assert importances["y1"]["ab"] > 0.5
    assert abs(importances["y2"]["c"]) < 0.05


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

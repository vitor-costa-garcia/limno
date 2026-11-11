"""Global explainers, which measure how much a model relies on each feature over a stream."""

from __future__ import annotations

import collections.abc
import copy
import numbers
import random
import typing

from limno.base import (
    Feature,
    ImportanceExplainer,
    Imputer,
    Metric,
    Model,
    Tracker,
    _check_groups,
    _group,
    _predictor,
)
from limno.imputer import Marginal
from limno.storage import GeometricReservoir
from limno.tracker import EWMean

__all__ = [
    "PFI",
    "SAGE",
]


Groups = collections.abc.Mapping[typing.Hashable, collections.abc.Collection[Feature]]


class _LossImportance(ImportanceExplainer):
    """What the explainers that compare losses have in common."""

    def __init__(
        self,
        model: Model,
        metric: Metric,
        imputer: Imputer | None,
        tracker: Tracker | None,
        n_samples: int,
        seed: int | None,
        max_age: int | None,
        groups: Groups | None,
    ) -> None:
        super().__init__(model)
        self._predict = _predictor(model, metric)
        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {n_samples}")
        if max_age is not None and max_age < 1:
            raise ValueError(f"max_age must be at least 1 or None, got {max_age}")
        groups = {} if groups is None else groups
        _check_groups(groups)

        # A clone, as the loss of an instance must not be mixed with what the metric has seen
        self.metric = metric.clone()
        self.n_samples = n_samples
        self.seed = seed
        self.max_age = max_age
        self.groups = groups
        self._rng = random.Random(seed)
        if imputer is None:
            storage = GeometricReservoir(seed=self._rng.getrandbits(32))
            imputer = Marginal(storage, seed=self._rng.getrandbits(32))
        self.imputer = imputer
        self.tracker = EWMean() if tracker is None else tracker
        self._trackers: dict[Feature, Tracker] = {}
        self._sign = -1.0 if metric.bigger_is_better else 1.0
        self._n_seen = 0
        self._last_seen: dict[Feature, int] = {}

    def _see(self, x: dict) -> dict[typing.Hashable, list[Feature]]:
        """Note which features and groups are in `x`, and give its features by group."""
        grouped = _group(x, self.groups)
        self._n_seen += 1
        for name in grouped:
            self._last_seen[name] = self._n_seen
        return grouped

    def _features(self, name: typing.Hashable, grouped: dict) -> collections.abc.Collection:
        """All the features of a group, so that those missing from `x` are filled in."""
        return self.groups.get(name, grouped[name])

    def _loss(self, y: typing.Any, y_pred: typing.Any) -> float:
        self.metric.update(y, y_pred)
        loss = self.metric.get()
        self.metric.revert(y, y_pred)
        return self._sign * loss

    def _update_importance(self, feature: Feature, value: float) -> None:
        if feature not in self._trackers:
            self._trackers[feature] = copy.deepcopy(self.tracker)
        self._trackers[feature].update(value)

    @property
    def importances(self) -> dict[Feature, float]:
        """The current importance of each feature or group, without those hidden by `max_age`."""
        return {
            feature: tracker.get()
            for feature, tracker in self._trackers.items()
            if self.max_age is None or self._n_seen - self._last_seen[feature] < self.max_age
        }


class PFI(_LossImportance):
    """Incremental permutation feature importance (iPFI).

    For each instance and feature, the feature is replaced with its value in a past instance. The
    importance of a feature is how much this increases the loss of the model, as a running mean
    over the stream. A positive value means that the model does better with the feature, and a
    value close to zero that it does not rely on it.

    Each instance takes `1 + n_features * n_samples` predictions, or `1 + n_groups * n_samples` with
    `groups`.

    The method is from Fumagalli et al., "Incremental permutation feature importance (iPFI):
    towards online explanations on data streams", Machine Learning, 2023,
    https://doi.org/10.1007/s10994-023-06385-y.

    Parameters
    ----------
    model
        The model to explain. It is not updated, but explanations follow it as it learns.
    metric
        The metric that gives the loss, such as `river.metrics.MAE` or `river.metrics.LogLoss`.
        Whether the model is asked for labels or probabilities depends on the metric, as in River.
        It is computed on one instance at a time, so metrics that need many instances, such as
        `river.metrics.ROCAUC` or `river.metrics.F1`, are not useful here.
    imputer
        How features are replaced. By default, an `imputer.Marginal` with a
        `storage.GeometricReservoir` of 100 instances, which follows changes in the stream.
    tracker
        The running statistic of each importance, which is copied for each feature. By default, a
        `tracker.EWMean` with an `alpha` of 0.001. The authors of iPFI found that an `alpha`
        between 0.001 and 0.01 works well. Use `tracker.Mean` with a
        `storage.UniformReservoir` if the stream does not change.
    n_samples
        The number of past instances to use per feature and instance. More samples give less noise
        but take more predictions.
    seed
        Random seed of the default imputer. To use a seed with your own imputer, give it to the
        imputer and its storage.
    max_age
        Hide a feature from `importances` once it is missing from this many instances in a row, as
        its value is out of date. It comes back with its old value if it appears again. Raise this
        for features that are rare, and use `None` to never hide a feature. The default matches the
        window of the default tracker. A group is seen when any of its features is. This is not
        part of iPFI, which assumes that the features do not change.
    groups
        Features to treat as one, such as `{"location": ["lat", "lon"]}`. A group is replaced as a
        whole and gets one importance, under its name. With `imputer.Marginal`, its values all come
        from the same past instance. Features in no group get their own importance. This suits
        features that only make sense together, such as the columns of a one-hot encoding, or that
        are correlated. A model can split its weight between correlated features, and then each one
        alone looks less important than they are together.


    Examples
    --------
    >>> from river import datasets, linear_model, metrics, preprocessing

    >>> model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    >>> explainer = PFI(model, metrics.MAE(), seed=42)

    >>> for x, y in datasets.synth.Friedman(seed=42).take(1000):
    ...     explainer.learn_one(x, y)
    ...     model.learn_one(x, y)

    In this stream, feature 3 has the largest linear effect on the target.

    >>> max(explainer.importances, key=explainer.importances.get)
    3

    """

    def __init__(
        self,
        model: Model,
        metric: Metric,
        imputer: Imputer | None = None,
        tracker: Tracker | None = None,
        n_samples: int = 1,
        seed: int | None = None,
        max_age: int | None = 2000,
        groups: Groups | None = None,
    ) -> None:
        super().__init__(model, metric, imputer, tracker, n_samples, seed, max_age, groups)
        if imputer is not None and seed is not None:
            raise ValueError("seed only applies to the default imputer, give it to your imputer")

    def learn_one(self, x: dict, y: typing.Any) -> None:
        grouped = self._see(x)
        if self.imputer.ready:
            y_pred = self._predict(x)
            # A River classifier that has not learned yet predicts None or {}
            if y_pred is not None and y_pred != {}:
                loss = self._loss(y, y_pred)
                for name in grouped:
                    imputed_loss = 0.0
                    for _ in range(self.n_samples):
                        imputed = self.imputer.impute_one(x, self._features(name, grouped))
                        imputed_loss += self._loss(y, self._predict(imputed))
                    self._update_importance(name, imputed_loss / self.n_samples - loss)
        self.imputer.learn_one(x)


def _number(y_pred: typing.Any) -> float:
    """Check that a prediction can be averaged."""
    if isinstance(y_pred, bool) or not isinstance(y_pred, numbers.Real):
        raise TypeError(
            "SAGE averages predictions, so they must be numbers or probabilities, got "
            f"{type(y_pred).__name__}"
        )
    return float(y_pred)


def _average(y_preds: list) -> typing.Any:
    """The mean of predictions, in which a key that is missing from a dictionary counts as 0."""
    if isinstance(y_preds[0], dict):
        keys = dict.fromkeys(key for y_pred in y_preds for key in y_pred)
        return {
            key: sum(_number(y_pred.get(key, 0.0)) for y_pred in y_preds) / len(y_preds)
            for key in keys
        }
    return sum(_number(y_pred) for y_pred in y_preds) / len(y_preds)


def _check_no_labels(metric: Metric) -> None:
    """Check that `metric` is not a River metric for labels, as labels cannot be averaged."""
    try:
        from river import linear_model
    except ImportError:
        return
    works_with = getattr(metric, "works_with", None)
    if (
        getattr(metric, "requires_labels", False)
        and works_with
        and works_with(linear_model.LogisticRegression())
    ):
        raise ValueError(
            f"{type(metric).__name__} needs labels, but SAGE averages predictions. Use a metric "
            "for probabilities, such as LogLoss"
        )


class SAGE(_LossImportance):
    """Incremental Shapley additive global importance (iSAGE).

    SAGE shares out the drop in loss between predicting the mean and using the model, based on
    Shapley values. For each instance, it reveals the features in a random order. Features that are
    not revealed yet take their values from past instances, and the importance of a feature is how
    much revealing it lowers the loss, as a running mean over the stream.

    A positive value means that the model does better with the feature. Unlike `PFI`, the values
    add up to the drop in loss, as long as the features of the stream do not change. This also
    accounts for features that interact, as each feature is seen with different others.

    Each instance takes `1 + (n_features - 1) * n_samples` predictions, or
    `1 + (n_groups - 1) * n_samples` with `groups`.

    Features are replaced from the marginal distribution, the interventional approach of the paper.
    So correlated features can make unrealistic instances. The observational approach of the paper
    is not implemented.

    The method is from Muschalik et al., "iSAGE: An Incremental Version of SAGE for Online
    Explanation on Data Streams", ECML PKDD, 2023, https://doi.org/10.1007/978-3-031-43418-1_26.

    Parameters
    ----------
    model
        The model to explain. It is not updated, but explanations follow it as it learns.
    metric
        The metric that gives the loss, such as `river.metrics.MAE` or `river.metrics.LogLoss`.
        Predictions are averaged, so it must take numbers or probabilities. Metrics that need
        labels, such as `river.metrics.Accuracy`, are rejected. A metric of your own for labels
        that are numbers cannot be detected, so do not use one.
    imputer
        How features are replaced. By default, an `imputer.Marginal` with a
        `storage.GeometricReservoir` of 100 instances, which follows changes in the stream.
    tracker
        The running statistic of each importance and of the mean prediction, which is copied for
        each of them. By default, a `tracker.EWMean` with an `alpha` of 0.001. Use `tracker.Mean`
        with a `storage.UniformReservoir` if the stream does not change.
    n_samples
        The number of past instances to average over at each step. The paper only proves that the
        values reach the true ones as this grows. With few samples, importance moves from unused
        features to used ones, so that unused features are below zero, even though the sum stays
        right. Lower this only if predictions are slow, and raise it if the exact values matter.
    seed
        Random seed of the order of the features and of the default imputer. To also use a seed
        with your own imputer, give it to the imputer and its storage.
    max_age
        Hide a feature from `importances` once it is missing from this many instances in a row, as
        its value is out of date. It comes back with its old value if it appears again. Raise this
        for features that are rare, and use `None` to never hide a feature. The default matches the
        window of the default tracker. A group is seen when any of its features is. This is not
        part of iSAGE, which assumes that the features do not change.
    groups
        Features to treat as one, such as `{"location": ["lat", "lon"]}`. A group is revealed as a
        whole and gets one importance, under its name. With `imputer.Marginal`, its values all come
        from the same past instance. Features in no group get their own importance. This suits
        features that only make sense together, such as the columns of a one-hot encoding, or that
        are correlated. A model can split its weight between correlated features, and then each one
        alone looks less important than they are together.


    Examples
    --------
    >>> from river import datasets, linear_model, metrics, preprocessing

    >>> model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    >>> explainer = SAGE(model, metrics.MAE(), seed=42)

    >>> for x, y in datasets.synth.Friedman(seed=42).take(1000):
    ...     explainer.learn_one(x, y)
    ...     model.learn_one(x, y)

    In this stream, feature 3 has the largest linear effect on the target.

    >>> max(explainer.importances, key=explainer.importances.get)
    3

    """

    def __init__(
        self,
        model: Model,
        metric: Metric,
        imputer: Imputer | None = None,
        tracker: Tracker | None = None,
        n_samples: int = 5,
        seed: int | None = None,
        max_age: int | None = 2000,
        groups: Groups | None = None,
    ) -> None:
        super().__init__(model, metric, imputer, tracker, n_samples, seed, max_age, groups)
        _check_no_labels(metric)
        self._means: dict[typing.Any, Tracker] = {}
        self._n_means = 0

    def _mean_prediction(self, y_pred: typing.Any) -> typing.Any:
        """Update the running mean of the predictions, and give its current value."""
        is_dict = isinstance(y_pred, dict)
        values = y_pred if is_dict else {None: y_pred}
        for key in {**self._means, **values}:
            if key not in self._means:
                self._means[key] = copy.deepcopy(self.tracker)
                # A new class had a probability of 0 until now, and all must still add up to 1
                for _ in range(self._n_means):
                    self._means[key].update(0.0)
            self._means[key].update(_number(values.get(key, 0.0)))
        self._n_means += 1
        if is_dict:
            return {key: mean.get() for key, mean in self._means.items()}
        return self._means[None].get()

    def _explain(self, x: dict, y: typing.Any, y_pred: typing.Any, grouped: dict) -> None:
        """Reveal the features or groups of an instance in a random order."""
        previous_loss = self._loss(y, self._mean_prediction(y_pred))
        order = list(grouped)
        self._rng.shuffle(order)
        contributions = {}
        for i, name in enumerate(order, start=1):
            hidden = [feature for later in order[i:] for feature in self._features(later, grouped)]
            # With no feature left to hide, every sample would give `y_pred`
            y_mean = y_pred
            if hidden:
                y_mean = _average(
                    [
                        self._predict(self.imputer.impute_one(x, hidden))
                        for _ in range(self.n_samples)
                    ]
                )
            loss = self._loss(y, y_mean)
            contributions[name] = previous_loss - loss
            previous_loss = loss
        # In the order of `x`, so that the order of `importances` is not random
        for name in grouped:
            self._update_importance(name, contributions[name])

    def learn_one(self, x: dict, y: typing.Any) -> None:
        grouped = self._see(x)
        if self.imputer.ready:
            y_pred = self._predict(x)
            # A River classifier that has not learned yet predicts None or {}
            if y_pred is not None and y_pred != {}:
                self._explain(x, y, y_pred, grouped)
        self.imputer.learn_one(x)

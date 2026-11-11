"""Global explainers, which measure how much a model relies on each feature over a stream."""

from __future__ import annotations

import copy
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
]


class PFI(ImportanceExplainer):
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
        are correlated, as each one alone can be recovered from the others.

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
        groups: dict[typing.Hashable, list[Feature]] | None = None,
    ) -> None:
        super().__init__(model)
        self._predict = _predictor(model, metric)
        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {n_samples}")
        if imputer is not None and seed is not None:
            raise ValueError("seed only applies to the default imputer, give it to your imputer")
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
        if imputer is None:
            rng = random.Random(seed)
            storage = GeometricReservoir(seed=rng.getrandbits(32))
            imputer = Marginal(storage, seed=rng.getrandbits(32))
        self.imputer = imputer
        self.tracker = EWMean() if tracker is None else tracker
        self._trackers: dict[Feature, Tracker] = {}
        self._sign = -1.0 if metric.bigger_is_better else 1.0
        self._n_seen = 0
        self._last_seen: dict[Feature, int] = {}

    def _loss(self, y: typing.Any, y_pred: typing.Any) -> float:
        self.metric.update(y, y_pred)
        loss = self.metric.get()
        self.metric.revert(y, y_pred)
        return self._sign * loss

    def learn_one(self, x: dict, y: typing.Any) -> None:
        grouped = _group(x, self.groups)
        self._n_seen += 1
        for name in grouped:
            self._last_seen[name] = self._n_seen

        if self.imputer.ready:
            y_pred = self._predict(x)
            # A River classifier that has not learned yet predicts None or {}
            if y_pred is not None and y_pred != {}:
                loss = self._loss(y, y_pred)
                for name, features in grouped.items():
                    imputed_loss = 0.0
                    for _ in range(self.n_samples):
                        imputed = self.imputer.impute_one(x, features)
                        imputed_loss += self._loss(y, self._predict(imputed))
                    if name not in self._trackers:
                        self._trackers[name] = copy.deepcopy(self.tracker)
                    self._trackers[name].update(imputed_loss / self.n_samples - loss)
        self.imputer.learn_one(x)

    @property
    def importances(self) -> dict[Feature, float]:
        """The current importance of each feature or group, without those hidden by `max_age`."""
        return {
            feature: tracker.get()
            for feature, tracker in self._trackers.items()
            if self.max_age is None or self._n_seen - self._last_seen[feature] < self.max_age
        }

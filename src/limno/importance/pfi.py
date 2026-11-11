"""Incremental permutation feature importance (iPFI)."""

from __future__ import annotations

import typing

from limno.base import Imputer, Metric, Model, Tracker
from limno.importance._base import Groups, _LossImportance


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

"""What the explainers that compare losses have in common."""

from __future__ import annotations

import collections.abc
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

"""Base classes and protocols shared across limno.

limno follows River's conventions: features are a dictionary `x`, and everything is processed one
instance at a time through `*_one` methods.

Explainers only rely on the `Model` and `Metric` protocols, not on a library. River models and
metrics satisfy them as they are. Models from other libraries can be used through an adapter that
exposes the same methods.

"""

from __future__ import annotations

import abc
import collections.abc
import typing
from collections.abc import Hashable

__all__ = [
    "Explainer",
    "Feature",
    "ImportanceExplainer",
    "Imputer",
    "Metric",
    "Model",
    "Storage",
    "Tracker",
]

Feature = Hashable
"""The name of a feature, i.e. a key of `x`."""


@typing.runtime_checkable
class Model(typing.Protocol):
    """An incremental model.

    Any River model satisfies this protocol, including pipelines. To be scored with a metric that
    needs probabilities, such as `river.metrics.LogLoss`, a model also needs a
    `predict_proba_one(x)` method.

    """

    def predict_one(self, x: dict) -> typing.Any: ...


@typing.runtime_checkable
class Metric(typing.Protocol):
    """A metric that can score one instance at a time.

    Any River metric satisfies this protocol. It tells which way is better with
    `bigger_is_better`. A metric that has a `requires_labels` attribute set to `False` is given
    probabilities instead of predicted labels, as in River.

    """

    bigger_is_better: bool

    def update(self, y_true: typing.Any, y_pred: typing.Any) -> typing.Any: ...

    def revert(self, y_true: typing.Any, y_pred: typing.Any) -> typing.Any: ...

    def get(self) -> float: ...

    def clone(self) -> Metric: ...


@typing.runtime_checkable
class Tracker(typing.Protocol):
    """A running statistic of a series of numbers.

    Statistics from `river.stats` satisfy this protocol.

    """

    def update(self, x: float) -> typing.Any: ...

    def get(self) -> float: ...


class Storage(abc.ABC):
    """Base class for storages, which keep some of the past instances of a stream."""

    def __init__(self) -> None:
        self.instances: list[dict] = []

    def __len__(self) -> int:
        return len(self.instances)

    @abc.abstractmethod
    def update(self, x: dict) -> None:
        """Offer an instance to the storage, which may keep it.

        Parameters
        ----------
        x
            A dictionary of features.

        """


class Imputer(abc.ABC):
    """Base class for imputers, which replace the values of some features of an instance."""

    @property
    def ready(self) -> bool:
        """Whether the imputer can impute."""
        return True

    @abc.abstractmethod
    def impute_one(self, x: dict, features: collections.abc.Iterable[Feature]) -> dict:
        """Replace the values of some features.

        Parameters
        ----------
        x
            A dictionary of features. It is not changed.
        features
            The features to replace.

        Returns
        -------
        A new dictionary of features.

        """

    def learn_one(self, x: dict) -> None:
        """Update the imputer with an instance of the stream. By default, it does nothing.

        Parameters
        ----------
        x
            A dictionary of features.

        """


def _check_groups(groups: collections.abc.Mapping) -> None:
    """Check that every group has features, and that no feature is in two groups."""
    seen = set()
    for name, features in groups.items():
        if not features:
            raise ValueError(f"group {name!r} has no features")
        for feature in features:
            if feature in seen:
                raise ValueError(f"feature {feature!r} is in more than one group")
            seen.add(feature)


def _group(x: dict, groups: collections.abc.Mapping) -> dict[Hashable, list[Feature]]:
    """The features of `x` by group. A feature that is in no group is a group of its own."""
    group_of = {feature: name for name, features in groups.items() for feature in features}
    grouped: dict[Hashable, list[Feature]] = {}
    for feature in x:
        if feature not in group_of and feature in groups:
            raise ValueError(f"{feature!r} is both a feature and the name of a group")
        grouped.setdefault(group_of.get(feature, feature), []).append(feature)
    return grouped


def _check_model(model: typing.Any) -> None:
    """Check that `model` can predict."""
    if not isinstance(model, Model):
        raise TypeError(f"model must have a predict_one method, got {type(model).__name__}")


def _check_river_metric(model: Model, metric: Metric) -> None:
    """Check that River allows `metric` with `model`, if `model` is from River."""
    try:
        from river import base as river_base
    except ImportError:
        return
    works_with = getattr(metric, "works_with", None)
    if isinstance(model, river_base.Estimator) and works_with and not works_with(model):
        raise ValueError(f"{type(metric).__name__} does not work with {type(model).__name__}")


def _predictor(model: Model, metric: Metric) -> collections.abc.Callable[[dict], typing.Any]:
    """The method of `model` that gives what `metric` expects, following River's rule."""
    if not isinstance(metric, Metric):
        raise TypeError(
            "metric must have update, revert, get and clone methods and a bigger_is_better "
            f"attribute, got {type(metric).__name__}"
        )
    # An adapter around a River model can have its own check, as River does not know the adapter
    check_metric = getattr(model, "_check_metric", None)
    if check_metric is not None:
        check_metric(metric)
    else:
        _check_river_metric(model, metric)

    if getattr(metric, "requires_labels", True):
        return model.predict_one
    predict_proba_one = getattr(model, "predict_proba_one", None)
    if predict_proba_one is None:
        raise TypeError(
            f"{type(metric).__name__} needs probabilities, but {type(model).__name__} has no "
            "predict_proba_one method"
        )
    return predict_proba_one


class Explainer(abc.ABC):
    """Base class for local explainers, which explain one prediction.

    A local explainer scores how much each feature of an instance contributes to the prediction of
    a model. It keeps a reference to the model, so its explanations follow the model as it learns.
    The explainer never updates the model: that is up to the code that runs the stream.

    Parameters
    ----------
    model
        The model to explain.

    Examples
    --------
    >>> class Linear:
    ...     def predict_one(self, x):
    ...         return 2 * x.get("a", 0) + x.get("b", 0)

    >>> class DropFeature(Explainer):
    ...     def explain_one(self, x):
    ...         y_pred = self.model.predict_one(x)
    ...         return {
    ...             f: y_pred - self.model.predict_one({k: v for k, v in x.items() if k != f})
    ...             for f in x
    ...         }

    >>> DropFeature(Linear()).explain_one({"a": 1, "b": 3})
    {'a': 2, 'b': 3}

    """

    def __init__(self, model: Model) -> None:
        _check_model(model)
        self.model = model

    @abc.abstractmethod
    def explain_one(self, x: dict) -> dict[Feature, float]:
        """Explain the prediction of the model for an instance.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The contribution of each feature to the prediction.

        """

    def learn_one(self, x: dict, y: typing.Any) -> None:
        """Update the explainer with an instance of the stream.

        By default, it does nothing. Explainers that keep a state, such as a sample of past
        instances, update it here.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            The target of the instance.

        """


class ImportanceExplainer(abc.ABC):
    """Base class for global explainers, which measure how much a model relies on each feature.

    It learns from the stream with `learn_one`, and `importances` gives the current values. It keeps
    a reference to the model, but never updates it: that is up to the code that runs the stream.

    Parameters
    ----------
    model
        The model to explain.

    """

    def __init__(self, model: Model) -> None:
        _check_model(model)
        self.model = model

    @abc.abstractmethod
    def learn_one(self, x: dict, y: typing.Any) -> None:
        """Update the importances with an instance of the stream.

        Call this before the model learns from the instance, so that the instance is new to the
        model.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            The target of the instance.

        """

    @property
    @abc.abstractmethod
    def importances(self) -> dict[Feature, float]:
        """The current importance of each feature."""

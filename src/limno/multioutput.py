"""Wrappers that explain each output of a multi-output model on its own."""

from __future__ import annotations

import collections.abc
import typing
from collections.abc import Hashable

from limno.base import Feature, ImportanceExplainer, Metric, Model, _check_model

__all__ = [
    "PerOutput",
]


def _stand_in(model: typing.Any) -> typing.Any:
    """A River model with one output of the same kind as `model`, or `None` if it is not from River.

    River's `works_with` rejects a metric for one output when given a multi-output model, so it is
    given this model instead.

    """
    try:
        from river import base as river_base
        from river import linear_model
    except ImportError:
        return None
    if isinstance(model, river_base.MultiTargetRegressor):
        return linear_model.LinearRegression()
    if isinstance(model, river_base.MultiLabelClassifier):
        return linear_model.LogisticRegression()
    if isinstance(model, river_base.Estimator):
        raise TypeError(f"model must have many outputs, got {type(model).__name__}")
    return None


class _Output:
    """One output of a multi-output model, shown as a model with one output."""

    def __init__(self, model: typing.Any, output: Hashable, stand_in: typing.Any) -> None:
        self.model = model
        self.output = output
        self._stand_in = stand_in

    def predict_one(self, x: dict) -> typing.Any:
        return self.model.predict_one(x).get(self.output)

    @property
    def predict_proba_one(self) -> collections.abc.Callable[[dict], typing.Any]:
        # A property, so that this is missing if the model does not have the method
        predict_proba_one = self.model.predict_proba_one
        return lambda x: predict_proba_one(x).get(self.output)

    def _check_metric(self, metric: Metric) -> None:
        works_with = getattr(metric, "works_with", None)
        if self._stand_in is not None and works_with and not works_with(self._stand_in):
            raise ValueError(
                f"{type(metric).__name__} does not work with one output of "
                f"{type(self.model).__name__}"
            )


class PerOutput:
    """Measure how much each output of a multi-output model relies on each feature.

    There is one explainer per output, which is created when the output first appears in `y`. For
    one value per feature over all the outputs, use a global explainer with a metric from
    `river.metrics.multioutput` instead.

    Each output costs as much as a model with one output, so `PFI` takes
    `n_outputs * (1 + n_features * n_samples)` predictions per instance. Each output also keeps its
    own explainer state, such as past instances.

    In a chain, a later output uses the predictions of earlier ones. So the importance of a feature
    for an output includes its effect through earlier outputs.

    This is not part of iPFI, whose paper has examples with one output.

    Parameters
    ----------
    model
        The model to explain, whose `predict_one` gives a dictionary with one value per output. It
        is not updated, but explanations follow it as it learns.
    make_explainer
        Creates the explainer of an output. It is given a model with that one output. Use a metric
        for one output here, such as `river.metrics.MAE`.

    Examples
    --------
    >>> import random

    >>> from river import linear_model, metrics, multioutput

    >>> from limno.importance import PFI

    >>> model = multioutput.RegressorChain(linear_model.LinearRegression(), order=["y1", "y2"])
    >>> explainer = PerOutput(model, lambda output: PFI(output, metrics.MAE(), seed=42))

    >>> rng = random.Random(42)
    >>> for _ in range(2000):
    ...     x = {"a": rng.random(), "b": rng.random()}
    ...     y = {"y1": 3 * x["a"], "y2": 3 * x["b"]}
    ...     explainer.learn_one(x, y)
    ...     model.learn_one(x, y)

    >>> {output: max(values, key=values.get) for output, values in explainer.importances.items()}
    {'y1': 'a', 'y2': 'b'}

    """

    def __init__(
        self,
        model: Model,
        make_explainer: collections.abc.Callable[[Model], ImportanceExplainer],
    ) -> None:
        _check_model(model)
        self.model = model
        self.make_explainer = make_explainer
        self.explainers: dict[Hashable, ImportanceExplainer] = {}
        self._stand_in = _stand_in(model)

    def learn_one(self, x: dict, y: dict) -> None:
        """Update the explainers with an instance of the stream.

        Call this before the model learns from the instance, so that the instance is new to the
        model. Only the outputs in `y` are updated.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            A dictionary with the target of each output.

        """
        for output, y_output in y.items():
            if output not in self.explainers:
                explainer = self.make_explainer(_Output(self.model, output, self._stand_in))
                if not isinstance(explainer, ImportanceExplainer):
                    raise TypeError(
                        "make_explainer must give an ImportanceExplainer, got "
                        f"{type(explainer).__name__}"
                    )
                self.explainers[output] = explainer
            self.explainers[output].learn_one(x, y_output)

    @property
    def importances(self) -> dict[Hashable, dict[Feature, float]]:
        """The current importance of each feature, for each output seen so far."""
        return {output: explainer.importances for output, explainer in self.explainers.items()}

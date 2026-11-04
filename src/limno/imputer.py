"""Imputers, which replace the values of some features of an instance."""

from __future__ import annotations

import random
from collections.abc import Iterable

from limno.base import Feature, Imputer, Storage
from limno.storage import GeometricReservoir

__all__ = [
    "Marginal",
]


class Marginal(Imputer):
    """Replace features with their values in a random past instance.

    The replaced features of an instance are always taken from the same past instance. If that
    instance does not have a feature, the feature is removed, as River models accept instances
    with missing features.

    Parameters
    ----------
    storage
        Where the past instances are kept. By default, a `storage.GeometricReservoir`.
    seed
        Random seed for picking past instances.

    Examples
    --------
    >>> imputer = Marginal(seed=42)
    >>> imputer.learn_one({"a": 1, "b": 2})
    >>> imputer.impute_one({"a": 5, "b": 6}, ["b"])
    {'a': 5, 'b': 2}

    """

    def __init__(self, storage: Storage | None = None, seed: int | None = None) -> None:
        self.storage = GeometricReservoir() if storage is None else storage
        self.seed = seed
        self._rng = random.Random(seed)

    @property
    def ready(self) -> bool:
        """Whether there is a past instance to take values from."""
        return len(self.storage) > 0

    def impute_one(self, x: dict, features: Iterable[Feature]) -> dict:
        if not self.ready:
            raise ValueError("the storage is empty, call learn_one first")
        past = self._rng.choice(self.storage.instances)
        imputed = dict(x)
        for feature in features:
            if feature in past:
                imputed[feature] = past[feature]
            else:
                imputed.pop(feature, None)
        return imputed

    def learn_one(self, x: dict) -> None:
        self.storage.update(x)

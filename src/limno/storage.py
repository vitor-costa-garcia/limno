"""Storages, which keep some of the past instances of a stream using a fixed amount of memory."""

from __future__ import annotations

import random

from limno.base import Storage

__all__ = [
    "GeometricReservoir",
    "UniformReservoir",
]


class _Reservoir(Storage):
    def __init__(self, size: int = 100, seed: int | None = None) -> None:
        if size < 1:
            raise ValueError(f"size must be at least 1, got {size}")
        super().__init__()
        self.size = size
        self.seed = seed
        self._rng = random.Random(seed)


class UniformReservoir(_Reservoir):
    """Keep a sample in which every past instance is equally likely to be found.

    This suits a stream that does not change, since old and recent instances count the same.

    Parameters
    ----------
    size
        The number of instances to keep.
    seed
        Random seed.

    Examples
    --------
    >>> storage = UniformReservoir(size=3, seed=42)
    >>> for i in range(100):
    ...     storage.update({"a": i})
    >>> len(storage)
    3

    """

    def __init__(self, size: int = 100, seed: int | None = None) -> None:
        super().__init__(size, seed)
        self._n_seen = 0

    def update(self, x: dict) -> None:
        self._n_seen += 1
        if len(self.instances) < self.size:
            self.instances.append(dict(x))
            return
        i = self._rng.randrange(self._n_seen)
        if i < self.size:
            self.instances[i] = dict(x)


class GeometricReservoir(_Reservoir):
    """Keep a sample that favours recent instances, which suits a stream that changes.

    Once full, each new instance replaces a random one. So an instance is kept for about `size`
    steps, and the chance that it is still kept falls by `1 - 1 / size` at each step.

    Parameters
    ----------
    size
        The number of instances to keep.
    seed
        Random seed.

    Examples
    --------
    >>> storage = GeometricReservoir(size=3, seed=42)
    >>> for i in range(100):
    ...     storage.update({"a": i})
    >>> len(storage)
    3

    """

    def update(self, x: dict) -> None:
        if len(self.instances) < self.size:
            self.instances.append(dict(x))
        else:
            self.instances[self._rng.randrange(self.size)] = dict(x)

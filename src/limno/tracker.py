"""Trackers, which keep a running statistic of a series of numbers.

Statistics from `river.stats` can be used as trackers too.

"""

from __future__ import annotations

__all__ = [
    "EWMean",
    "Mean",
]


class Mean:
    """The mean of all the values so far, which suits a stream that does not change.

    Examples
    --------
    >>> tracker = Mean()
    >>> for value in [1.0, 2.0, 6.0]:
    ...     tracker.update(value)
    >>> tracker.get()
    3.0

    """

    def __init__(self) -> None:
        self._mean = 0.0
        self._n = 0

    def update(self, x: float) -> None:
        self._n += 1
        self._mean += (x - self._mean) / self._n

    def get(self) -> float:
        return self._mean


class EWMean:
    """An exponentially weighted mean, which favours recent values and so follows changes.

    The weights are normalized. So it does not start from zero or from the first value: at the
    start it is close to a plain mean, and over time it becomes the usual
    `(1 - alpha) * mean + alpha * x`.

    Parameters
    ----------
    alpha
        The weight of a new value, between 0 and 1. Larger values follow changes faster, but are
        noisier. It covers about the last `2 / alpha` values.

    Examples
    --------
    >>> tracker = EWMean(alpha=0.5)
    >>> tracker.update(3.0)
    >>> tracker.get()
    3.0
    >>> tracker.update(6.0)
    >>> tracker.get()
    5.0

    """

    def __init__(self, alpha: float = 0.001) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be above 0 and at most 1, got {alpha}")
        self.alpha = alpha
        self._sum = 0.0
        self._weight = 0.0

    def update(self, x: float) -> None:
        self._sum = (1 - self.alpha) * self._sum + self.alpha * x
        self._weight = (1 - self.alpha) * self._weight + self.alpha

    def get(self) -> float:
        return self._sum / self._weight if self._weight else 0.0

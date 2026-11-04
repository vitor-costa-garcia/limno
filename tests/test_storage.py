import collections

import pytest

import limno as lm

RESERVOIRS = [lm.storage.UniformReservoir, lm.storage.GeometricReservoir]


def mean_kept_position(cls, n=1000, size=50, repeats=50):
    """The mean position in the stream of the kept instances, as a share of the stream length."""
    total = 0.0
    for seed in range(repeats):
        storage = cls(size=size, seed=seed)
        for i in range(n):
            storage.update({"i": i})
        total += sum(x["i"] for x in storage.instances) / size
    return total / repeats / n


@pytest.mark.parametrize("cls", RESERVOIRS)
def test_fills_up_then_keeps_its_size(cls):
    storage = cls(size=3, seed=0)
    for i in range(3):
        storage.update({"i": i})
    assert storage.instances == [{"i": 0}, {"i": 1}, {"i": 2}]
    for i in range(3, 100):
        storage.update({"i": i})
    assert len(storage) == 3


@pytest.mark.parametrize("cls", RESERVOIRS)
def test_rejects_bad_size(cls):
    with pytest.raises(ValueError, match="size must be at least 1, got 0"):
        cls(size=0)


@pytest.mark.parametrize("cls", RESERVOIRS)
def test_seed_makes_it_reproducible(cls):
    a, b = cls(size=5, seed=1), cls(size=5, seed=1)
    for i in range(200):
        a.update({"i": i})
        b.update({"i": i})
    assert a.instances == b.instances


@pytest.mark.parametrize("cls", RESERVOIRS)
def test_keeps_a_copy(cls):
    x = {"a": 1}
    storage = cls(size=1, seed=0)
    storage.update(x)
    x["a"] = 2
    assert storage.instances == [{"a": 1}]


def test_uniform_reservoir_keeps_every_instance_equally_often():
    counts = collections.Counter()
    for seed in range(3000):
        storage = lm.storage.UniformReservoir(size=1, seed=seed)
        for i in range(4):
            storage.update({"i": i})
        counts[storage.instances[0]["i"]] += 1
    assert set(counts) == {0, 1, 2, 3}
    for count in counts.values():
        assert count == pytest.approx(750, abs=100)


def test_reservoirs_differ_in_how_recent_their_instances_are():
    assert mean_kept_position(lm.storage.UniformReservoir) == pytest.approx(0.5, abs=0.02)
    assert mean_kept_position(lm.storage.GeometricReservoir) > 0.9


def test_geometric_reservoir_keeps_an_instance_for_about_its_size():
    # Each step must replace a slot. With a replacement chance of 1 / size on top of that, as in
    # iXAI, an instance would stay for about size ** 2 steps.
    size = 20
    storage = lm.storage.GeometricReservoir(size=size, seed=0)
    for i in range(size):
        storage.update({"i": i})
    ages = []
    for i in range(size, 20000):
        storage.update({"i": i})
        assert {"i": i} in storage.instances
        ages.extend(i - x["i"] for x in storage.instances)
    assert sum(ages) / len(ages) == pytest.approx(size, rel=0.1)

import pytest

import limno as lm


def test_replaces_only_the_given_features_and_keeps_x():
    imputer = lm.imputer.Marginal(seed=0)
    imputer.learn_one({"a": 1, "b": 2, "c": 3})
    x = {"a": 10, "b": 20, "c": 30}
    assert imputer.impute_one(x, ["a", "c"]) == {"a": 1, "b": 20, "c": 3}
    assert x == {"a": 10, "b": 20, "c": 30}


def test_features_come_from_the_same_past_instance():
    imputer = lm.imputer.Marginal(lm.storage.UniformReservoir(size=50, seed=0), seed=0)
    for i in range(50):
        imputer.learn_one({"a": i, "b": i})
    for _ in range(100):
        imputed = imputer.impute_one({"a": -1, "b": -1}, ["a", "b"])
        assert imputed["a"] == imputed["b"]


def test_feature_missing_from_the_past_instance_is_removed():
    imputer = lm.imputer.Marginal(seed=0)
    imputer.learn_one({"a": 1})
    assert imputer.impute_one({"a": 10, "b": 20}, ["b"]) == {"a": 10}


def test_feature_missing_from_x_is_added():
    imputer = lm.imputer.Marginal(seed=0)
    imputer.learn_one({"a": 1, "b": 2})
    assert imputer.impute_one({"a": 10}, ["b"]) == {"a": 10, "b": 2}


def test_empty_storage():
    imputer = lm.imputer.Marginal()
    assert not imputer.ready
    with pytest.raises(ValueError, match="storage is empty"):
        imputer.impute_one({"a": 1}, ["a"])
    imputer.learn_one({"a": 1})
    assert imputer.ready


def test_seed_makes_it_reproducible():
    def run():
        imputer = lm.imputer.Marginal(lm.storage.UniformReservoir(size=20, seed=1), seed=1)
        for i in range(20):
            imputer.learn_one({"a": i})
        return [imputer.impute_one({"a": -1}, ["a"]) for _ in range(20)]

    assert run() == run()

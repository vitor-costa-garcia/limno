import pytest

import limno as lm


class MeanModel:
    def __init__(self):
        self.total = 0.0
        self.n = 0

    def predict_one(self, x):
        return self.total / self.n if self.n else 0.0

    def learn_one(self, x, y):
        self.total += y
        self.n += 1


class PredictionExplainer(lm.base.Explainer):
    def explain_one(self, x):
        return dict.fromkeys(x, self.model.predict_one(x))


class TrainPredict:
    def train(self, instance):
        pass

    def predict(self, instance):
        pass


def test_model_protocol_only_needs_predict_one():
    class PredictOnly:
        def predict_one(self, x):
            return 0.0

    assert isinstance(PredictOnly(), lm.base.Model)
    assert not isinstance(TrainPredict(), lm.base.Model)


def test_explainer_requires_explain_one():
    with pytest.raises(TypeError, match="explain_one"):
        lm.base.Explainer(MeanModel())


@pytest.mark.parametrize("cls", [PredictionExplainer, lm.importance.PFI])
def test_explainers_reject_object_that_is_not_a_model(cls):
    args = (object(),) if cls is lm.importance.PFI else ()
    with pytest.raises(TypeError, match="predict_one.*TrainPredict"):
        cls(TrainPredict(), *args)


def test_explanations_follow_the_model():
    model = MeanModel()
    explainer = PredictionExplainer(model)
    assert explainer.explain_one({"a": 1}) == {"a": 0.0}
    model.learn_one({"a": 1}, 4.0)
    assert explainer.explain_one({"a": 1}) == {"a": 4.0}


def test_explainer_learn_one_does_not_update_the_model():
    model = MeanModel()
    PredictionExplainer(model).learn_one({"a": 1}, 4.0)
    assert model.n == 0


def test_importance_explainer_requires_its_methods():
    with pytest.raises(TypeError, match="importances.*learn_one|learn_one.*importances"):
        lm.base.ImportanceExplainer(MeanModel())


def test_imputer_is_ready_and_ignores_learn_one_by_default():
    class Zero(lm.base.Imputer):
        def impute_one(self, x, features):
            return {**x, **dict.fromkeys(features, 0)}

    imputer = Zero()
    assert imputer.ready
    assert imputer.learn_one({"a": 1}) is None
    assert imputer.impute_one({"a": 1, "b": 2}, ["a"]) == {"a": 0, "b": 2}

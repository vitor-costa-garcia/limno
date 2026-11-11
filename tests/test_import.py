import subprocess
import sys

import limno


def test_version():
    assert limno.__version__


def test_works_without_river():
    code = """
import sys

sys.modules["river"] = None
import limno as lm


class Fixed:
    def predict_one(self, x):
        return 2 * x["a"]


class AbsoluteError:
    bigger_is_better = False

    def __init__(self):
        self.total, self.n = 0.0, 0

    def update(self, y_true, y_pred):
        self.total += abs(y_true - y_pred)
        self.n += 1

    def revert(self, y_true, y_pred):
        self.total -= abs(y_true - y_pred)
        self.n -= 1

    def get(self):
        return self.total / self.n

    def clone(self):
        return AbsoluteError()


explainer = lm.importance.PFI(Fixed(), AbsoluteError(), seed=0)
explainer.learn_one({"a": 0.0}, 0.0)
explainer.learn_one({"a": 1.0}, 2.0)
assert explainer.importances["a"] == 2.0


class Outputs:
    def predict_one(self, x):
        return {"y": 2 * x["a"]}


explainer = lm.multioutput.PerOutput(
    Outputs(), lambda output: lm.importance.PFI(output, AbsoluteError(), seed=0)
)
explainer.learn_one({"a": 0.0}, {"y": 0.0})
explainer.learn_one({"a": 1.0}, {"y": 2.0})
assert explainer.importances["y"]["a"] == 2.0

explainer = lm.importance.SAGE(Fixed(), AbsoluteError(), seed=0)
explainer.learn_one({"a": 0.0}, 0.0)
explainer.learn_one({"a": 1.0}, 2.0)
assert explainer.importances["a"] == 0.0
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_does_not_load_river():
    code = "import sys, limno; assert 'river' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

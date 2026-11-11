# Quickstart

## Feature importance on a stream

`lm.importance.PFI` measures how much a model relies on each feature, as the stream goes on:

```python
from river import datasets, linear_model, metrics, preprocessing

import limno as lm

model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
explainer = lm.importance.PFI(model, metrics.LogLoss(), seed=42)

for x, y in datasets.Phishing():
    explainer.learn_one(x, y)  # before the model learns, so the instance is new to it
    model.learn_one(x, y)

explainer.importances
```

```
{'empty_server_form_handler': 0.177, 'popup_window': 0.11, 'https': 0.106, ...}
```

A positive value means that the model does better with the feature, and a value close to zero that
it does not rely on it. The explainer keeps a reference to the model, so the values follow it as it
learns. It never updates the model: the loop does that.

Whether the model is asked for labels or probabilities depends on the metric, as in River. Here,
`LogLoss` needs probabilities, and `Accuracy` would need labels. The metric is computed on one
instance at a time, so use one that makes sense for a single instance, not `ROCAUC` or `F1`.

Each instance takes `1 + n_features * n_samples` predictions, which can be slow with large models.

## Streams that change

The default pieces follow changes in the stream. To react faster, raise `alpha`. If the stream does
not change, use pieces that weigh the whole stream the same:

```python
explainer = lm.importance.PFI(
    model,
    metrics.LogLoss(),
    imputer=lm.imputer.Marginal(lm.storage.UniformReservoir(size=500, seed=1), seed=2),
    tracker=lm.tracker.Mean(),
)
```

| Piece | Stream that changes (default) | Stream that does not change |
|---|---|---|
| Storage | `storage.GeometricReservoir` | `storage.UniformReservoir` |
| Tracker | `tracker.EWMean` | `tracker.Mean` |

Statistics from `river.stats` can also be used as trackers.

## Features that stop appearing

A feature that is missing from an instance gets no update, so its value goes out of date.
`importances` hides a feature once it is missing from `max_age` instances in a row, 2000 by default.
It comes back with its old value if it appears again.

```python
explainer = lm.importance.PFI(model, metrics.LogLoss(), max_age=500)
```

Raise `max_age` for features that are rare, and use `None` to never hide a feature. This is not part
of iPFI, which assumes that the features do not change.

## Groups of features

`groups` treats some features as one. A group is replaced as a whole, from one past instance, and
gets one importance under its name:

```python
explainer = lm.importance.PFI(
    model, metrics.LogLoss(), groups={"location": ["lat", "lon"], "time": ["hour", "weekday"]}
)
```

Features in no group get their own importance. Use groups for features that only make sense
together, such as the columns of a one-hot encoding. Also use them for correlated features: replacing
one of them alone can underrate it, as the model can recover it from the others. Each instance then
takes `1 + n_groups * n_samples` predictions.

## Models with many outputs

`lm.multioutput.PerOutput` gives the importances of each output:

```python
from river import linear_model, metrics, multioutput

model = multioutput.RegressorChain(linear_model.LinearRegression(), order=["y1", "y2"])
explainer = lm.multioutput.PerOutput(
    model, lambda output: lm.importance.PFI(output, metrics.MAE(), seed=42)
)

for x, y in stream:  # `y` is a dictionary, such as {"y1": 1.5, "y2": 0.2}
    explainer.learn_one(x, y)
    model.learn_one(x, y)

explainer.importances
```

```
{'y1': {'a': 0.947, 'b': -0.0, 'c': -0.0}, 'y2': {'a': -0.0, 'b': 0.962, 'c': -0.0}}
```

There is one explainer per output, so this takes `n_outputs` times the predictions and state of a
model with one output. In a chain, the importance of a feature for an output includes its effect
through earlier outputs.

For one value per feature over all the outputs, give `lm.importance.PFI` a metric from
`river.metrics.multioutput`, such as `MicroAverage(metrics.MAE())`.

## Write your own explainer

A local explainer scores the features of one prediction. This one measures how much the prediction
changes when a feature is left out:

```python
class DropFeature(lm.base.Explainer):
    def explain_one(self, x):
        y_pred = self.model.predict_one(x)
        return {
            f: y_pred - self.model.predict_one({k: v for k, v in x.items() if k != f}) for f in x
        }
```

Global explainers extend `lm.base.ImportanceExplainer`, with `learn_one` and `importances`.

## Other libraries

Explainers only need a model with `predict_one`, and `predict_proba_one` for metrics that need
probabilities. Models from a library with other methods can be used through a small class that
exposes these.

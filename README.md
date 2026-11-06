# limno

[![CI](https://github.com/vitor-costa-garcia/limno/actions/workflows/ci.yml/badge.svg)](https://github.com/vitor-costa-garcia/limno/actions/workflows/ci.yml)

**Explainable AI for data streams and continual learning.**

> limno is in early development: the API may change.

## What is this?

A model that learns from a stream keeps changing, and so do the reasons behind its predictions.
Most explanation tools assume a fixed model and a fixed dataset, so they do not fit this setting.
limno provides explanations that are computed one instance at a time and that follow the model as
it learns. Limnology is the study of inland waters, and limno looks at what goes on inside a
stream.

limno follows [River](https://riverml.xyz)'s conventions and works with River's models, pipelines,
metrics and statistics as they are. It complements River rather than replacing it. Other libraries,
such as [CapyMOA](https://capymoa.org), are not supported yet, but explainers do not depend on
River, so an adapter can add them later.

## Features

- **Incremental permutation feature importance** (`lm.importance.PFI`), from
  [Fumagalli et al. (2023)](https://doi.org/10.1007/s10994-023-06385-y). It shows how much a model
  relies on each feature, as the stream goes on.
- **Models with many outputs** (`lm.multioutput.PerOutput`), to see which features each output
  relies on.
- **River metrics as the loss.** Whether the model is asked for labels or probabilities depends on
  the metric, as in River.
- **Separate pieces** that can be swapped:
  - Storages (`lm.storage`) keep past instances: `UniformReservoir` and `GeometricReservoir`.
  - Imputers (`lm.imputer`) replace feature values: `Marginal`.
  - Trackers (`lm.tracker`) keep running values: `Mean`, `EWMean`, or any of `river.stats`.
- **Features can appear and disappear** along the stream, with no list of names to give upfront. A
  feature that stops appearing is hidden after `max_age` instances.
- **Reproducible runs** with a `seed`.
- **A base class for your own explainers** (`lm.base.Explainer` and
  `lm.base.ImportanceExplainer`).

## Installation

limno is not on PyPI yet. To install it from a clone:

```bash
pip install -e ".[river]"
```

The `river` extra installs River. limno itself has no required dependencies.

## Quickstart

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
it does not rely on it. The explainer never updates the model.

## Notebooks

The [notebooks](https://github.com/vitor-costa-garcia/limno/tree/main/notebooks) show limno in use
with River. [How incremental PFI works](https://github.com/vitor-costa-garcia/limno/blob/main/notebooks/01_how_pfi_works.ipynb)
explains the method step by step.

## Contributing and feedback

Ideas, use cases and criticism are welcome through
[GitHub issues](https://github.com/vitor-costa-garcia/limno/issues).

## License

BSD 3-Clause. See [LICENSE](https://github.com/vitor-costa-garcia/limno/blob/main/LICENSE).

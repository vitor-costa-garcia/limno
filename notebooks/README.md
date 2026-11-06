# Notebooks

Examples of limno in use, with [River](https://riverml.xyz) for the data, models and metrics.

| Notebook | What it shows |
|---|---|
| [01_how_pfi_works.ipynb](01_how_pfi_works.ipynb) | How `lm.importance.PFI` works, step by step: one instance by hand, running means, following a model as it learns, the storage and the tracker, drift, River classifiers and metrics, models with many outputs, its cost, and a comparison with batch PFI |

To run them, install limno from a clone of this repository, along with what the notebooks use. They
were last run with River 0.26:

```bash
pip install -e ".[river]" matplotlib pandas jupyter
jupyter lab notebooks/
```

# CHANGELOG



## Unreleased



## v0.1.0 (2026-11-11)

### Feature

* Add `lm.base` with the `Model`, `Metric` and `Tracker` protocols, and the `Explainer`, `ImportanceExplainer`, `Storage` and `Imputer` base classes

* Add `lm.importance.PFI`, the incremental permutation feature importance (iPFI) of Fumagalli et al. (2023)

* Add `lm.importance.SAGE`, the interventional version of incremental SAGE (iSAGE) of Muschalik et al. (2023)

* Add `max_age` to `lm.importance.PFI`, which hides features that have stopped appearing

* Add `groups` to `lm.importance.PFI` and `lm.importance.SAGE`, which treats some features as one

* Add `lm.multioutput` with `PerOutput`, which gives the importances of each output of a model

* Add `lm.storage` with `UniformReservoir` and `GeometricReservoir`

* Add `lm.imputer` with `Marginal`

* Add `lm.tracker` with `Mean` and `EWMean`

### Documentation

* Add a notebook on how `lm.importance.PFI` works

* Add a notebook on how `lm.importance.SAGE` works

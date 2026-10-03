# Model Card · Fuel Economy & CO₂ Prediction (Fuel Economy Guide)

**English** · [Deutsch](model_card.de.md)

## Summary

| | |
|---|---|
| **Task** | Regression of certified combined fuel economy (mpg) and CO₂ emissions (g/mi) |
| **Model** | `GradientBoostingRegressor` (scikit-learn) with one-hot encoded categorical features, inside a `Pipeline` |
| **Hyperparameters** | `n_estimators=400`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8` (selected with `GroupKFold` on training data) |
| **Data** | Fuel Economy Guide 2024 (EPA / DOE), 964 combustion vehicle configurations, 596 model families |
| **Notebook** | `notebooks/02_modelado_consumo_co2.ipynb` |

## Intended use

Early estimation of the certified fuel economy and CO₂ of a combustion vehicle configuration from its technical
specification (engine, transmission, drivetrain, class, fuel, hybridization), before test results are available.
Supports product line definition, comparison of powertrain alternatives and screening of configurations exposed to the
*Gas Guzzler Tax*.

**Out of scope:** battery-electric or plug-in hybrid vehicles, certification or official declaration of fuel
consumption, heavy-duty vehicles, cycles other than the EPA combined cycle (e.g. WLTP) without retraining.

## Input features

| Feature | Type |
|---|---|
| Displacement (L), number of cylinders, number of gears | numeric |
| Full hybrid | binary |
| Fuel, drivetrain, EPA class, air aspiration, cylinder deactivation | categorical (one-hot) |

Features derived from fuel economy itself (city, highway, unadjusted, label ratings, CO₂, gas guzzler flag) are
**excluded** to prevent information leakage.

## Performance

Evaluated on **unseen model families** (`GroupKFold`, 5 folds), so that variants of the same model are never split
between training and validation.

| Target | R² | MAE | Baseline (mean) MAE |
|---|---|---|---|
| Combined fuel economy | 0.91 | 1.41 mpg | 4.95 mpg |
| Combined CO₂ | 0.89 | 23.2 g/mi | — |

Per-segment performance (test split, fuel economy):

| Segment | MAE |
|---|---|
| Conventional | 1.3 mpg |
| Full hybrids | 4.9 mpg |
| Diesel | 0.8 mpg (n = 6) |

## Data validation

The pipeline refuses to train unless the dataset passes the contracts in `vehicle_emissions.validation`: schema, no
nulls, physical ranges, category domains, city/highway/combined consistency, exemption rule and a **row-level physical
consistency check CO₂ ≈ k / mpg** that detects misaligned columns. The contracts run in CI together with tests that
inject defects and assert they are caught.

## Limitations and risks

* **Hybrids:** error almost four times higher than for conventional vehicles. The guide does not report battery
  capacity or electric power, which drive hybrid fuel economy.
* **Diesel:** only 20 vehicles in the dataset; estimates for this fuel carry high uncertainty.
* **Temporal coverage:** a single model year (2024). Technology and regulation change; the model should be retrained
  and monitored for drift with each new edition of the guide.
* **Market:** U.S. certification data (FTP/HWY cycles with EPA adjustment), not directly comparable to European WLTP values.
* **Misuse:** the model does not replace a certification test and must not be used to declare official fuel economy.

## Reproducibility

```bash
pip install -e ".[dev]"
make all   # lint + tests + full notebook execution
```

Fixed seeds (`random_state=42`); CI runs all four notebooks end to end on every change.

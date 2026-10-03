"""Shared evaluation protocol: metrics, grouped cross-validation and plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_validate

from .paths import FIGURES_DIR


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_regressor(name, model, X_train, X_test, y_train, y_test, X_all=None, y_all=None,
                       groups=None, cv_splits=5, random_state=3) -> dict:
    """Evaluate a model on the hold-out split and with cross-validation.

    * Hold-out: the model is fitted on ``X_train`` and scored on ``X_test``.
    * Random CV (``KFold``) on the full dataset.
    * Grouped CV (``GroupKFold``) when ``groups`` is given: no group (model family, tested
      vehicle...) appears in both training and validation. This is the honest estimate of
      performance on new vehicles.

    Scaling, encoding and polynomial features must live inside ``model`` (a ``Pipeline``) so
    they are fitted on the training data of each split only.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    res = {"Experiment": name, **regression_metrics(y_test, y_pred)}
    if X_all is not None:
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        s = cross_validate(clone(model), X_all, y_all, cv=kf, scoring="r2")["test_score"]
        res["R2 CV"] = s.mean()
        res["± std"] = s.std()
        if groups is not None:
            g = cross_validate(clone(model), X_all, y_all, groups=groups, cv=GroupKFold(n_splits=cv_splits),
                               scoring=("r2", "neg_mean_absolute_error"))
            res["R2 grouped CV"] = g["test_r2"].mean()
            res["MAE grouped CV"] = -g["test_neg_mean_absolute_error"].mean()
    res["_y_pred"] = y_pred
    return res


def plot_pred_vs_real(y_true, y_pred, title="", units="", ax=None, save_as=None):
    """Actual vs predicted scatter with the identity line as reference."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ax.scatter(y_true, y_pred, s=28, alpha=0.6, edgecolor="none", label="Observations (test)")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="crimson", lw=1.5, label="Perfect prediction")
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"{title}  (R² = {r2:.3f})" if title else f"R² = {r2:.3f}")
    ax.set_xlabel(f"Actual {units}".strip())
    ax.set_ylabel(f"Predicted {units}".strip())
    ax.legend(loc="upper left")
    if save_as:
        ax.figure.savefig(FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return ax


def plot_residuals(y_true, y_pred, units="", title="", save_as=None):
    """Residuals vs prediction and their distribution: reveals bias and heteroscedasticity."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    res = y_true - y_pred
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].scatter(y_pred, res, s=22, alpha=0.6, edgecolor="none")
    axs[0].axhline(0, color="crimson", ls="--")
    axs[0].set(xlabel=f"Prediction {units}", ylabel=f"Residual (actual − predicted) {units}",
               title="Residuals vs prediction")
    axs[1].hist(res, bins=40, color="#4C72B0")
    axs[1].axvline(0, color="crimson", ls="--")
    axs[1].set(xlabel=f"Residual {units}", ylabel="Count",
               title=f"Distribution (mean={res.mean():.2f}, P5–P95=[{np.percentile(res, 5):.1f}, {np.percentile(res, 95):.1f}])")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return fig


def results_table(results: list[dict]) -> pd.DataFrame:
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    return pd.DataFrame(rows).set_index("Experiment").round(3)


def plot_correlation(df: pd.DataFrame, title="", annot=True, figsize=(12, 10), save_as=None):
    import seaborn as sns

    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                annot=annot, fmt=".2f", annot_kws={"size": 7}, linewidths=0.4, ax=ax,
                cbar_kws={"shrink": 0.7})
    ax.set_title(title)
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150, bbox_inches="tight")
    return corr


def top_correlations(corr: pd.DataFrame, n=10) -> pd.DataFrame:
    """Most strongly correlated variable pairs (positive and negative)."""
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()

    def fmt(s):
        return [f"{a} ↔ {b}: {v:.3f}" for (a, b), v in s.items()]

    return pd.DataFrame({"Most positive": fmt(upper.sort_values(ascending=False).head(n)),
                         "Most negative": fmt(upper.sort_values().head(n))})

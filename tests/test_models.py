"""Tests de regresión del rendimiento: protegen contra degradaciones silenciosas del pipeline.

Los umbrales están por debajo de las métricas actuales con margen, de modo que un cambio que
rompa la ingesta o introduzca un error de codificación los hará fallar.
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

NUM = ["Eng Displ", "# Cyl", "# Gears", "Hybrid"]
CAT = ["Fuel Usage", "Drive Desc", "Carline Class Desc", "Air Aspiration Method Desc", "Cyl Deact?"]
TARGET = "Comb FE (Guide)"


def _pipe(est, scale=True):
    pre = ColumnTransformer([("num", MinMaxScaler() if scale else "passthrough", NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("model", est)])


def _grouped_r2(fe, model):
    X, y, g = fe[NUM + CAT], fe[TARGET].astype(float), fe["Model Family"]
    return cross_val_score(model, X, y, groups=g, cv=GroupKFold(5), scoring="r2").mean()


def test_baseline_has_no_skill(fe):
    assert _grouped_r2(fe, _pipe(DummyRegressor())) < 0.0 + 1e-6


def test_linear_onehot_beats_baseline(fe):
    assert _grouped_r2(fe, _pipe(LinearRegression())) > 0.75


def test_gradient_boosting_is_best(fe):
    lin = _grouped_r2(fe, _pipe(LinearRegression()))
    gb = _grouped_r2(fe, _pipe(GradientBoostingRegressor(random_state=42), scale=False))
    assert gb > 0.80 and gb > lin

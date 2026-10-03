"""Tests de los contratos de calidad de datos.

Además de comprobar que el dataset real los supera, se inyectan defectos típicos
(columna desalineada, valores fuera de rango, nulos, columnas ausentes) y se verifica
que cada uno se detecta.
"""

import numpy as np
import pytest

from vehicle_emissions.data import FE_SENTINEL_THRESHOLD
from vehicle_emissions.validation import (
    DataValidationError,
    check_co2_consistency,
    co2_relative_error,
    validate_epa_test_cars,
    validate_fe_guide,
)


def test_real_dataset_passes_all_checks(fe):
    report = validate_fe_guide(fe)
    assert report["OK"].all()


def test_co2_physics_holds_tightly(fe):
    err = co2_relative_error(fe)
    assert err.median() < 0.01
    assert err.max() < 0.05


@pytest.mark.parametrize("column", ["Comb CO2", "Comb Unrd Adj FE"])
def test_detects_silently_misaligned_column(fe, column):
    """Reordenar una sola columna (error típico al ordenar en una hoja de cálculo) debe fallar."""
    broken = fe.copy()
    broken[column] = broken[column].sample(frac=1, random_state=0).to_numpy()
    assert not check_co2_consistency(broken).passed
    with pytest.raises(DataValidationError):
        validate_fe_guide(broken)


def test_detects_out_of_range(fe):
    broken = fe.copy()
    broken.loc[0, "Eng Displ"] = 42.0
    with pytest.raises(DataValidationError, match="Eng Displ"):
        validate_fe_guide(broken)


def test_detects_nulls(fe):
    broken = fe.copy()
    broken["# Gears"] = broken["# Gears"].astype("Float64")
    broken.loc[3, "# Gears"] = np.nan
    with pytest.raises(DataValidationError, match="nulos"):
        validate_fe_guide(broken)


def test_detects_missing_column(fe):
    with pytest.raises(DataValidationError, match="Hybrid"):
        validate_fe_guide(fe.drop(columns=["Hybrid"]))


def test_report_mode_does_not_raise(fe):
    broken = fe.copy()
    broken.loc[0, "Comb CO2"] = 5000
    report = validate_fe_guide(broken, raise_on_error=False)
    assert not report["OK"].all()


def test_epa_validation_rejects_sentinels(epa):
    with pytest.raises(DataValidationError):
        validate_epa_test_cars(epa)
    clean = epa[~(epa["RND_ADJ_FE"] >= FE_SENTINEL_THRESHOLD)]
    assert validate_epa_test_cars(clean)["OK"].all()

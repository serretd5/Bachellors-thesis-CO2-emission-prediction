"""Validación de datos: contratos de esquema, rangos físicos y coherencia entre columnas.

Cada comprobación devuelve un :class:`Check`. ``validate_fe_guide`` las ejecuta todas y
falla de forma explícita (``DataValidationError``) si alguna no se cumple, de modo que un
cambio en la fuente o un error de transformación se detecta antes de entrenar ningún modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FE_SENTINEL_THRESHOLD

# g de CO2 por galón de combustible (factores de la EPA)
CO2_PER_GALLON = {"gasoline": 8887.0, "diesel": 10180.0}

REQUIRED_FE_COLUMNS = [
    "Eng Displ", "# Cyl", "# Gears", "Trans Desc", "Drive Desc", "Fuel Usage", "Carline Class Desc",
    "Air Aspiration Method Desc", "Comb FE (Guide)", "Comb Unadj FE", "Comb Unrd Adj FE", "Comb CO2",
    "Guzzler?", "Gas Guzzler Exempt", "Hybrid", "Model Family",
]

PHYSICAL_RANGES = {
    "Eng Displ": (0.5, 9.0),
    "# Cyl": (2, 16),
    "# Gears": (1, 12),
    "Comb FE (Guide)": (5, 150),
    "Comb Unadj FE": (5, 200),
    "Comb CO2": (50, 1500),
}

DRIVE_DOMAIN = {"2-Wheel Drive, Front", "2-Wheel Drive, Rear", "All Wheel Drive",
                "4-Wheel Drive", "Part-time 4-Wheel Drive"}


class DataValidationError(ValueError):
    """Se lanza cuando el dataset incumple algún contrato de calidad."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _report(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in checks]).rename(
        columns={"name": "Comprobación", "passed": "OK", "detail": "Detalle"})


def check_schema(df: pd.DataFrame, required=REQUIRED_FE_COLUMNS) -> Check:
    missing = [c for c in required if c not in df.columns]
    return Check("Esquema: columnas requeridas", not missing,
                 f"faltan {missing}" if missing else f"{len(required)} columnas presentes")


def check_no_nulls(df: pd.DataFrame, cols=REQUIRED_FE_COLUMNS) -> Check:
    nulls = df[cols].isna().sum()
    nulls = nulls[nulls > 0]
    return Check("Sin valores nulos en variables de modelado", nulls.empty,
                 "0 nulos" if nulls.empty else nulls.to_dict().__repr__())


def check_ranges(df: pd.DataFrame, ranges=PHYSICAL_RANGES) -> list[Check]:
    out = []
    for col, (lo, hi) in ranges.items():
        s = df[col].astype(float)
        bad = int(((s < lo) | (s > hi)).sum())
        out.append(Check(f"Rango físico {col} ∈ [{lo}, {hi}]", bad == 0,
                         f"min={s.min():g}, max={s.max():g}, fuera de rango={bad}"))
    return out


def check_binary(df: pd.DataFrame, cols=("Guzzler?", "Gas Guzzler Exempt", "Hybrid")) -> Check:
    bad = [c for c in cols if not set(df[c].unique()) <= {0, 1}]
    return Check("Indicadores binarios ∈ {0, 1}", not bad, f"no binarias: {bad}" if bad else f"{list(cols)}")


def check_domain(df: pd.DataFrame) -> Check:
    unknown = set(df["Drive Desc"].unique()) - DRIVE_DOMAIN
    return Check("Dominio de categorías (tracción)", not unknown,
                 f"desconocidas: {unknown}" if unknown else f"{df['Drive Desc'].nunique()} categorías conocidas")


def co2_relative_error(df: pd.DataFrame) -> pd.Series:
    """Desviación relativa entre el CO2 de la etiqueta y el implicado por el consumo.

    CO2 [g/mi] ≈ k / FE [mpg], con k = 8887 (gasolina) o 10180 (diésel).
    """
    k = np.where(df["Fuel Usage"].str.contains("Diesel"), CO2_PER_GALLON["diesel"], CO2_PER_GALLON["gasoline"])
    implied = k / df["Comb Unrd Adj FE"].astype(float)
    return (df["Comb CO2"].astype(float) / implied - 1).abs()


def check_co2_consistency(df: pd.DataFrame, median_tol=0.01, row_tol=0.05, min_share=0.99) -> Check:
    """Coherencia física CO2–consumo fila a fila.

    Detecta desalineaciones silenciosas: si una columna se reordena por separado (p. ej. al
    ordenar en una hoja de cálculo), la relación CO2·mpg ≈ constante deja de cumplirse.
    """
    err = co2_relative_error(df)
    share = float((err <= row_tol).mean())
    ok = err.median() <= median_tol and share >= min_share
    return Check("Coherencia CO₂ ≈ k / mpg (fila a fila)", bool(ok),
                 f"error mediano={err.median():.2%}, filas con error ≤ {row_tol:.0%}: {share:.1%}")


def check_combined_between_city_hwy(df: pd.DataFrame, tol=1) -> Check:
    lo = df[["City FE (Guide)", "Hwy FE (Guide)"]].min(axis=1) - tol
    hi = df[["City FE (Guide)", "Hwy FE (Guide)"]].max(axis=1) + tol
    bad = int((~df["Comb FE (Guide)"].between(lo, hi)).sum())
    return Check("Consumo combinado entre ciudad y carretera", bad == 0, f"incoherentes={bad}")


def check_guzzler_rule(df: pd.DataFrame) -> Check:
    bad = int(((df["Gas Guzzler Exempt"] == 1) & (df["Guzzler?"] == 1)).sum())
    return Check("Ningún vehículo exento paga la tasa gas guzzler", bad == 0, f"violaciones={bad}")


def validate_fe_guide(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """Ejecuta todos los contratos sobre la Fuel Economy Guide limpia."""
    schema = check_schema(df)
    if not schema.passed:
        if raise_on_error:
            raise DataValidationError(schema.detail)
        return _report([schema])
    checks = [schema, check_no_nulls(df), *check_ranges(df), check_binary(df), check_domain(df),
              check_combined_between_city_hwy(df), check_co2_consistency(df), check_guzzler_rule(df)]
    report = _report(checks)
    if raise_on_error and not report["OK"].all():
        failed = report.loc[~report["OK"], ["Comprobación", "Detalle"]].to_string(index=False)
        raise DataValidationError(f"El dataset no supera la validación:\n{failed}")
    return report


def validate_epa_test_cars(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """Contratos mínimos del extracto de ensayos EPA (tras eliminar centinelas)."""
    fe = df["RND_ADJ_FE"].dropna()
    checks = [
        Check("Sin centinelas en RND_ADJ_FE", bool((fe < FE_SENTINEL_THRESHOLD).all()),
              f"máx={fe.max():g}, ≥{FE_SENTINEL_THRESHOLD}: {int((fe >= FE_SENTINEL_THRESHOLD).sum())}"),
        Check("Consumo positivo", bool((fe > 0).all()), f"min={fe.min():g}"),
        Check("Peso de ensayo en [1500, 12000] lb",
              bool(df["Equivalent Test Weight (lbs.)"].between(1500, 12000).all()),
              f"rango=[{df['Equivalent Test Weight (lbs.)'].min()}, {df['Equivalent Test Weight (lbs.)'].max()}]"),
        Check("Coeficiente A de carretera > 0", bool((df["Target Coef A (lbf)"].dropna() > 0).all()),
              f"min={df['Target Coef A (lbf)'].min():g}"),
    ]
    report = _report(checks)
    if raise_on_error and not report["OK"].all():
        raise DataValidationError(report.loc[~report["OK"]].to_string(index=False))
    return report

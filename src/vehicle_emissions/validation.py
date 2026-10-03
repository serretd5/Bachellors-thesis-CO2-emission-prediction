"""Data validation: schema contracts, physical ranges and cross-column consistency.

Each check returns a :class:`Check`. ``validate_fe_guide`` runs them all and fails loudly
(``DataValidationError``) if any of them does not hold, so that a change in the source or a
transformation bug is caught before any model is trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FE_SENTINEL_THRESHOLD

# Grams of CO2 per gallon of fuel (EPA factors)
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
    """Raised when the dataset violates a data quality contract."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _report(checks: list[Check]) -> pd.DataFrame:
    return pd.DataFrame([c.__dict__ for c in checks]).rename(
        columns={"name": "Check", "passed": "OK", "detail": "Detail"})


def check_schema(df: pd.DataFrame, required=REQUIRED_FE_COLUMNS) -> Check:
    missing = [c for c in required if c not in df.columns]
    return Check("Schema: required columns", not missing,
                 f"missing {missing}" if missing else f"{len(required)} columns present")


def check_no_nulls(df: pd.DataFrame, cols=REQUIRED_FE_COLUMNS) -> Check:
    nulls = df[cols].isna().sum()
    nulls = nulls[nulls > 0]
    return Check("No nulls in modeling variables", nulls.empty,
                 "0 nulls" if nulls.empty else f"nulls: {nulls.to_dict()}")


def check_ranges(df: pd.DataFrame, ranges=PHYSICAL_RANGES) -> list[Check]:
    out = []
    for col, (lo, hi) in ranges.items():
        s = df[col].astype(float)
        bad = int(((s < lo) | (s > hi)).sum())
        out.append(Check(f"Physical range {col} ∈ [{lo}, {hi}]", bad == 0,
                         f"min={s.min():g}, max={s.max():g}, out of range={bad}"))
    return out


def check_binary(df: pd.DataFrame, cols=("Guzzler?", "Gas Guzzler Exempt", "Hybrid")) -> Check:
    bad = [c for c in cols if not set(df[c].unique()) <= {0, 1}]
    return Check("Binary flags ∈ {0, 1}", not bad, f"non-binary: {bad}" if bad else f"{list(cols)}")


def check_domain(df: pd.DataFrame) -> Check:
    unknown = set(df["Drive Desc"].unique()) - DRIVE_DOMAIN
    return Check("Category domain (drivetrain)", not unknown,
                 f"unknown: {unknown}" if unknown else f"{df['Drive Desc'].nunique()} known categories")


def co2_relative_error(df: pd.DataFrame) -> pd.Series:
    """Relative deviation between label CO2 and the CO2 implied by fuel economy.

    CO2 [g/mi] ≈ k / FE [mpg], with k = 8887 (gasoline) or 10180 (diesel).
    """
    k = np.where(df["Fuel Usage"].str.contains("Diesel"), CO2_PER_GALLON["diesel"], CO2_PER_GALLON["gasoline"])
    implied = k / df["Comb Unrd Adj FE"].astype(float)
    return (df["Comb CO2"].astype(float) / implied - 1).abs()


def check_co2_consistency(df: pd.DataFrame, median_tol=0.01, row_tol=0.05, min_share=0.99) -> Check:
    """Row-level physical consistency between CO2 and fuel economy.

    Detects silent misalignment: if one column is re-sorted on its own (e.g. sorting a
    spreadsheet without selecting every column), the relation CO2·mpg ≈ constant breaks.
    """
    err = co2_relative_error(df)
    share = float((err <= row_tol).mean())
    ok = err.median() <= median_tol and share >= min_share
    return Check("CO₂ ≈ k / mpg consistency (row level)", bool(ok),
                 f"median error={err.median():.2%}, rows with error ≤ {row_tol:.0%}: {share:.1%}")


def check_combined_between_city_hwy(df: pd.DataFrame, tol=1) -> Check:
    lo = df[["City FE (Guide)", "Hwy FE (Guide)"]].min(axis=1) - tol
    hi = df[["City FE (Guide)", "Hwy FE (Guide)"]].max(axis=1) + tol
    bad = int((~df["Comb FE (Guide)"].between(lo, hi)).sum())
    return Check("Combined FE between city and highway", bad == 0, f"inconsistent={bad}")


def check_guzzler_rule(df: pd.DataFrame) -> Check:
    bad = int(((df["Gas Guzzler Exempt"] == 1) & (df["Guzzler?"] == 1)).sum())
    return Check("No exempt vehicle pays the gas guzzler tax", bad == 0, f"violations={bad}")


def validate_fe_guide(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """Run every contract on the cleaned Fuel Economy Guide."""
    schema = check_schema(df)
    if not schema.passed:
        if raise_on_error:
            raise DataValidationError(schema.detail)
        return _report([schema])
    checks = [schema, check_no_nulls(df), *check_ranges(df), check_binary(df), check_domain(df),
              check_combined_between_city_hwy(df), check_co2_consistency(df), check_guzzler_rule(df)]
    report = _report(checks)
    if raise_on_error and not report["OK"].all():
        failed = report.loc[~report["OK"], ["Check", "Detail"]].to_string(index=False)
        raise DataValidationError(f"Dataset failed validation:\n{failed}")
    return report


def validate_epa_test_cars(df: pd.DataFrame, raise_on_error: bool = True) -> pd.DataFrame:
    """Minimum contracts for the EPA test extract (after removing sentinel values)."""
    fe = df["RND_ADJ_FE"].dropna()
    weight = df["Equivalent Test Weight (lbs.)"]
    checks = [
        Check("No sentinel values in RND_ADJ_FE", bool((fe < FE_SENTINEL_THRESHOLD).all()),
              f"max={fe.max():g}, ≥{FE_SENTINEL_THRESHOLD}: {int((fe >= FE_SENTINEL_THRESHOLD).sum())}"),
        Check("Positive fuel economy", bool((fe > 0).all()), f"min={fe.min():g}"),
        Check("Test weight in [1500, 12000] lb", bool(weight.between(1500, 12000).all()),
              f"range=[{weight.min()}, {weight.max()}]"),
        Check("Road-load coefficient A > 0", bool((df["Target Coef A (lbf)"].dropna() > 0).all()),
              f"min={df['Target Coef A (lbf)'].min():g}"),
    ]
    report = _report(checks)
    if raise_on_error and not report["OK"].all():
        raise DataValidationError(report.loc[~report["OK"]].to_string(index=False))
    return report

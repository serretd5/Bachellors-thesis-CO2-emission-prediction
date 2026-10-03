"""Ingestion and cleaning of certification data.

All processing starts from the Excel files published by the EPA/DOE. Every transformation is
deterministic and applied in a single pass over the full DataFrame, so each row always
corresponds to exactly one vehicle (see ``validation.check_co2_consistency``).
"""

from __future__ import annotations

import re

import pandas as pd

from .paths import EPA_TEST_CARS_PATH, FE_GUIDE_PATH

# Original column -> working name
RENAME = {
    "Mfr Name": "Mfr Name",
    "Division": "Division",
    "Carline": "Carline",
    "Eng Displ": "Eng Displ",
    "# Cyl": "# Cyl",
    "City FE (Guide) - Conventional Fuel": "City FE (Guide)",
    "Hwy FE (Guide) - Conventional Fuel": "Hwy FE (Guide)",
    "Comb FE (Guide) - Conventional Fuel": "Comb FE (Guide)",
    "Comb Unadj FE - Conventional Fuel": "Comb Unadj FE",
    "Comb Unrd Adj FE - Conventional Fuel": "Comb Unrd Adj FE",
    "Guzzler?": "Guzzler?",
    "Air Aspiration Method Desc": "Air Aspiration Method Desc",
    "Trans Desc": "Trans Desc",
    "# Gears": "# Gears",
    "Drive Desc": "Drive Desc",
    "Fuel Usage Desc - Conventional Fuel": "Fuel Usage",
    "Gas Guzzler Exempt (Where Truck = 1975 NHTSA truck definition)": "Gas Guzzler Exempt",
    "Descriptor - Model Type (40 Char or less)": "Descriptor",
    "Carline Class Desc": "Carline Class Desc",
    "Cyl Deact?": "Cyl Deact?",
    "Var Valve Timing?": "Var Valve Timing?",
    "Var Valve Lift?": "Var Valve Lift?",
    "Stop/Start System (Engine Management System) Code": "Stop/Start",
    "FE Rating (1-10 rating on Label)": "FE Rating",
    "GHG Rating (1-10 rating on Label)": "GHG Rating",
    "Comb CO2 Rounded Adjusted (as shown on FE Label)": "Comb CO2",
}

# EPA classes grouped as "large vehicle" (1) vs "compact" (0)
LARGE_CLASSES = {
    "Midsize Cars", "Large Cars", "Midsize Station Wagons",
    "Standard SUV 2WD", "Standard SUV 4WD",
    "Small Pick-up Trucks 2WD", "Small Pick-up Trucks 4WD",
    "Standard Pick-up Trucks 2WD", "Standard Pick-up Trucks 4WD",
    "Special Purpose Vehicle 2WD", "Special Purpose Vehicle 4WD",
    "Special Purpose Vehicle cab chassis",
    "Special Purpose Vehicle, minivan 2WD", "Special Purpose Vehicle, minivan 4WD",
}

_DRIVE_TOKENS = re.compile(r"\b(2WD|4WD|AWD|FWD|RWD|4MATIC\+?|XDRIVE|QUATTRO|SH-AWD)\b", re.IGNORECASE)


def load_fe_guide_raw(path=FE_GUIDE_PATH) -> pd.DataFrame:
    """Read sheet 24MY exactly as published."""
    df = pd.read_excel(path, sheet_name="24MY")
    df.columns = [c.strip() for c in df.columns]
    return df


def model_family(df: pd.DataFrame) -> pd.Series:
    """Model family identifier (division + carline without the drivetrain suffix).

    Used as the group in cross-validation: 2WD/AWD variants of the same model share engine and
    body, and splitting them between training and test would inflate the metrics.
    """
    base = df["Carline"].astype(str).str.upper().str.replace(_DRIVE_TOKENS, "", regex=True)
    base = base.str.replace(r"\s+", " ", regex=True).str.strip()
    return df["Division"].astype(str).str.upper().str.strip() + " | " + base


def clean_fe_guide(raw: pd.DataFrame) -> pd.DataFrame:
    """Select, rename and type the modeling variables.

    * Drop fully empty rows.
    * ``Guzzler?``: 'G' -> 1, empty -> 0.
    * ``Gas Guzzler Exempt``: 'T' (truck under the 1975 NHTSA definition, exempt) -> 1, 'N' -> 0.
    * ``Hybrid`` / ``Mild Hybrid``: extracted from the model type descriptor.
    * Y/N flags (cylinder deactivation, variable valve timing/lift, stop/start) -> 0/1.
    * ``Model Family``: group for leakage-free cross-validation across variants.
    """
    df = raw[list(RENAME)].rename(columns=RENAME).copy()
    df = df.dropna(how="all")

    df["Guzzler?"] = (df["Guzzler?"].astype(str).str.strip() == "G").astype(int)
    df["Gas Guzzler Exempt"] = (df["Gas Guzzler Exempt"].astype(str).str.strip() == "T").astype(int)

    desc = df["Descriptor"].fillna("")
    df["Hybrid"] = (desc.str.contains("Hybrid") & ~desc.str.contains("Mild Hybrid")).astype(int)
    df["Mild Hybrid"] = desc.str.contains("Mild Hybrid").astype(int)

    for col in ["Cyl Deact?", "Var Valve Timing?", "Var Valve Lift?", "Stop/Start"]:
        df[col] = (df[col].astype(str).str.upper().str.strip() == "Y").astype(int)

    df["Eng Displ"] = df["Eng Displ"].astype(float)
    for col in ["# Cyl", "# Gears", "Comb FE (Guide)", "City FE (Guide)", "Hwy FE (Guide)",
                "FE Rating", "GHG Rating", "Comb CO2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df["Model Family"] = model_family(df)
    return df.drop(columns=["Descriptor"]).reset_index(drop=True)


def load_fe_guide(path=FE_GUIDE_PATH) -> pd.DataFrame:
    """Load + clean the Fuel Economy Guide."""
    return clean_fe_guide(load_fe_guide_raw(path))


def add_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compact binary encoding of the categorical variables.

    ===================  =========================================
    Variable             1 means
    ===================  =========================================
    ``Automatic``        non-manual transmission
    ``AWD/4WD``          all-wheel or four-wheel drive (incl. part-time 4WD)
    ``Gasoline``         gasoline (0 = diesel)
    ``Large``            midsize/large class, pickup or special purpose vehicle
    ===================  =========================================
    """
    out = df.copy()
    out["Automatic"] = (out["Trans Desc"] != "Manual").astype(int)
    out["AWD/4WD"] = (~out["Drive Desc"].str.startswith("2-Wheel")).astype(int)
    out["Gasoline"] = (~out["Fuel Usage"].str.contains("Diesel")).astype(int)
    out["Large"] = out["Carline Class Desc"].isin(LARGE_CLASSES).astype(int)
    return out


# ---------------------------------------------------------------------------
# EPA Test Car List
# ---------------------------------------------------------------------------

#: Above this value ``RND_ADJ_FE`` is not a physical fuel economy (markers such as 999, 10000).
FE_SENTINEL_THRESHOLD = 999


def load_epa_test_cars(path=EPA_TEST_CARS_PATH) -> pd.DataFrame:
    """Read the curated extract of the EPA Test Car List."""
    return pd.read_excel(path)


def vehicle_signature(df: pd.DataFrame) -> pd.Series:
    """Tested-vehicle identifier (same test weight, power, displacement and road-load coefficients).

    Each vehicle appears in several cycles (FTP, HWY, US06...). Grouping by this signature keeps
    the same vehicle from appearing in both training and test.
    """
    cols = ["Equivalent Test Weight (lbs.)", "Rated Horsepower", "Test Veh Displacement (L)",
            "Target Coef A (lbf)", "Target Coef B (lbf/mph)", "Target Coef C (lbf/mph**2)"]
    return df[cols].round(4).astype(str).agg("|".join, axis=1)

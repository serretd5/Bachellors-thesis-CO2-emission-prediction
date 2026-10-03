"""Ingesta y limpieza de los datos de homologación.

Todo el tratamiento parte del Excel publicado por la EPA/DOE. Cada transformación es
determinista y se aplica en un único paso sobre el DataFrame completo, de modo que
cada fila corresponde siempre a un único vehículo (ver ``validation.check_co2_consistency``).
"""

from __future__ import annotations

import re

import pandas as pd

from .paths import EPA_TEST_CARS_PATH, FE_GUIDE_PATH

# Columna original -> nombre de trabajo
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

# Clases EPA agrupadas como "vehículo grande" (1) frente a "compacto" (0)
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
    """Lee la hoja 24MY tal cual viene en el Excel original."""
    df = pd.read_excel(path, sheet_name="24MY")
    df.columns = [c.strip() for c in df.columns]
    return df


def model_family(df: pd.DataFrame) -> pd.Series:
    """Identificador de familia de modelo (división + denominación sin la variante de tracción).

    Se usa como grupo en la validación cruzada: las variantes 2WD/AWD de un mismo modelo
    comparten motor y carrocería, y repartirlas entre entrenamiento y prueba inflaría las métricas.
    """
    base = df["Carline"].astype(str).str.upper().str.replace(_DRIVE_TOKENS, "", regex=True)
    base = base.str.replace(r"\s+", " ", regex=True).str.strip()
    return df["Division"].astype(str).str.upper().str.strip() + " | " + base


def clean_fe_guide(raw: pd.DataFrame) -> pd.DataFrame:
    """Selecciona, renombra y tipa las variables de modelado.

    * Elimina filas totalmente vacías.
    * ``Guzzler?``: 'G' -> 1, vacío -> 0.
    * ``Gas Guzzler Exempt``: 'T' (camión según NHTSA 1975, exento) -> 1, 'N' -> 0.
    * ``Hybrid`` / ``Mild Hybrid``: extraídos del descriptor del modelo.
    * Indicadores Y/N (desactivación de cilindros, distribución variable, stop/start) -> 0/1.
    * ``Model Family``: grupo para validación cruzada sin fuga entre variantes.
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
    """Carga + limpieza de la Fuel Economy Guide."""
    return clean_fe_guide(load_fe_guide_raw(path))


def add_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codificación binaria compacta de las variables categóricas.

    ===================  =========================================
    Variable             1 significa
    ===================  =========================================
    ``Automatic``        transmisión no manual
    ``AWD/4WD``          tracción total o 4x4 (incluye 4WD parcial)
    ``Gasoline``         gasolina (0 = diésel)
    ``Large``            clase media/grande, pick-up o vehículo especial
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

#: Por encima de este valor ``RND_ADJ_FE`` no es un consumo físico (marcadores 999, 10000...).
FE_SENTINEL_THRESHOLD = 999


def load_epa_test_cars(path=EPA_TEST_CARS_PATH) -> pd.DataFrame:
    """Lee el extracto curado del EPA Test Car List."""
    return pd.read_excel(path)


def vehicle_signature(df: pd.DataFrame) -> pd.Series:
    """Identificador de vehículo ensayado (mismo peso, potencia, cilindrada y coeficientes de carretera).

    Cada vehículo aparece en varios ciclos (FTP, HWY, US06...). Agrupar por esta firma evita que el
    mismo vehículo esté a la vez en entrenamiento y prueba.
    """
    cols = ["Equivalent Test Weight (lbs.)", "Rated Horsepower", "Test Veh Displacement (L)",
            "Target Coef A (lbf)", "Target Coef B (lbf/mph)", "Target Coef C (lbf/mph**2)"]
    return df[cols].round(4).astype(str).agg("|".join, axis=1)

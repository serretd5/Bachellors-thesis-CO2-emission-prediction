"""Rutas del proyecto, independientes del directorio de ejecución."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

# Fuente primaria: Fuel Economy Guide 2024 (EPA / DOE), hoja 24MY.
FE_GUIDE_PATH = RAW_DIR / "2024_FE_Guide_DOE.xlsx"

# Fuente secundaria: extracto curado del EPA Test Car List 2024 (ensayos de laboratorio).
EPA_TEST_CARS_PATH = RAW_DIR / "epa_test_car_list_2024_curated.xlsx"

for _d in (PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

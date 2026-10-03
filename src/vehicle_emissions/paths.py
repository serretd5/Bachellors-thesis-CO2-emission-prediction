"""Project paths, independent of the working directory."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

# Primary source: Fuel Economy Guide 2024 (EPA / DOE), sheet 24MY.
FE_GUIDE_PATH = RAW_DIR / "2024_FE_Guide_DOE.xlsx"

# Secondary source: curated extract of the EPA Test Car List 2024 (laboratory tests).
EPA_TEST_CARS_PATH = RAW_DIR / "epa_test_car_list_2024_curated.xlsx"

for _d in (PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

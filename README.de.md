# Modellierung von Kraftstoffverbrauch und CO₂-Emissionen von Fahrzeugen

[English](README.md) · **Deutsch**

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

Machine-Learning-Modelle, die den **zertifizierten Kraftstoffverbrauch und die CO₂-Emissionen** von Pkw aus ihrer
technischen Spezifikation schätzen. Grundlage sind offizielle Zertifizierungsdaten der US-Umweltbehörde EPA, verarbeitet
in einer reproduzierbaren Pipeline mit Datenqualitätsverträgen und einem Evaluationsprotokoll, das die Modellgüte
nicht überschätzt.

## Zentrale Ergebnisse

| Fragestellung | Bestes Modell | Metrik auf ungesehenen Daten* | Referenz |
|---|---|---|---|
| Kombinierter Verbrauch aus der Spezifikation | Gradient Boosting | **R² 0,91 · MAE 1,4 mpg** | MAE 4,9 mpg (Mittelwert) |
| Kombiniertes CO₂ aus der Spezifikation | Gradient Boosting | **R² 0,89 · MAE 23 g/mi** | — |
| Verbrauch je Zyklus aus Prüfstandsgrößen | Gradient Boosting | **R² 0,97 · MAE 1,6 mpg** | MAE 9,5 mpg (Mittelwert) |
| CO₂ je Zyklus aus Prüfstandsgrößen | Gradient Boosting | **R² 0,94 · MAE 18 g/mi** | — |
| Betroffenheit von der *Gas Guzzler Tax* | Neuronales Netz (MLP) | **F1 0,80 · PR-AUC 0,88** | F1 0,68 (Expertenregel) |

\* Gruppierte Kreuzvalidierung: nach Modellfamilie (Fuel Economy Guide) bzw. nach geprüftem Fahrzeug (Test Car List),
sodass keine Variante desselben Fahrzeugs gleichzeitig in Training und Validierung vorkommt.

**Wesentliche Erkenntnisse**

* Der **Hubraum** ist mit Abstand der stärkste Einflussfaktor auf den Verbrauch, gefolgt von Aufladung, Hybridisierung
  und Fahrzeugklasse. Ohne das Hybrid-Merkmal sinkt R² von 0,91 auf 0,77.
* Mit den **physikalischen Prüfstandsgrößen** (äquivalente Prüfmasse, Fahrwiderstandskoeffizienten A/B/C, Leistung,
  Prüfzyklus) lässt sich der Verbrauch eines Verbrenners nahezu vollständig erklären (R² 0,97).
* **One-Hot-Kodierung des Prüfzyklus statt ordinaler Kodierung** hebt das R² des linearen Modells von 0,55 auf 0,83.
* Bei **Elektrofahrzeugen** liefert eine zufällige Kreuzvalidierung R² 0,82, gruppiert nach Fahrzeug jedoch nur 0,28:
  Das Modell hatte wiederholt vorkommende Fahrzeuge auswendig gelernt. Genau diese Art von Datenleck soll das Projekt aufdecken.

## Engineering und Validierung

| Praxis | Umsetzung |
|---|---|
| **Eine einzige, nachvollziehbare Quelle** | Alles wird über deterministische Funktionen aus den von der EPA veröffentlichten Excel-Dateien abgeleitet (`src/vehicle_emissions/data.py`) |
| **Datenverträge** | `validation.py`: Schema, fehlende Werte, physikalische Wertebereiche, Kategoriedomänen, Konsistenz Stadt/Autobahn, Befreiungsregel sowie eine **zeilenweise physikalische Plausibilitätsprüfung CO₂ ≈ k / mpg** |
| **Erkennung stiller Datenfehler** | Tests injizieren eine verschobene Spalte, Werte außerhalb des Bereichs, fehlende Werte und fehlende Spalten und prüfen, dass jeder Fehler erkannt wird |
| **Kein Informationsleck** | Aus der Zielgröße abgeleitete Merkmale ausgeschlossen; Vorverarbeitung innerhalb der `Pipeline`; Splits gruppiert nach Modellfamilie bzw. Prüffahrzeug |
| **Verbindliche Referenzmodelle** | Jedes Modell wird mit `DummyRegressor`/`DummyClassifier` und bei der Klassifikation mit einer Expertenregel verglichen |
| **Saubere Modellauswahl** | Hyperparameter ausschließlich auf Trainingsdaten optimiert (`GroupKFold` / `StratifiedGroupKFold`) |
| **Robustheit** | Ablationsstudie, Sensitivität gegenüber Ausreißern, wiederholte Kreuzvalidierung mit mehreren Seeds, Residuen- und Segmentanalyse |
| **Interpretierbarkeit** | Permutation Importance auf den Originalmerkmalen; nachvollziehbarer Entscheidungsbaum |
| **Regressionstests der Modellgüte** | Mindestwerte für R² in `tests/test_models.py`, um schleichende Verschlechterungen zu erkennen |
| **CI** | GitHub Actions: `ruff`, `pytest` unter Python 3.11 und 3.12 sowie vollständige Ausführung der Notebooks |

## Projektstruktur

```
├── .github/workflows/ci.yml     # Linting, Tests und Reproduzierbarkeit der Notebooks
├── data/
│   ├── raw/                     # EPA-Quellen (unverändert)
│   └── processed/               # Validierter Datensatz (erzeugt von Notebook 01)
├── docs/                        # Model Card (EN / DE)
├── notebooks/
│   ├── 01_ingestion_validation_eda.ipynb     # Datenaufnahme, Validierung und EDA
│   ├── 02_fuel_economy_co2_modeling.ipynb    # Modellierung von Verbrauch und CO₂
│   ├── 03_gas_guzzler_classification.ipynb   # Klassifikation Gas Guzzler
│   └── 04_laboratory_test_models.ipynb       # Modelle auf Prüfstandsdaten
├── reports/figures/             # Erzeugte Abbildungen
├── src/vehicle_emissions/
│   ├── data.py                  # Datenaufnahme, Bereinigung, Kodierung, CV-Gruppen
│   ├── validation.py            # Datenqualitätsverträge
│   ├── evaluation.py            # Evaluationsprotokoll und Grafiken
│   └── paths.py
├── tests/                       # Daten, Verträge und Gütemindestwerte
├── Makefile
└── pyproject.toml
```

## Notebooks

1. **Datenaufnahme, Validierung und EDA** — Laden des offiziellen Datensatzes, Ausführung der Qualitätsverträge
   (inklusive Nachweis, dass eine verschobene Spalte erkannt wird), explorative Analyse mit Blick auf
   Modellierungsentscheidungen und Kontrolle von Datenlecks.
2. **Modellierung von Verbrauch und CO₂** — Referenzmodell, lineare, polynomiale und Gradient-Boosting-Modelle;
   gruppierte Hyperparametersuche; Ablation; Ausreißersensitivität; Residuendiagnose und Permutation Importance.
3. **Klassifikation Gas Guzzler** — Unausgewogenes Problem (5,7 % Positive): Expertenregel, logistische Regression,
   Entscheidungsbaum und MLP mit stratifizierten, gruppierten Splits, Precision-Recall-Kurven und wiederholter Kreuzvalidierung.
4. **Modelle auf Prüfstandsdaten** — Physikalisch motivierte Modelle auf der EPA Test Car List: Entfernen nicht-physikalischer
   Markierungswerte, Kodierung des Prüfzyklus, Verbrauch und CO₂ je Zyklus sowie Erweiterung auf Elektrofahrzeuge.

Die Notebooks sind auf Englisch dokumentiert.

## Daten

| Quelle | Inhalt |
|---|---|
| `2024_FE_Guide_DOE.xlsx` | Fuel Economy Guide 2024 (EPA / DOE), Blatt `24MY`: 964 Verbrenner-Konfigurationen, Modelljahr 2024 |
| `epa_test_car_list_2024_curated.xlsx` | Kuratierter Auszug der EPA Test Car List 2024 (28 Spalten): 4.137 Prüfungen mit Prüfmasse, Fahrwiderstandskoeffizienten, Zyklus und Emissionen |

## Reproduzieren

```bash
python -m venv .venv && source .venv/bin/activate
make install     # pip install -e ".[dev]"
make test        # Datenverträge + Tests
make notebooks   # führt alle vier Notebooks vollständig aus
```

## Einschränkungen

* Ein einziges Modelljahr (2024) und US-Zertifizierungsdaten; die Ergebnisse sind nicht direkt mit europäischen WLTP-Werten vergleichbar.
* Der Fehler bei Vollhybriden ist fast viermal so hoch wie bei konventionellen Fahrzeugen: Merkmale zu Batterie und
  E-Antrieb fehlen.
* Nur 20 Dieselfahrzeuge und 55 Gas-Guzzler-Fälle: Die Metriken für diese Segmente weisen eine hohe Varianz auf.

Weitere Details in der [Model Card](docs/model_card.de.md).

## Lizenz

Apache 2.0 — siehe [`LICENSE`](LICENSE).

# Model Card · Prognose von Verbrauch und CO₂ (Fuel Economy Guide)

[English](model_card.md) · **Deutsch**

## Überblick

| | |
|---|---|
| **Aufgabe** | Regression des zertifizierten kombinierten Kraftstoffverbrauchs (mpg) und der CO₂-Emissionen (g/mi) |
| **Modell** | `GradientBoostingRegressor` (scikit-learn) mit One-Hot-kodierten kategorialen Merkmalen innerhalb einer `Pipeline` |
| **Hyperparameter** | `n_estimators=400`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8` (ausgewählt mit `GroupKFold` auf den Trainingsdaten) |
| **Daten** | Fuel Economy Guide 2024 (EPA / DOE), 964 Verbrenner-Konfigurationen, 596 Modellfamilien |
| **Notebook** | `notebooks/02_fuel_economy_co2_modeling.ipynb` |

## Vorgesehener Einsatz

Frühzeitige Schätzung des zertifizierten Verbrauchs und der CO₂-Emissionen einer Verbrenner-Konfiguration aus ihrer
technischen Spezifikation (Motor, Getriebe, Antrieb, Klasse, Kraftstoff, Hybridisierung), bevor Prüfergebnisse
vorliegen. Unterstützt die Definition des Modellprogramms, den Vergleich von Antriebsalternativen und das Screening von
Konfigurationen, die von der *Gas Guzzler Tax* betroffen sind.

**Nicht vorgesehen für:** batterieelektrische Fahrzeuge oder Plug-in-Hybride, Zertifizierung oder offizielle Angabe von
Verbrauchswerten, Nutzfahrzeuge sowie andere Zyklen als den kombinierten EPA-Zyklus (z. B. WLTP) ohne erneutes Training.

## Eingangsmerkmale

| Merkmal | Typ |
|---|---|
| Hubraum (l), Zylinderzahl, Anzahl der Gänge | numerisch |
| Vollhybrid | binär |
| Kraftstoff, Antrieb, EPA-Fahrzeugklasse, Aufladung, Zylinderabschaltung | kategorial (One-Hot) |

Merkmale, die aus dem Verbrauch selbst abgeleitet sind (Stadt, Autobahn, unkorrigiert, Label-Bewertungen, CO₂,
Gas-Guzzler-Kennzeichen), sind **ausgeschlossen**, um Informationslecks zu vermeiden.

## Modellgüte

Bewertet auf **ungesehenen Modellfamilien** (`GroupKFold`, 5 Folds), sodass Varianten desselben Modells nie auf
Training und Validierung aufgeteilt werden.

| Zielgröße | R² | MAE | Referenz (Mittelwert) MAE |
|---|---|---|---|
| Kombinierter Verbrauch | 0,91 | 1,41 mpg | 4,95 mpg |
| Kombiniertes CO₂ | 0,89 | 23,2 g/mi | — |

Güte nach Segment (Test-Split, Verbrauch):

| Segment | MAE |
|---|---|
| Konventionell | 1,3 mpg |
| Vollhybride | 4,9 mpg |
| Diesel | 0,8 mpg (n = 6) |

## Datenvalidierung

Die Pipeline trainiert nur, wenn der Datensatz die Verträge in `vehicle_emissions.validation` erfüllt: Schema, keine
fehlenden Werte, physikalische Wertebereiche, Kategoriedomänen, Konsistenz von Stadt-/Autobahn-/Kombiwert,
Befreiungsregel sowie eine **zeilenweise physikalische Plausibilitätsprüfung CO₂ ≈ k / mpg**, die verschobene Spalten
erkennt. Die Verträge laufen in der CI zusammen mit Tests, die gezielt Fehler injizieren und deren Erkennung prüfen.

## Einschränkungen und Risiken

* **Hybride:** Der Fehler ist fast viermal so hoch wie bei konventionellen Fahrzeugen. Die Guide enthält weder
  Batteriekapazität noch elektrische Leistung, die den Verbrauch von Hybriden bestimmen.
* **Diesel:** Nur 20 Fahrzeuge im Datensatz; Schätzungen für diesen Kraftstoff sind mit hoher Unsicherheit behaftet.
* **Zeitliche Abdeckung:** Ein einziges Modelljahr (2024). Technik und Regulierung ändern sich; das Modell sollte mit
  jeder neuen Ausgabe der Guide neu trainiert und auf Drift überwacht werden.
* **Markt:** US-Zertifizierungsdaten (FTP/HWY-Zyklen mit EPA-Korrektur), nicht direkt mit europäischen WLTP-Werten vergleichbar.
* **Fehlanwendung:** Das Modell ersetzt keine Homologationsprüfung und darf nicht zur Angabe offizieller Verbrauchswerte
  verwendet werden.

## Reproduzierbarkeit

```bash
pip install -e ".[dev]"
make all   # Linting + Tests + vollständige Ausführung der Notebooks
```

Feste Seeds (`random_state=42`); die CI führt bei jeder Änderung alle vier Notebooks vollständig aus.

# Modelado del consumo y las emisiones de CO₂ de vehículos

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

Modelos de machine learning que estiman el **consumo homologado y las emisiones de CO₂** de turismos a partir de su
especificación técnica, construidos sobre datos oficiales de la EPA (EE. UU.) con un pipeline reproducible,
contratos de calidad de datos y una evaluación diseñada para no sobrestimar el rendimiento.

## Resultados clave

| Problema | Mejor modelo | Métrica sobre datos no vistos* | Referencia |
|---|---|---|---|
| Consumo combinado desde la especificación | Gradient Boosting | **R² 0,91 · MAE 1,4 mpg** | MAE 4,9 mpg (media) |
| CO₂ combinado desde la especificación | Gradient Boosting | **R² 0,89 · MAE 23 g/mi** | — |
| Consumo por ciclo desde magnitudes de ensayo | Gradient Boosting | **R² 0,97 · MAE 1,6 mpg** | MAE 9,5 mpg (media) |
| CO₂ por ciclo desde magnitudes de ensayo | Gradient Boosting | **R² 0,94 · MAE 18 g/mi** | — |
| Exposición a la *Gas Guzzler Tax* | Red neuronal (MLP) | **F1 0,80 · PR-AUC 0,88** | F1 0,68 (regla experta) |

\* Validación cruzada agrupada: por familia de modelo (Fuel Economy Guide) o por vehículo ensayado (Test Car List),
de forma que ninguna variante del mismo vehículo está a la vez en entrenamiento y validación.

**Hallazgos principales**

* La **cilindrada** es el determinante dominante del consumo; le siguen sobrealimentación, hibridación y clase de vehículo.
  Retirar el indicador de híbrido baja el R² de 0,91 a 0,77.
* Con las **magnitudes físicas del ensayo** (masa equivalente, coeficientes de resistencia al avance A/B/C, potencia,
  ciclo) el consumo de un vehículo de combustión se explica casi por completo (R² 0,97).
* **Codificar el ciclo de ensayo con one-hot en lugar de ordinal** sube el R² lineal de 0,55 a 0,83.
* En **eléctricos**, la validación cruzada aleatoria da R² 0,82 y la agrupada por vehículo 0,28: el modelo memorizaba
  vehículos repetidos. Es el tipo de fuga que este proyecto está diseñado para detectar.

## Ingeniería y validación

| Práctica | Implementación |
|---|---|
| **Fuente única y trazable** | Todo se deriva del Excel publicado por la EPA mediante funciones deterministas (`src/vehicle_emissions/data.py`) |
| **Contratos de datos** | `validation.py`: esquema, nulos, rangos físicos, dominios, coherencia ciudad/carretera, regla de exención y **coherencia física CO₂ ≈ k / mpg fila a fila** |
| **Detección de corrupción silenciosa** | Los tests inyectan una columna desalineada, valores fuera de rango, nulos y columnas ausentes, y verifican que cada defecto se detecta |
| **Sin fuga de información** | Variables derivadas del objetivo excluidas; preprocesado dentro de `Pipeline`; particiones agrupadas por familia o vehículo |
| **Referencias obligatorias** | Cada modelo se compara con `DummyRegressor`/`DummyClassifier` y, en clasificación, con una regla experta |
| **Selección honesta** | Hiperparámetros elegidos sólo con datos de entrenamiento (`GroupKFold` / `StratifiedGroupKFold`) |
| **Robustez** | Ablación de variables, sensibilidad a extremos, CV repetida con varias semillas, análisis de residuos y de error por segmento |
| **Interpretabilidad** | Importancia por permutación sobre variables originales; árbol de decisión auditable |
| **Tests de regresión del rendimiento** | Umbrales mínimos de R² en `tests/test_models.py` para detectar degradaciones |
| **CI** | GitHub Actions: `ruff`, `pytest` en Python 3.11 y 3.12, y ejecución completa de los notebooks |

## Estructura

```
├── .github/workflows/ci.yml     # Lint, tests y reproducibilidad de notebooks
├── data/
│   ├── raw/                     # Fuentes EPA (sin modificar)
│   └── processed/               # Dataset validado (generado por el notebook 01)
├── docs/model_card.md           # Ficha del modelo: uso previsto, métricas, limitaciones
├── notebooks/
│   ├── 01_ingesta_validacion_eda.ipynb
│   ├── 02_modelado_consumo_co2.ipynb
│   ├── 03_clasificacion_gas_guzzler.ipynb
│   └── 04_ensayos_laboratorio_epa.ipynb
├── reports/figures/             # Figuras generadas
├── src/vehicle_emissions/
│   ├── data.py                  # Ingesta, limpieza, codificación, grupos para CV
│   ├── validation.py            # Contratos de calidad de datos
│   ├── evaluation.py            # Protocolo de evaluación y gráficos
│   └── paths.py
├── tests/                       # Datos, contratos y umbrales de rendimiento
├── Makefile
└── pyproject.toml
```

## Notebooks

1. **Ingesta, validación y EDA** — Carga del dataset oficial, contratos de calidad (con demostración de detección de
   una columna desalineada), análisis exploratorio orientado a decisiones de modelado y control de fuga.
2. **Modelado de consumo y CO₂** — Referencia, modelos lineales, polinómicos y Gradient Boosting; ajuste de
   hiperparámetros agrupado; ablación; sensibilidad a extremos; diagnóstico de residuos e importancia por permutación.
3. **Clasificación *Gas Guzzler*** — Problema desequilibrado (5,7 % positivos): regla experta, regresión logística,
   árbol de decisión y MLP con particiones estratificadas y agrupadas, curvas precisión-recall y CV repetida.
4. **Ensayos de laboratorio** — Modelos físicos sobre el EPA Test Car List: limpieza de marcadores no físicos,
   codificación del ciclo, consumo y CO₂ por ciclo, y extensión a vehículos eléctricos.

## Datos

| Fuente | Contenido |
|---|---|
| `2024_FE_Guide_DOE.xlsx` | Fuel Economy Guide 2024 (EPA / DOE), hoja `24MY`: 964 configuraciones de combustión del año modelo 2024 |
| `epa_test_car_list_2024_curated.xlsx` | Extracto curado del EPA Test Car List 2024 (28 columnas): 4137 ensayos con masa, coeficientes de carretera, ciclo y emisiones |

## Reproducir

```bash
python -m venv .venv && source .venv/bin/activate
make install     # pip install -e ".[dev]"
make test        # contratos de datos + tests
make notebooks   # ejecuta los cuatro notebooks de principio a fin
```

## Limitaciones

* Un único año modelo (2024) y datos de homologación de EE. UU.; no directamente comparables con WLTP.
* El error en híbridos es casi cuatro veces el de los convencionales: faltan variables de batería y tren eléctrico.
* Sólo 20 vehículos diésel y 55 positivos de *Gas Guzzler*: las métricas de estos segmentos tienen alta varianza.

Más detalle en la [ficha del modelo](docs/model_card.md).

## Licencia

Apache 2.0 — ver [`LICENSE`](LICENSE).

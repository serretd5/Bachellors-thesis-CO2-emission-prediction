# Ficha del modelo · Predicción de consumo y CO₂ (Fuel Economy Guide)

## Resumen

| | |
|---|---|
| **Tarea** | Regresión del consumo combinado homologado (mpg) y de las emisiones de CO₂ (g/mi) |
| **Modelo** | `GradientBoostingRegressor` (scikit-learn) con one-hot de variables categóricas, dentro de un `Pipeline` |
| **Hiperparámetros** | `n_estimators=400`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8` (seleccionados con `GroupKFold` sobre entrenamiento) |
| **Datos** | Fuel Economy Guide 2024 (EPA / DOE), 964 configuraciones de vehículos de combustión, 596 familias de modelo |
| **Notebook** | `notebooks/02_modelado_consumo_co2.ipynb` |

## Uso previsto

Estimación temprana del consumo y del CO₂ homologados de una configuración de vehículo de combustión a partir de su
especificación técnica (motor, transmisión, tracción, clase, combustible, hibridación), antes de disponer de
resultados de ensayo. Apoyo a la definición de gama, comparación de alternativas de tren motor y cribado de
configuraciones sujetas a la *Gas Guzzler Tax*.

**Fuera de alcance:** vehículos eléctricos o híbridos enchufables, certificación o declaración oficial de consumos,
vehículos pesados, ciclos distintos del combinado EPA (p. ej. WLTP) sin reentrenar.

## Variables de entrada

| Variable | Tipo |
|---|---|
| Cilindrada (L), nº de cilindros, nº de marchas | numérica |
| Híbrido completo | binaria |
| Combustible, tracción, clase EPA, sobrealimentación, desactivación de cilindros | categórica (one-hot) |

Las variables derivadas del propio consumo (ciudad, carretera, no ajustado, notas de etiqueta, CO₂, gas guzzler) están
**excluidas** para evitar fuga de información.

## Rendimiento

Evaluado sobre **familias de modelo no vistas** (`GroupKFold`, 5 particiones), para que las variantes de un mismo modelo
no se repartan entre entrenamiento y validación.

| Objetivo | R² | MAE | Referencia (media) MAE |
|---|---|---|---|
| Consumo combinado | 0,91 | 1,41 mpg | 4,95 mpg |
| CO₂ combinado | 0,89 | 23,2 g/mi | — |

Rendimiento por segmento (partición de prueba, consumo):

| Segmento | MAE |
|---|---|
| Convencionales | 1,3 mpg |
| Híbridos completos | 4,9 mpg |
| Diésel | 0,8 mpg (n = 6) |

## Validación de datos

El pipeline no entrena si el dataset no supera los contratos de `vehicle_emissions.validation`: esquema, ausencia de
nulos, rangos físicos, dominios de categorías, coherencia ciudad/carretera/combinado, regla de exención y
**coherencia física CO₂ ≈ k / mpg fila a fila**, que detecta columnas desalineadas. Los contratos se ejecutan en CI
junto con tests que inyectan defectos y verifican que se detectan.

## Limitaciones y riesgos

* **Híbridos:** error casi cuatro veces mayor que en convencionales. La guía no incluye capacidad de batería ni potencia
  eléctrica, que son los determinantes de su consumo.
* **Diésel:** sólo 20 vehículos en el conjunto de datos; las estimaciones para este combustible tienen alta incertidumbre.
* **Cobertura temporal:** un único año modelo (2024). La tecnología y la normativa cambian; se debe reentrenar y vigilar
  la deriva con cada nueva edición de la guía.
* **Mercado:** datos de homologación de EE. UU. (ciclos FTP/HWY con ajuste EPA). No son directamente comparables con
  valores WLTP europeos.
* **Uso indebido:** el modelo no sustituye un ensayo de homologación ni debe usarse para declarar consumos oficiales.

## Reproducibilidad

```bash
pip install -e ".[dev]"
make all   # lint + tests + ejecución completa de los notebooks
```

Semillas fijadas (`random_state=42`); la CI ejecuta los cuatro notebooks de principio a fin en cada cambio.

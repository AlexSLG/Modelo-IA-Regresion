# California Housing — Modelos de Regresión

**Caso 1** de la plataforma *Modelos de Inteligencia Artificial*.

Predicción del valor medio de una vivienda mediante **Regresión Lineal
Múltiple** y **Regresión Polinomial** a partir de características
geográficas, demográficas y económicas de los distritos censales de
California.

---

## 1. Descripción del proyecto

El problema consiste en estimar el **valor medio de una vivienda** en cada
distrito censal de California a partir de características geográficas,
demográficas y económicas del distrito. Es un problema de **regresión
supervisada**: a partir de los descriptores, el modelo aprende a estimar una
variable numérica continua.

## 2. Objetivo

Predecir el valor medio de una vivienda (`median_house_value`) y comparar el
desempeño de dos familias de modelos:

- **Regresión Lineal Múltiple** (modelo base).
- **Regresión Polinomial** con grados 2 y 3 (relaciones no lineales).

## 3. Dataset

- **Nombre:** `data/housing(1).csv` (dataset *California Housing*).
- **Registros:** 20,640 distritos censales.
- **Variables:** 10 (9 predictoras + 1 objetivo).
- **Variable objetivo:** `median_house_value` (USD).

Contiene valores **censurados** en el tope de `500,001` (965 registros,
4.68%) y 207 valores nulos en `total_bedrooms`.

## 4. Descriptores

| Descriptor | Descripción | Tipo | Uso |
| ---------- | ----------- | ---- | --- |
| `longitude` | Longitud geográfica del distrito | Numérica continua | Ubicación geográfica |
| `latitude` | Latitud geográfica del distrito | Numérica continua | Ubicación geográfica |
| `housing_median_age` | Edad mediana de las viviendas (años) | Numérica entera | Antigüedad del parque inmobiliario |
| `total_rooms` | Total de habitaciones del distrito | Numérica entera | Tamaño del parque inmobiliario |
| `total_bedrooms` | Total de dormitorios del distrito | Numérica entera | Tamaño del parque inmobiliario (contiene nulos) |
| `population` | Población total del distrito | Numérica entera | Tamaño demográfico |
| `households` | Número de hogares del distrito | Numérica entera | Viviendas ocupadas |
| `median_income` | Ingreso medio de los habitantes | Numérica continua | Poder adquisitivo (mayor correlación) |
| `ocean_proximity` | Proximidad al océano (5 categorías) | Categórica | Zona geográfica (One-Hot Encoding) |

**Ninguna variable fue descartada.** Las correlaciones altas entre los
conteos se documentaron mediante VIF, pero todas se conservan.

## 5. Variable objetivo

`median_house_value` — valor medio de una vivienda en el distrito, en USD.
Rango observado: 14,999 a 500,001. El tope 500,001 corresponde a registros
censurados del dataset original; se conservan y documentan como dato real.

## 6. Análisis exploratorio

- **Dimensiones:** 20,640 filas × 10 columnas.
- **Nulos:** solo `total_bedrooms` (207 valores, **1.003%**).
- **Duplicados:** 0.
- **Categóricas:** `ocean_proximity` con 5 categorías
  (`<1H OCEAN` 9,136 · `INLAND` 6,551 · `NEAR OCEAN` 2,658 · `NEAR BAY`
  2,290 · `ISLAND` 5).
- **Outliers (IQR):** `total_rooms` (6.24%), `total_bedrooms` (6.16%),
  `population` (5.79%), `households` (5.91%), `median_income` (3.30%),
  `median_house_value` (5.19%). Son distritos grandes reales (p. ej. Los
  Ángeles) y el tope censurado; **se conservan**.
- **Correlación:** `median_income` es el predictor más correlacionado con el
  objetivo (r = 0.688). Alta correlación entre conteos: `total_rooms`–
  `total_bedrooms` (0.93), `total_bedrooms`–`households` (0.98),
  `longitude`–`latitude` (−0.93).

## 7. Preprocesamiento

| Paso | Técnica | Justificación |
| ---- | ------- | ------------- |
| Imputación | `SimpleImputer(strategy="median")` sobre `total_bedrooms` | Solo 1% de nulos; la mediana es robusta |
| Codificación | `OneHotEncoder(handle_unknown="ignore", drop="first")` sobre `ocean_proximity` | Binarias; `drop="first"` evita la trampa de las variables dummy |
| Escalado | `StandardScaler` sobre las numéricas | Media 0 y desviación 1; evita que escalas grandes dominen |
| Train/Test | 80% / 20%, `random_state=42` | Sin data leakage (transformadores solo con train) |

## 8. Flujo del modelo

```
housing(1).csv
 ↓
EDA (nulos, outliers, correlación, VIF)
 ↓
Preprocesamiento: Imputación (mediana) → One-Hot (ocean_proximity)
 → StandardScaler (numéricas)
 ↓
[PolynomialFeatures grado 2/3 — solo modelos polinomiales]
 ↓
LinearRegression (mínimos cuadrados)
 ↓
Evaluación (MAE, MSE, RMSE, R² — train vs test)
 ↓
Guardado con joblib (models/california_housing/*.pkl)
 ↓
Streamlit carga el pipeline y predice (no reentrena)
```

## 9. Regresión Lineal Múltiple

**¿Qué es?** Extiende la regresión lineal simple a múltiples variables,
modelando el objetivo como combinación lineal de los descriptores.

**¿Por qué se utiliza?** Modelo base: simple, rápido e interpretable.

**¿Cómo funciona?** Estima los coeficientes β que minimizan el error
cuadrático (mínimos cuadrados ordinarios, OLS).

**Fórmula:**

```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

**Variables utilizadas:** las 12 características finales (8 numéricas
estandarizadas + 4 binarias de `ocean_proximity`).

**Interpretación:** al estar estandarizadas, un coeficiente positivo eleva el
valor predicho y uno negativo lo reduce. `median_income` tiene el mayor
efecto positivo (≈ +75,168 USD); `ISLAND` aporta la mayor prima de zona
(≈ +136,125 USD).

## 10. Regresión Polinomial

**¿Qué es?** Regresión lineal aplicada a características polinomiales:
potencias e interacciones de las variables originales.

**¿Por qué se utiliza?** La relación no es estrictamente lineal; los
términos de mayor grado capturan curvaturas.

**¿Cómo funciona?** `PolynomialFeatures` genera nuevas columnas con las
potencias (x², x³) y productos cruzados (xᵢ·xⱼ); luego `LinearRegression`
aprende los coeficientes sobre esa matriz ampliada.

**Fórmula (una variable):**

```
ŷ = β₀ + β₁·x + β₂·x² + ... + βₙ·xⁿ
```

**Grados utilizados:** 2 (78 características) y 3 (364 características).
Grados mayores se evitan por sobreajuste y costo computacional.

## 11. Modelos implementados

1. `LinearRegression` — `[preprocesador → LinearRegression]`.
2. `PolynomialFeatures(degree=2, include_bias=False)` + `LinearRegression`.
3. `PolynomialFeatures(degree=3, include_bias=False)` + `LinearRegression`.

Los tres comparten el mismo preprocesador; la comparación es justa. Se
entrenan con `python -m cases.california_housing.train` y se guardan con
`joblib`.

## 12. Métricas

- **MAE:** promedio de |real − predicho|. Menor = mejor; robusta a outliers.
- **MSE:** promedio de (real − predicho)². Penaliza más los errores grandes.
- **RMSE:** raíz del MSE, en las mismas unidades (USD). Menor = mejor.
- **R²:** proporción de la varianza explicada. Máximo 1; un R² negativo
  indica un modelo peor que predecir la media.

## 13. Resultados

Tabla real obtenida durante el entrenamiento (test 20%, 4,128 registros):

```
Modelo                  MAE       MSE          RMSE       R²
------------------------------------------------------------
Regresión Lineal        50,670    4,908,290,571 70,059     0.625
Polinomial grado 2      46,386    4,507,006,486 67,134     0.656
Polinomial grado 3      43,248    9,293,897,642 96,405     0.291
```

## 14. Selección del mejor modelo

**Regresión Polinomial grado 2** es el mejor: mayor R² en prueba (0.656) y
menor RMSE (67,134 USD). El **grado 3 sobreajusta**: R² de entrenamiento
0.781 (el más alto) pero cae a 0.291 en prueba con RMSE de 96,405 USD —
memoriza el entrenamiento y no generaliza.

## 15. Predicción

La aplicación Streamlit carga los pipelines guardados (`joblib`) — **no
reentrena en tiempo real**. El usuario introduce las 9 variables (con rangos
reales del dataset) y el modelo devuelve el valor estimado de la vivienda.

## 16. Fórmula del modelo

Para el modelo **lineal** (variables estandarizadas y codificadas):

```
ŷ = 219,899.78
    − 53,826.65 · longitude
    − 54,415.70 · latitude
    + 13,889.87 · housing_median_age
    − 13,094.25 · total_rooms
    + 43,068.18 · total_bedrooms
    − 43,403.43 · population
    + 18,382.20 · households
    + 75,167.77 · median_income
    − 39,786.66 · ocean_INLAND
    + 136,125.07 · ocean_ISLAND
    − 5,136.64 · ocean_NEAR BAY
    + 3,431.14 · ocean_NEAR OCEAN
```

Los coeficientes corresponden a variables **estandarizadas**; la categoría
de referencia (`NEAR BAY`, omitida por `drop="first"`) queda en el
intercepto.

## 17. Arquitectura

```
Dataset → EDA → Preprocesamiento → Train/Test (80/20)
 → Regresión (lineal · polinomial 2 · polinomial 3)
 → Evaluación (train vs test) → joblib → Streamlit → Predicción
```

## 18. Estructura del proyecto

```
cases/california_housing/
├── __init__.py
├── config.py             # Rutas, features, random_state
├── preprocessing.py      # load_data + build_preprocessor
├── train.py              # Entrenamiento y guardado
├── evaluate.py           # Métricas y gráficos
├── predict.py            # Carga de modelos + predicción
├── app.py                # Interfaz Streamlit del caso
└── README.md             # Documentación académica

models/california_housing/
├── linear_regression.pkl
├── polynomial_degree_2.pkl
├── polynomial_degree_3.pkl
├── feature_names.json
└── input_ranges.json

results/california_housing/
├── metrics.csv
├── input_ranges.json
├── correlation_matrix.png
├── target_distribution.png
├── metrics_comparison.png
├── actual_vs_predicted_*.png
└── residuals_*.png
```

## 19. Instalación

```bash
pip install -r requirements.txt
```

Entrenamiento (opcional; los `.pkl` ya están incluidos):

```bash
python -m cases.california_housing.train
```

## 20. Ejecución

```bash
streamlit run app.py
```

El menú principal lista los tres casos; "Ingresar al Caso 1" abre la interfaz
con las secciones: **Información · Descriptores · Modelos · Métricas ·
Predicción**.

## 21. Conclusiones

1. `median_income` es el descriptor con mayor influencia (r = 0.688).
2. La multicolinealidad entre conteos (VIF > 10) dificulta interpretar
   coeficientes individuales pero no impide predecir.
3. Los outliers (~6%) y los valores censurados son datos reales; se
   conservaron.
4. **La polinomial de grado 2 es el mejor modelo** (R² 0.656), superando a
   la lineal.
5. El grado 3 demuestra **sobreajuste**: R² train 0.781 frente a 0.291 test.

## 22. Referencias

- Scikit-learn: California Housing dataset, `LinearRegression`,
  `PolynomialFeatures`, `Pipeline`, `ColumnTransformer`.
- Documentación oficial de Streamlit.

## 23. Preguntas para la exposición — Conceptos

**¿Qué es la regresión lineal múltiple?** Un modelo que predice una variable
continua como combinación lineal de varias variables predictoras.

**¿Qué es R²?** La proporción de la varianza de la variable objetivo que el
modelo explica. 1 es perfecto; 0 es predecir la media; negativo es peor que
la media.

**¿Qué es el RMSE?** La raíz cuadrada del error cuadrático medio; expresa el
error típico del modelo en las mismas unidades que el objetivo (USD).

**¿Qué es el sobreajuste (overfitting)?** Cuando el modelo memoriza el
entrenamiento (R² train muy alto) pero falla en datos nuevos (R² test bajo).

**¿Qué es el VIF?** Factor de inflación de la varianza; mide la
multicolinealidad de un predictor. Valores > 10 indican correlación severa
con otros predictores.

## 24. Preguntas para la exposición — Resultados

**¿Por qué el grado 3 sobreajusta?** Porque con 364 características y solo
20,640 registros (4,128 de prueba), el modelo ajusta ruido en lugar de la
relación real.

**¿Por qué `median_income` es el predictor más importante?** Porque tiene la
mayor correlación con el valor de la vivienda (r = 0.688) y el mayor
coeficiente estandarizado.

**¿Por qué se usa One-Hot Encoding?** Para convertir la variable categórica
`ocean_proximity` (5 zonas) en variables binarias numéricas que el modelo
puede usar.

**¿Qué implica el R² de 0.656 del mejor modelo?** Que el modelo explica el
65.6% de la variabilidad del valor de la vivienda; el resto depende de
factores no incluidos.

## 25. Preguntas para la exposición — Aplicación

**¿La aplicación reentrena el modelo?** No: los pipelines se guardan con
`joblib` y la app solo carga y predice, lo que la hace rápida y apta para
desplegar en la nube.

**¿Qué pasa si introduzco valores fuera del rango del dataset?** La
interfaz limita los controles a los rangos reales (`input_ranges.json`), y el
modelo estandariza con la media/desviación de entrenamiento; valores muy
lejanos producirían predicciones poco confiables.
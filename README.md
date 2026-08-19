# Modelos de Inteligencia Artificial

Plataforma de modelos de **regresión** (Regresión Lineal Múltiple y
Regresión Polinomial) aplicados a distintos problemas de predicción,
construida con Python, scikit-learn y Streamlit.

## Casos de estudio

| Caso | Problema | Estado |
| ---- | -------- | ------ |
| Caso 1 — California Housing | Predicción del valor medio de viviendas | ✅ Implementado |
| Caso 2 — Wine Quality | Estimación de la calidad del vino | ✅ Implementado |
| Caso 3 — Diabetes | Progresión de la diabetes | ✅ Implementado |

Cada caso implementa EDA, preprocesamiento, entrenamiento (Lineal +
Polinomial grados 2 y 3), evaluación y una interfaz propia dentro del
menú principal.

Documentación académica completa de cada caso:

- **Caso 1:** `cases/california_housing/README.md`
- **Caso 2:** `cases/wine_quality/README.md`
- **Caso 3:** `cases/diabetes/README.md`

A continuación, la documentación académica completa del **Caso 1**.

---

# California Housing — Modelos de Regresión

## 1. Descripción del proyecto

El problema consiste en estimar el **valor medio de una vivienda** en cada
distrito censal de California a partir de características geográficas,
demográficas y económicas del distrito. Se trata de un problema clásico de
**regresión supervisada**: a partir de un conjunto de variables
predictoras (descriptores), el modelo aprende a estimar una variable
numérica continua.

La aplicación se construye con **Python + Streamlit** y permite:

1. Explorar los datos y el análisis realizado.
2. Comparar los modelos de regresión implementados.
3. Introducir las características de un distrito y obtener una predicción.

## 2. Objetivo

Predecir el valor medio de una vivienda (`median_house_value`) en distritos
de California y comparar el desempeño de dos familias de modelos:

- **Regresión Lineal Múltiple** (modelo base).
- **Regresión Polinomial** con grados 2 y 3 (relaciones no lineales).

## 3. Dataset

- **Nombre:** `data/housing(1).csv` (dataset *California Housing*).
- **Registros:** 20,640 distritos censales.
- **Variables:** 10 (9 predictoras + 1 objetivo).
- **Variable objetivo:** `median_house_value` (valor medio de la vivienda en USD).

El dataset es el clásico *California Housing* publicado con el ejemplo de
*scikit-learn*. Contiene valores **censurados** en el tope de `500,001`
(965 registros, 4.68%) y 207 valores nulos en `total_bedrooms`.

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
| `median_income` | Ingreso medio de los habitantes | Numérica continua | Poder adquisitivo (mayor correlación con el objetivo) |
| `ocean_proximity` | Proximidad al océano (5 categorías) | Categórica | Zona geográfica (One-Hot Encoding) |

**Ninguna variable fue descartada.** Las correlaciones altas entre los
conteos (`total_rooms`, `total_bedrooms`, `population`, `households`) se
documentaron mediante VIF, pero todas se conservan como predictores.

## 5. Variable objetivo

`median_house_value` — valor medio de una vivienda en el distrito, en USD.
Rango observado: 14,999 a 500,001. El valor 500,001 corresponde a
registros censurados del dataset original (el censo truncó el precio);
se conservan y se documentan como dato real.

## 6. Análisis Exploratorio

- **Dimensiones:** 20,640 filas × 10 columnas.
- **Nulos:** solo `total_bedrooms` (207 valores, **1.003%**).
- **Duplicados:** 0.
- **Categóricas:** `ocean_proximity` con 5 categorías
  (`<1H OCEAN` 9,136 · `INLAND` 6,551 · `NEAR OCEAN` 2,658 ·
  `NEAR BAY` 2,290 · `ISLAND` 5).
- **Outliers (IQR):** `total_rooms` (6.24%), `total_bedrooms` (6.16%),
  `population` (5.79%), `households` (5.91%), `median_income` (3.30%),
  `median_house_value` (5.19%). Corresponden a distritos grandes reales
  (p. ej. Los Ángeles) y al tope censurado; **se conservan**.
- **Correlación:** `median_income` es el predictor más correlacionado con
  el objetivo (r = 0.688). Existe alta correlación entre los conteos:
  `total_rooms`–`total_bedrooms` (0.93), `total_bedrooms`–`households`
  (0.98), `longitude`–`latitude` (−0.93).
- **VIF:**

```
Variable                VIF
--------------------------------
longitude               18.03
latitude                19.93
housing_median_age       1.32
total_rooms             12.35
total_bedrooms          27.04
population               6.34
households              28.32
median_income            1.74
ocean_proximity_INLAND   2.85
ocean_proximity_ISLAND   1.00
ocean_proximity_NEAR BAY 1.57
ocean_proximity_NEAR OCEAN 1.20
```

Los VIF de `households`, `total_bedrooms`, `latitude`, `longitude` y
`total_rooms` superan 10, lo que indica **multicolinealidad**. Consecuencia:
los coeficientes individuales son poco estables/interpretables, pero la
predicción global del modelo sigue siendo válida. No se eliminan variables.

## 7. Preprocesamiento

Se construye un `ColumnTransformer` compartido por todos los modelos:

| Paso | Técnica | Justificación |
| ---- | ------- | ------------- |
| Imputación | `SimpleImputer(strategy="median")` sobre `total_bedrooms` | Solo 1% de nulos; la mediana es robusta a outliers |
| Codificación | `OneHotEncoder(handle_unknown="ignore", drop="first")` sobre `ocean_proximity` | Convierte la categórica en binarias; `drop="first"` evita la trampa de las variables dummy |
| Escalado | `StandardScaler` sobre las numéricas | Media 0 y desviación 1; evita que escalas grandes dominen el modelo |
| Train/Test | 80% / 20%, `random_state=42` | Los transformadores solo se ajustan con train (sin data leakage) |

Pipeline conceptual:

```
Datos
 ↓
Imputación (mediana)
 ↓
One-Hot Encoding (ocean_proximity)
 ↓
StandardScaler (numéricas)
 ↓
[PolynomialFeatures grado 2/3 — solo modelos polinomiales]
 ↓
LinearRegression
 ↓
Predicción
```

## 8. Regresión Lineal Múltiple

**¿Qué es?** Extiende la regresión lineal simple a múltiples variables
predictoras, modelando la variable objetivo como una combinación lineal
de los descriptores.

**¿Por qué se utiliza?** Es el modelo base: simple, rápido e
interpretable. Permite medir el efecto marginal de cada variable y sirve
de referencia frente a modelos más complejos.

**¿Cómo funciona?** Estima los coeficientes β que minimizan el error
cuadrático entre las predicciones y los valores reales (mínimos
cuadrados ordinarios, OLS).

**Fórmula matemática:**

```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

Donde `ŷ` es el valor predicho, `β₀` el intercepto, `β₁ … βₙ` los
coeficientes y `x₁ … xₙ` los descriptores (después de estandarizar y
codificar).

**Variables utilizadas:** las 12 características finales (8 numéricas
estandarizadas + 4 binarias de `ocean_proximity`).

**Interpretación básica de los coeficientes:** al estar estandarizadas,
un coeficiente positivo indica que el aumento de esa variable (una
desviación estándar) eleva el valor predicho, y uno negativo lo reduce.
`median_income` tiene el mayor efecto positivo (≈ +75,168 USD por
desviación estándar); `ISLAND` aporta la mayor prima de zona
(≈ +136,125 USD).

## 9. Regresión Polinomial

**¿Qué es?** Regresión lineal aplicada a características polinomiales:
potencias e interacciones de las variables originales.

**¿Por qué se utiliza?** La relación entre los descriptores y el valor de
la vivienda no es estrictamente lineal; los términos de mayor grado capturan
curvaturas.

**¿Cómo funciona?** `PolynomialFeatures` genera nuevas columnas con las
potencias (x², x³) y productos cruzados (xᵢ·xⱼ); luego `LinearRegression`
aprende los coeficientes sobre esa matriz ampliada.

**Fórmula matemática (una variable):**

```
ŷ = β₀ + β₁·x + β₂·x² + ... + βₙ·xⁿ
```

Con múltiples variables, `PolynomialFeatures` genera todas las
combinaciones de potencias e interacciones hasta el grado indicado.

**Grados utilizados:** 2 y 3. Se evitan grados mayores por riesgo de
sobreajuste y costo computacional.

**Diferencias frente a la lineal:** puede modelar relaciones no lineales,
pero añade muchas características (grado 2 → 78, grado 3 → 364) y aumenta
el riesgo de sobreajuste si el grado es alto.

## 10. Modelos implementados

1. `LinearRegression` — pipeline `[preprocesador → LinearRegression]`.
2. `PolynomialFeatures(degree=2, include_bias=False)` + `LinearRegression`.
3. `PolynomialFeatures(degree=3, include_bias=False)` + `LinearRegression`.

Los tres comparten el mismo preprocesador, por lo que la comparación es
justa. Se entrenan con `python -m cases.california_housing.train` y se
guardan con `joblib`.

## 11. Métricas

- **MAE** (Error Absoluto Medio): promedio de |real − predicho|. Menor es
  mejor; es robusta a outliers.
- **MSE** (Error Cuadrático Medio): promedio de (real − predicho)². Penaliza
  más los errores grandes.
- **RMSE** (Raíz del MSE): en las mismas unidades que la variable objetivo
  (USD). Menor es mejor.
- **R²** (Coeficiente de determinación): proporción de la varianza explicada
  por el modelo. Máximo 1 (predicción perfecta); un R² negativo indica un
  modelo peor que predecir la media.

## 12. Resultados

Tabla comparativa obtenida realmente durante el entrenamiento
(test 20%, 4,128 registros):

```
Modelo                  MAE       MSE          RMSE       R²
------------------------------------------------------------
Regresión Lineal        50,670    4,908,290,571 70,059     0.625
Polinomial grado 2      46,386    4,507,006,486 67,134     0.656
Polinomial grado 3      43,248    9,293,897,642 96,405     0.291
```

## 13. Selección del mejor modelo

**Regresión Polinomial grado 2** es el mejor modelo:

- Mayor R² en prueba (0.656) y menor RMSE (67,134 USD).
- Mejora al modelo lineal (R² 0.625) al capturar relaciones no lineales.

El **grado 3 sobreajusta**: alcanza R² de entrenamiento 0.781 (el más alto)
pero el R² en prueba cae a 0.291 y el RMSE sube a 96,405 USD — memoriza el
entrenamiento y no generaliza.

## 14. Predicción

La aplicación Streamlit carga los pipelines guardados con `joblib`
(`models/california_housing/*.pkl`) — **no reentrena en tiempo real**.
El usuario introduce las 9 variables (con rangos reales del dataset) y el
modelo devuelve el valor estimado de la vivienda.

## 15. Fórmula del modelo

Para el modelo **lineal** (con variables estandarizadas y codificadas):

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

Los coeficientes corresponden a las variables **estandarizadas** y
binarias; la variable `NEAR BAY` de referencia (categoría omitida por
`drop="first"`) queda representada en el intercepto. La regresión
polinomial usa los mismos descriptores expandidos a potencias e
interacciones.

## 16. Arquitectura

```
Dataset (housing(1).csv)
 ↓
EDA (nulos, outliers, correlación, VIF)
 ↓
Preprocesamiento (imputación, One-Hot, StandardScaler)
 ↓
Train/Test (80/20, random_state=42)
 ↓
Regresión (lineal · polinomial 2 · polinomial 3)
 ↓
Evaluación (MAE, MSE, RMSE, R² — train vs test)
 ↓
Modelo entrenado (joblib)
 ↓
Streamlit (menú principal → caso 1)
 ↓
Predicción
```

## 17. Estructura del proyecto

```
Proyecto03/
├── app.py                        # Menú principal + despacho de casos
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── housing(1).csv
│   ├── WineQT.csv
│   └── diabetes.tab.txt
├── notebooks/
│   ├── california_housing_analysis.ipynb
│   ├── wine_quality_analysis.ipynb
│   └── diabetes_analysis.ipynb
├── cases/
│   ├── __init__.py
│   ├── registry.py               # Registro de casos (menú extensible)
│   ├── california_housing/
│   │   ├── __init__.py
│   │   ├── config.py             # Rutas, features, random_state
│   │   ├── preprocessing.py      # load_data + build_preprocessor
│   │   ├── train.py              # Entrenamiento y guardado
│   │   ├── evaluate.py           # Métricas y gráficos
│   │   ├── predict.py            # Carga de modelos + predicción
│   │   └── app.py                # Interfaz Streamlit del caso
│   └── wine_quality/
│       ├── __init__.py
│       ├── config.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       ├── app.py
│       └── README.md             # Documentación académica del caso
│   └── diabetes/
│       ├── __init__.py
│       ├── config.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       ├── app.py
│       └── README.md             # Documentación académica del caso
├── models/
│   ├── california_housing/
│   │   ├── linear_regression.pkl
│   │   ├── polynomial_degree_2.pkl
│   │   ├── polynomial_degree_3.pkl
│   │   ├── feature_names.json
│   │   └── input_ranges.json
│   └── wine_quality/
│       ├── linear_regression.pkl
│       ├── polynomial_degree_2.pkl
│       ├── polynomial_degree_3.pkl
│       ├── feature_names.json
│       └── input_ranges.json
│   └── diabetes/
│       ├── linear_regression.pkl
│       ├── polynomial_degree_2.pkl
│       ├── polynomial_degree_3.pkl
│       ├── feature_names.json
│       └── input_ranges.json
├── results/
│   ├── california_housing/
│   │   ├── metrics.csv
│   │   ├── input_ranges.json
│   │   ├── correlation_matrix.png
│   │   ├── target_distribution.png
│   │   ├── metrics_comparison.png
│   │   ├── actual_vs_predicted_*.png
│   │   └── residuals_*.png
│   └── wine_quality/
│       ├── metrics.csv
│       ├── input_ranges.json
│       ├── correlation_matrix.png
│       ├── target_distribution.png
│       ├── metrics_comparison.png
│       ├── actual_vs_predicted_*.png
│       └── residuals_*.png
│   └── diabetes/
│       ├── metrics.csv
│       ├── input_ranges.json
│       ├── correlation_matrix.png
│       ├── target_distribution.png
│       ├── metrics_comparison.png
│       ├── actual_vs_predicted_*.png
│       └── residuals_*.png
└── .streamlit/
    └── config.toml
```

**Extensión a otros casos:** para agregar un caso basta crear `cases/<caso>/`
con su `app.py` (función `render()`) y registrarlo en `cases/registry.py`.
El menú principal no requiere cambios. Así se incorporaron los tres casos.

## 18. Instalación

```bash
pip install -r requirements.txt
```

Entrenamiento de los modelos (opcional; los `.pkl` ya están incluidos):

```bash
python -m cases.california_housing.train
python -m cases.wine_quality.train
python -m cases.diabetes.train
```

## 19. Ejecución

```bash
streamlit run app.py
```

Al abrir la aplicación aparece el **menú principal**:

```
MODELOS DE INTELIGENCIA ARTIFICIAL
Seleccione un caso de estudio

[ California Housing ]        -> Ingresar al Caso 1
[ Wine Quality ]              -> Ingresar al Caso 2
[ Diabetes ]                  -> Ingresar al Caso 3
```

Al entrar a cualquiera de los tres casos se navega por:
**Información · Descriptores · Modelos · Métricas · Predicción**, con botón
para volver al menú principal.

### Preparación para Streamlit Cloud

- Todas las rutas son relativas (no se depende de rutas locales).
- Los modelos y datos ya están en el repositorio (no se reentrenan en la
  nube).
- Ejecutar: `pip install -r requirements.txt && streamlit run app.py`.

## 20. Conclusiones

1. `median_income` es el descriptor con mayor influencia sobre el valor de
   la vivienda (r = 0.688).
2. Existe multicolinealidad entre los conteos demográficos (VIF > 10);
   dificulta interpretar coeficientes individuales pero no impide predecir.
3. Los outliers (~6% por IQR) y los valores censurados en 500,001 son datos
   reales del dataset; se conservaron y documentaron.
4. La **Regresión Polinomial de grado 2 es el mejor modelo** (R² 0.656 en
   prueba), superando a la lineal.
5. El grado 3 demuestra el **sobreajuste**: R² de entrenamiento 0.781 frente
   a solo 0.291 en prueba.
6. La arquitectura con registro de casos permitió incorporar Wine Quality
   y dejar preparada la incorporación de Diabetes sin rediseñar la
   aplicación.

## 21. Caso 2 — Wine Quality (resumen)

**Problema:** estimar la calidad de un vino (puntuación 0-10) a partir de
sus propiedades físico-químicas (acidez, azúcar residual, cloruros, alcohol,
etc.). Dataset real `data/WineQT.csv`: **1,143 registros × 13 columnas**, sin
nulos ni duplicados; la columna `Id` (índice) se descarta y `quality` (3-8,
discreta) se modela como variable continua por el planteamiento de regresión.

**Descriptores (11):** `fixed acidity`, `volatile acidity`, `citric acid`,
`residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`,
`density`, `pH`, `sulphates`, `alcohol`.

**EDA:** los predictores más correlacionados con la calidad son `alcohol`
(+0.49) y `volatile acidity` (−0.41). La multicolinealidad es severa
(VIF: `density` 1435, `pH` 1043, `alcohol` 124); se documenta y no se
eliminan variables. Los outliers (residual sugar 9.6%, chlorides 6.7%) son
vinos reales atípicos y se conservan.

**Resultados reales (test 20%):**

```
Modelo                  MAE    MSE     RMSE    R²
---------------------------------------------------
Regresión Lineal        0.477  0.380   0.616   0.317   ← mejor
Polinomial grado 2      0.487  0.400   0.633   0.281
Polinomial grado 3      0.866  4.156   2.039  -6.469  ← sobreajuste
```

**Conclusión:** la Regresión Lineal es el mejor modelo (R² test 0.317);
el grado 3 sobreajusta severamente (R² train 0.715 → test −6.469).

Documentación completa en `cases/wine_quality/README.md`.

## 22. Caso 3 — Diabetes (resumen)

**Problema:** predecir la **progresión cuantitativa de la diabetes** un año
después del inicio a partir de 10 variables basales. Dataset real
`data/diabetes.tab.txt`: archivo **separado por tabulaciones**, **442
registros × 11 columnas**, sin nulos ni duplicados; objetivo `Y` (25–346,
media 152.1).

**Descriptores (10):** `AGE`, `SEX` (1/2 numérica binaria), `BMI`, `BP`,
`S1`–`S6`.

**EDA:** los predictores más correlacionados con la progresión son `BMI`
(+0.586) y `S5` (+0.566); `S3` correlaciona negativamente (−0.395). La
multicolinealidad entre S1-S6 es severa (VIF: S1 577, S5 277, S2 245); se
documenta y no se eliminan variables. Los outliers (≤2%) son pacientes
reales y se conservan.

**Resultados reales (test 20%):**

```
Modelo                  MAE      MSE       RMSE     R²
---------------------------------------------------------
Regresión Lineal        42.79    2,900.19  53.85    0.453   ← mejor
Polinomial grado 2      43.58    3,096.03  55.64    0.416
Polinomial grado 3     162.32   80,980.76 284.57  -14.28   ← sobreajuste
```

**Conclusión:** la Regresión Lineal es el mejor modelo (R² test 0.453); el
grado 3 sobreajusta severamente (R² train 0.877 → test −14.28) y produce
predicciones absurdas fuera del rango de los datos.

Documentación completa en `cases/diabetes/README.md`.

## 23. Comparación general

| Caso | Dataset | Registros | Mejor modelo | R² test | RMSE test |
| ---- | ------- | --------- | ------------ | ------- | --------- |
| 1 · California Housing | `housing(1).csv` | 20,640 | Polinomial grado 2 | 0.656 | 67,134 USD |
| 2 · Wine Quality | `WineQT.csv` | 1,143 | Lineal | 0.317 | 0.616 |
| 3 · Diabetes | `diabetes.tab.txt` | 442 | Lineal | 0.453 | 53.85 |

El **Caso 1** gana con la polinomial de grado 2 por la gran cantidad de
datos (20,640); en los **Casos 2 y 3** (miles o cientos de registros) la
**lineal** es la mejor y los grados altos **sobreajustan** drásticamente.
# Ejercicio 1 — California Housing: Documentación por Fases

**Problema:** predecir el valor medio de una vivienda (`median_house_value`)
en los distritos censales de California a partir de 9 descriptores.
**Dataset:** `data/housing(1).csv` · **20,640 registros × 10 columnas.**

---

## FASE A — Análisis Exploratorio y Preprocesamiento

### A.1 Carga de datos y limpieza

Se carga el CSV con `pandas` (`pd.read_csv`). El dataset es el clásico
*California Housing*: 20,640 distritos censales y 10 variables (9
predictoras + 1 objetivo). No se elimina ninguna variable: todos los
descriptores se conservan.

### A.2 Valores nulos y su tratamiento

- Único campo con nulos: `total_bedrooms` (**207 valores, 1.003%**).
- **Tratamiento:** imputación con la **mediana** mediante
  `SimpleImputer(strategy="median")`. Se elige la mediana y no la media
  porque es robusta ante outliers (el distrito con más dormitorios no debe
  sesgar el valor imputado).
- La imputación vive **dentro del pipeline** de scikit-learn, por lo que se
  ajusta solo con los datos de entrenamiento (sin *data leakage*).

### A.3 Outliers y su tratamiento

Detección con la **regla del IQR** (valores fuera de
`[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`):

| Variable | Outliers (%) |
| -------- | ------------ |
| `total_rooms` | 6.24% |
| `total_bedrooms` | 6.16% |
| `households` | 5.91% |
| `population` | 5.79% |
| `median_house_value` | 5.19% |
| `median_income` | 3.30% |

**Tratamiento:** **se conservan.** Corresponden a distritos reales de gran
tamaño (p. ej. Los Ángeles) y al tope censurado de 500,001 USD. Eliminarlos
sesgaría la muestra; se documentan y se mantienen como observaciones válidas.

### A.4 Análisis de multicolinealidad (correlación / VIF)

**Matriz de correlación (Pearson):**

- `median_income` es el predictor más correlacionado con el objetivo
  (**r = 0.688**).
- Alta correlación entre los conteos: `total_rooms`–`total_bedrooms`
  (0.93), `total_bedrooms`–`households` (0.98), `longitude`–`latitude`
  (−0.93).

**VIF (Factor de Inflación de la Varianza):**

```
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

**Interpretación:** `households`, `total_bedrooms`, `latitude`, `longitude`
y `total_rooms` superan VIF = 10, lo que indica **multicolinealidad**.
Consecuencia: los coeficientes individuales son poco estables e
interpretables, pero la **predicción global del modelo sigue siendo
válida**. **No se eliminan variables** (decision documentada).

### A.5 Normalización / estandarización

Se usa **StandardScaler** sobre las variables numéricas: media 0 y
desviación estándar 1. La variable categórica `ocean_proximity` se codifica
con **One-Hot Encoding** (`drop="first"` para evitar la trampa de las
variables dummy). La estandarización evita que variables con escalas
grandes (precios, conteos) dominen el modelo.

### A.6 Resumen del preprocesador

```
ColumnTransformer
├── num: SimpleImputer(mediana) → StandardScaler
│         (longitude, latitude, housing_median_age, total_rooms,
│          total_bedrooms, population, households, median_income)
└── cat: OneHotEncoder(drop="first", handle_unknown="ignore")
          (ocean_proximity → 4 binarias)
```

Resultado: **12 características finales** (8 numéricas estandarizadas + 4
binarias). División **80% / 20%** con `random_state=42` (4,128 registros de
prueba).

---

## FASE B — Modelamiento Estadístico y Comparativa

### B.1 Regresión Lineal Múltiple — cómo funciona

**Concepto:** modela el objetivo como combinación lineal de los
descriptores:

```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

**Cómo se aplica aquí:** `LinearRegression` de scikit-learn estima los
coeficientes β por **mínimos cuadrados ordinarios (OLS)**, minimizando la
suma de los errores al cuadrado entre predicciones y valores reales. Se
entrena sobre las 12 características estandarizadas/codificadas. Al estar
estandarizadas, cada coeficiente indica el efecto de *una desviación
estándar* de esa variable sobre el precio: `median_income` aporta
≈ +75,168 USD y `ISLAND` la mayor prima de zona (≈ +136,125 USD).

### B.2 Regresión Polinomial — cómo funciona

**Concepto:** aplica regresión lineal sobre características polinomiales
(potencias e interacciones de las variables originales):

```
ŷ = β₀ + β₁·x + β₂·x² + ... + βₙ·xⁿ
```

**Cómo se aplica aquí:** `PolynomialFeatures(degree=n, include_bias=False)`
genera nuevas columnas con x², x³ y los productos cruzados xᵢ·xⱼ; luego
`LinearRegression` aprende los coeficientes sobre esa matriz ampliada.
Grados evaluados: **2** (78 características) y **3** (364 características).
Grados mayores se evitan por sobreajuste y costo computacional.

### B.3 Entrenamiento y comparativa

Los tres modelos comparten el mismo preprocesador (comparación justa) y se
entrenan con `python -m cases.california_housing.train`. Métricas reales
obtenidas en prueba (test 20%):

| Modelo | MAE | MSE | RMSE | R² |
| ------ | --- | --- | ---- | -- |
| **Regresión Lineal** | 50,670 | 4,908,290,571 | 70,059 | **0.625** |
| **Polinomial grado 2** | 46,386 | 4,507,006,486 | 67,134 | **0.656** ← mejor |
| Polinomial grado 3 | 43,248 | 9,293,897,642 | 96,405 | 0.291 |

**Selección del mejor modelo:** la **polinomial de grado 2** (R² 0.656,
RMSE 67,134). El **grado 3 sobreajusta**: R² de entrenamiento 0.781 (el más
alto) frente a solo 0.291 en prueba — memoriza el entrenamiento y no
generaliza.

### B.4 Métricas utilizadas

- **MAE:** promedio de |real − predicho|. Robusta a outliers.
- **MSE:** promedio de (real − predicho)². Penaliza errores grandes.
- **RMSE:** raíz del MSE, en las mismas unidades (USD).
- **R²:** proporción de varianza explicada (1 = perfecto, negativo = peor
  que predecir la media).

---

## FASE C — Desarrollo del Aplicativo Web Multi-Caso

### C.1 Arquitectura

```
data/housing(1).csv → cases/california_housing/
├── preprocessing.py   # carga + preprocesador
├── train.py           # entrena los 3 pipelines y guarda con joblib
├── evaluate.py        # métricas (metrics.csv) y gráficos PNG
├── predict.py         # carga el .pkl y predice
├── app.py             # interfaz Streamlit (función render())
└── README.md

models/california_housing/*.pkl      # pipelines serializados
results/california_housing/          # metrics.csv, input_ranges.json, *.png
```

Los modelos se guardan una vez con `joblib`; la aplicación **solo los
carga y predice, no reentrena** (rapidez y apto para la nube).

### C.2 Menú de navegación multi-caso

`app.py` raíz muestra el **menú principal** (`st.session_state["page"]`).
Cada caso se registra en `cases/registry.py` con botón "Ingresar al Caso N"
y módulo de interfaz:

```
MODELOS DE INTELIGENCIA ARTIFICIAL
[ California Housing ]  -> Ingresar al Caso 1
[ Wine Quality ]        -> Ingresar al Caso 2
[ Diabetes ]            -> Ingresar al Caso 3
```

Dentro del caso hay un sidebar con las secciones:
**Información · Descriptores · Modelos · Métricas · Predicción** y un botón
"← Volver al menú principal".

### C.3 Interacción dinámica (predicción en tiempo real)

En la sección **Predicción** el usuario:

1. Selecciona el **modelo** (Lineal / Polinomial grado 2 / grado 3).
2. Ingresa las 9 variables mediante controles (`slider`/`selectbox`)
   limitados a los **rangos reales del dataset** (`input_ranges.json`),
   con valores por defecto iguales a la mediana.
3. Presiona "Predecir"; la app carga el pipeline seleccionado y muestra el
   **valor estimado de la vivienda** (USD) con el modelo usado y las
   variables ingresadas.

La predicción es en tiempo real: el pipeline se carga en memoria (joblib) y
la operación es instantánea.

### C.4 Despliegue en la nube (Streamlit Community Cloud)

**Estado:** implementación local verificada (smoke test `health: ok`);
pendiente de desplegar. Pasos para desplegar:

1. Subir el repositorio a **GitHub** (rutas relativas; los `.pkl` y datos ya
   están incluidos; no se reentrena en la nube).
2. Entrar a **Streamlit Community Cloud** → *New app* → seleccionar
   repositorio, rama y archivo `app.py`.
3. La plataforma instala `requirements.txt` y publica la URL pública.

**Enlace de la aplicación (desplegada):** `[PEGAR AQUÍ EL LINK]`

**Capturas de la aplicación:**
- Menú principal (3 casos): `[PEGAR CAPTURA]`
- Interfaz del Caso 1 (Información): `[PEGAR CAPTURA]`
- Formulario de predicción y resultado: `[PEGAR CAPTURA]`
- Métricas y gráficos del caso: `[PEGAR CAPTURA]`

---

**Verificación del cumplimiento:** los archivos
`notebooks/california_housing_analysis.ipynb` (ejecutado sin errores),
`results/california_housing/metrics.csv`, `cases/california_housing/README.md`
y `ejemplos_de_prediccion.txt` respaldan las fases A y B; `app.py` +
`cases/registry.py` + `cases/california_housing/app.py` respaldan la fase C.
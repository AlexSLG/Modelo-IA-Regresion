# Ejercicio 2 — Wine Quality: Documentación por Fases

**Problema:** estimar la calidad de un vino (`quality`) a partir de sus
propiedades físico-químicas.
**Dataset:** `data/WineQT.csv` · **1,143 registros × 13 columnas**
(la columna `Id`, índice secuencial, se descarta → 12 efectivas).

---

## FASE A — Análisis Exploratorio y Preprocesamiento

### A.1 Carga de datos y limpieza

Se carga con `pd.read_csv`. **Limpieza:** se descarta la columna `Id`, que
es solo un índice secuencial de las filas y no una propiedad del vino.
Quedan **11 descriptores** físico-químicos + la variable objetivo
`quality`.

### A.2 Valores nulos y su tratamiento

- **Nulos:** 0 en todas las columnas.
- **Tratamiento:** de forma preventiva el pipeline incluye
  `SimpleImputer(strategy="median")`. Como no hay nulos, es un **no-op**
  (no modifica los datos), pero aporta robustez si el dataset cambiara.

### A.3 Outliers y su tratamiento

Detección por IQR:

| Variable | Outliers (%) |
| -------- | ------------ |
| `residual sugar` | 9.62% |
| `chlorides` | 6.74% |
| `fixed acidity` | 3.85% |
| `sulphates` | 3.76% |
| `total sulfur dioxide` | 3.50% |

**Tratamiento:** **se conservan.** Corresponden a **vinos reales**
atípicos (muy dulces, muy salados, muy ácidos). Eliminarlos eliminaría
observaciones válidas del mundo real; se documentan y mantienen.

### A.4 Análisis de multicolinealidad (correlación / VIF)

**Matriz de correlación (Pearson):**

- Con `quality`: `alcohol` (**+0.485**), `volatile acidity` (**−0.407**),
  `sulphates` (+0.258), `citric acid` (+0.241).
- Entre predictores: `fixed acidity`–`citric acid` (0.67),
  `fixed acidity`–`density` (0.68), `fixed acidity`–`pH` (−0.69),
  `free SO₂`–`total SO₂` (0.66).

**VIF:**

```
fixed acidity              74.55
volatile acidity           17.04
citric acid                 9.22
residual sugar              5.01
chlorides                   6.72
free sulfur dioxide         6.30
total sulfur dioxide        6.22
density                  1435.10
pH                       1043.34
sulphates                  21.51
alcohol                   123.96
```

**Interpretación:** `density`, `pH`, `alcohol`, `fixed acidity`,
`sulphates` y `volatile acidity` superan VIF = 10 (**multicolinealidad
severa**). Consecuencia: los coeficientes individuales son poco estables y
se interpretan con cautela, pero la **predicción global es válida**. **No
se eliminan variables.**

### A.5 Normalización / estandarización

Se usa **StandardScaler** sobre las 11 propiedades. Es necesario porque las
escalas difieren enormemente (densidad ~0.997 frente a `total sulfur
dioxide` hasta 289). Todas las variables son numéricas → no hay One-Hot
Encoding en este caso.

### A.6 Resumen del preprocesador

```
Dataset → descartar Id → SimpleImputer(mediana) → StandardScaler (11 vars)
```

División **80% / 20%** con `random_state=42` (229 registros de prueba).

---

## FASE B — Modelamiento Estadístico y Comparativa

### B.1 Regresión Lineal Múltiple — cómo funciona

**Concepto:** modela la calidad como combinación lineal de las propiedades:

```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

**Cómo se aplica aquí:** `LinearRegression` estima los coeficientes por
**mínimos cuadrados ordinarios (OLS)** sobre las 11 propiedades
estandarizadas. Intercepto 5.656. Coeficientes principales: `alcohol`
(+0.286) y `volatile acidity` (−0.239), es decir, más alcohol eleva la
calidad estimada y una mayor acidez volátil la reduce. Los coeficientes son
comparables entre sí por estar estandarizados.

### B.2 Regresión Polinomial — cómo funciona

**Concepto:** regresión lineal sobre características polinomiales:

```
ŷ = β₀ + β₁·x + β₂·x² + ... + βₙ·xⁿ
```

**Cómo se aplica aquí:** `PolynomialFeatures` genera potencias (x², x³) e
interacciones (xᵢ·xⱼ); luego `LinearRegression` aprende los coeficientes.
Grados evaluados: **2** (78 características) y **3** (364 características).
Con solo 1,143 registros, el grado 3 tiene alto riesgo de sobreajuste.

### B.3 Entrenamiento y comparativa

Los tres modelos comparten el preprocesador y se entrenan con
`python -m cases.wine_quality.train`. Métricas reales en prueba (test 20%):

| Modelo | MAE | MSE | RMSE | R² |
| ------ | --- | --- | ---- | -- |
| **Regresión Lineal** | 0.477 | 0.380 | 0.616 | **0.317** ← mejor |
| Polinomial grado 2 | 0.487 | 0.400 | 0.633 | 0.281 |
| Polinomial grado 3 | 0.866 | 4.156 | 2.039 | −6.469 |

Comparación train vs test (sobreajuste):

| Modelo | R² train | R² test |
| ------ | -------- | ------- |
| Lineal | 0.382 | **0.317** |
| Grado 2 | 0.459 | 0.281 |
| Grado 3 | 0.715 | **−6.469** |

**Selección del mejor modelo:** la **Regresión Lineal** (R² test 0.317). El
**grado 3 sobreajusta severamente** (R² train 0.715 → test −6.469): con 364
características memoriza el entrenamiento y colapsa en prueba.

### B.4 Métricas utilizadas

- **MAE:** promedio de |real − predicho|.
- **MSE:** promedio de (real − predicho)². Penaliza errores grandes.
- **RMSE:** raíz del MSE, en unidades de calidad.
- **R²:** proporción de varianza explicada; negativo = peor que la media.

*Nota:* `quality` es discreta (3-8), pero el ejercicio exige regresión, por
lo que se modela como **variable continua** y el modelo devuelve
predicciones decimales (p. ej. 5.72).

---

## FASE C — Desarrollo del Aplicativo Web Multi-Caso

### C.1 Arquitectura

```
data/WineQT.csv → cases/wine_quality/
├── preprocessing.py   # carga (descarta Id) + preprocesador
├── train.py           # entrena los 3 pipelines y guarda con joblib
├── evaluate.py        # métricas (metrics.csv) y gráficos PNG
├── predict.py         # carga el .pkl y predice
├── app.py             # interfaz Streamlit (función render())
└── README.md

models/wine_quality/*.pkl        # pipelines serializados
results/wine_quality/            # metrics.csv, input_ranges.json, *.png
```

Igual que los demás casos: los `.pkl` se entrenan una vez y la app **solo
carga y predice**.

### C.2 Menú de navegación multi-caso

El caso se registra en `cases/registry.py` con botón "Ingresar al Caso 2".
El menú principal (`app.py`) permite seleccionar cualquiera de los 3 casos.
Dentro del caso: sidebar con **Información · Descriptores · Modelos ·
Métricas · Predicción** y botón "← Volver al menú principal".

### C.3 Interacción dinámica (predicción en tiempo real)

En **Predicción** el usuario:

1. Selecciona el **modelo** (Lineal / Polinomial grado 2 / grado 3).
2. Ingresa las 11 propiedades físico-químicas con controles limitados a los
   **rangos reales del dataset** (`input_ranges.json`), con la mediana como
   valor por defecto.
3. Presiona "Predecir"; la app carga el pipeline seleccionado y muestra la
   **calidad estimada** en formato `X.XX / 10`, el modelo usado y los
   valores ingresados.

### C.4 Despliegue en la nube (Streamlit Community Cloud)

**Estado:** implementación local verificada (smoke test `health: ok`);
pendiente de desplegar. Pasos:

1. Subir el repositorio a **GitHub** (rutas relativas; `.pkl` y datos
   incluidos; no se reentrena en la nube).
2. **Streamlit Community Cloud** → *New app* → repositorio + `app.py`.
3. La plataforma instala `requirements.txt` y publica la URL.

**Enlace de la aplicación (desplegada):** `[PEGAR AQUÍ EL LINK]`

**Capturas de la aplicación:**
- Menú principal (3 casos): `[PEGAR CAPTURA]`
- Interfaz del Caso 2 (Información): `[PEGAR CAPTURA]`
- Formulario de predicción y resultado: `[PEGAR CAPTURA]`
- Métricas y gráficos del caso: `[PEGAR CAPTURA]`

---

**Verificación del cumplimiento:** `notebooks/wine_quality_analysis.ipynb`
(ejecutado sin errores), `results/wine_quality/metrics.csv`,
`cases/wine_quality/README.md` y `ejemplos_de_prediccion.txt` respaldan las
fases A y B; `app.py` + `cases/registry.py` + `cases/wine_quality/app.py`
respaldan la fase C.
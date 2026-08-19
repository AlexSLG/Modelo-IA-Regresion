# Ejercicio 3 — Diabetes: Documentación por Fases

**Problema:** predecir cuantitativamente la **progresión de la diabetes**
(`Y`) un año después del inicio, a partir de 10 variables basales del
paciente.
**Dataset:** `data/diabetes.tab.txt` · **442 registros × 11 columnas.**

---

## FASE A — Análisis Exploratorio y Preprocesamiento

### A.1 Carga de datos y limpieza

El dataset **no es un CSV con coma**: es un archivo de texto separado por
**tabulaciones** (`.tab.txt`). Se carga con
`pd.read_csv(..., sep="\t")`. Un `read_csv` normal fallaría al interpretar
las columnas. El encabezado es la primera línea:
`AGE SEX BMI BP S1 S2 S3 S4 S5 S6 Y`. No se descarta ninguna variable.

### A.2 Valores nulos y su tratamiento

- **Nulos:** 0 en todas las columnas.
- **Tratamiento:** de forma preventiva el pipeline incluye
  `SimpleImputer(strategy="median")` (no-op, por robustez).

### A.3 Outliers y su tratamiento

Detección por IQR: los valores extremos son **escasos en todas las
variables (máximo 2.04%)**.

**Tratamiento:** **se conservan.** Corresponden a pacientes reales con
valores basales extremos; eliminarlos sesgaría la muestra. Se documentan y
se mantienen.

### A.4 Análisis de multicolinealidad (correlación / VIF)

**Matriz de correlación (Pearson) con el objetivo:**

```
BMI    +0.586
S5     +0.566
BP     +0.441
S4     +0.430
S3     -0.395
```

`BMI` y `S5` son los predictores más correlacionados; `S3` correlaciona
negativamente (el HDL se asocia a menor progresión).

**VIF:**

```
AGE        1.49
SEX        1.19
BMI       55.29
BP        69.40
S1       577.07
S2       245.50
S3        43.75
S4        81.40
S5       277.10
S6        94.46
```

**Interpretación:** las medidas séricas S1-S6 están muy correlacionadas
entre sí (VIF de `S1` = 577, `S5` = 277, `S2` = 245). Consecuencia: los
coeficientes individuales son poco estables (p. ej. el signo negativo de
`S1` no debe leerse como un efecto causal aislado). La **predicción global
sigue siendo válida** y **no se eliminan variables**.

### A.5 Normalización / estandarización

Se usa **StandardScaler** sobre las 10 variables basales. Es necesario
porque las escalas difieren (S5 ~4 frente a S1 ~190). `SEX` (1 = hombre,
2 = mujer) se trata como **variable numérica binaria**: no se aplica
One-Hot Encoding (el caso no tiene categóricas con nombres).

### A.6 Resumen del preprocesador

```
ColumnTransformer
└── num: SimpleImputer(mediana) → StandardScaler
         (AGE, SEX, BMI, BP, S1, S2, S3, S4, S5, S6)
```

División **80% / 20%** con `random_state=42` (~353 entrenamiento / 89
prueba). Polinomial grado 2 → **66** características; grado 3 → **286**
características.

---

## FASE B — Modelamiento Estadístico y Comparativa

### B.1 Regresión Lineal Múltiple — cómo funciona

**Concepto:** modela la progresión como combinación lineal de las variables
basales:

```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

**Cómo se aplica aquí:** `LinearRegression` estima los coeficientes por
**mínimos cuadrados ordinarios (OLS)** sobre las 10 variables
estandarizadas. Intercepto 153.74. Coeficientes principales: `S5` (+35.16)
y `BMI` (+25.61) — los predictores con mayor correlación — y `SEX` (−11.51,
las mujeres del dataset presentan menor progresión estimada). Los
coeficientes son comparables entre sí por estar estandarizados, pero se
interpretan con cautela por el VIF alto.

### B.2 Regresión Polinomial — cómo funciona

**Concepto:** regresión lineal sobre características polinomiales:

```
ŷ = β₀ + β₁·x + β₂·x² + ... + βₙ·xⁿ
```

**Cómo se aplica aquí:** `PolynomialFeatures` genera potencias (x², x³) e
interacciones (xᵢ·xⱼ); luego `LinearRegression` aprende los coeficientes.
Grados evaluados: **2** (66 características) y **3** (286 características).
Con solo 442 registros (~353 de entrenamiento), el grado 3 tiene un riesgo
muy alto de sobreajuste.

### B.3 Entrenamiento y comparativa

Los tres modelos comparten el preprocesador y se entrenan con
`python -m cases.diabetes.train`. Métricas reales en prueba (test 20%):

| Modelo | MAE | MSE | RMSE | R² |
| ------ | --- | --- | ---- | -- |
| **Regresión Lineal** | 42.79 | 2,900.19 | 53.85 | **0.453** ← mejor |
| Polinomial grado 2 | 43.58 | 3,096.03 | 55.64 | 0.416 |
| Polinomial grado 3 | 162.32 | 80,980.76 | 284.57 | −14.28 |

Comparación train vs test (sobreajuste):

| Modelo | R² train | R² test |
| ------ | -------- | ------- |
| Lineal | 0.528 | **0.453** |
| Grado 2 | 0.606 | 0.416 |
| Grado 3 | 0.877 | **−14.28** |

**Selección del mejor modelo:** la **Regresión Lineal** (R² test 0.453,
RMSE 53.85). El **grado 3 sobreajusta de forma severa** (R² train 0.877 →
test −14.28): con 286 características memoriza el entrenamiento. Su
predicción sobre un paciente típico da **−7,458** (valor absurdo, fuera del
rango real 25–346), lo que evidencia que fuera de los datos el polinomio
"explota".

### B.4 Métricas utilizadas

- **MAE:** promedio de |real − predicho|.
- **MSE:** promedio de (real − predicho)². Penaliza errores grandes.
- **RMSE:** raíz del MSE, en las mismas unidades que `Y`.
- **R²:** proporción de varianza explicada; negativo = peor que predecir la
  media.

---

## FASE C — Desarrollo del Aplicativo Web Multi-Caso

### C.1 Arquitectura

```
data/diabetes.tab.txt → cases/diabetes/
├── config.py           # rutas, features, random_state
├── preprocessing.py    # carga con sep="\t" + preprocesador
├── train.py            # entrena los 3 pipelines y guarda con joblib
├── evaluate.py         # métricas (metrics.csv) y gráficos PNG
├── predict.py          # carga el .pkl y predice
├── app.py              # interfaz Streamlit (función render())
└── README.md

models/diabetes/*.pkl           # pipelines serializados
results/diabetes/               # metrics.csv, input_ranges.json, *.png
```

Igual que los demás casos: los `.pkl` se entrenan una vez y la app **solo
carga y predice**.

### C.2 Menú de navegación multi-caso

El caso se registra en `cases/registry.py` con botón "Ingresar al Caso 3".
El menú principal (`app.py`) permite seleccionar cualquiera de los 3 casos.
Dentro del caso: sidebar con **Información · Descriptores · Modelos ·
Métricas · Predicción** y botón "← Volver al menú principal".

### C.3 Interacción dinámica (predicción en tiempo real)

En **Predicción** el usuario:

1. Selecciona el **modelo** (Lineal / Polinomial grado 2 / grado 3).
2. Ingresa las 10 variables basales (edad, sexo, IMC, presión arterial y
   S1-S6) con controles limitados a los **rangos reales del dataset**
   (`input_ranges.json`), con la mediana como valor por defecto; el sexo se
   elige con `selectbox` (1 = hombre, 2 = mujer).
3. Presiona "Predecir"; la app carga el pipeline seleccionado y muestra la
   **progresión estimada de la diabetes** (`X.XX`) con un aviso de que es
   un ejercicio académico y **no un diagnóstico médico**.

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
- Interfaz del Caso 3 (Información): `[PEGAR CAPTURA]`
- Formulario de predicción y resultado: `[PEGAR CAPTURA]`
- Métricas y gráficos del caso: `[PEGAR CAPTURA]`

---

**Verificación del cumplimiento:** `notebooks/diabetes_analysis.ipynb`
(ejecutado sin errores), `results/diabetes/metrics.csv`,
`cases/diabetes/README.md` y `ejemplos_de_prediccion.txt` respaldan las
fases A y B; `app.py` + `cases/registry.py` + `cases/diabetes/app.py`
respaldan la fase C.
# Diabetes — Modelos de Regresión

**Caso 3** de la plataforma *Modelos de Inteligencia Artificial*.

Predicción cuantitativa de la **progresión de la diabetes** un año después
del inicio mediante **Regresión Lineal Múltiple** y **Regresión Polinomial**
a partir de variables basales de los pacientes.

---

## 1. Descripción del problema

El problema consiste en **predecir cuantitativamente la progresión de la
diabetes un año después del inicio** a partir de 10 variables basales del
paciente: edad, sexo, índice de masa corporal, presión arterial media y seis
medidas séricas. Es un problema de **regresión supervisada**: el modelo
aprende la relación entre los descriptores y una variable continua.

## 2. Objetivo

Predecir la progresión de la enfermedad (`Y`) y comparar el desempeño de:

- **Regresión Lineal Múltiple** (modelo base).
- **Regresión Polinomial** con grados 2 y 3.

## 3. Dataset

- **Nombre:** `data/diabetes.tab.txt` (dataset *Diabetes* de scikit-learn).
- **Formato:** archivo de texto separado por **tabulaciones** (`.tab.txt`),
  no es un CSV con comas; se lee con `sep="\t"`.
- **Registros:** 442 pacientes.
- **Variables:** 11 columnas (10 predictoras + 1 objetivo).
- **Variable objetivo:** `Y`.

## 4. Descriptores

| Descriptor | Descripción | Tipo | Uso |
| ---------- | ----------- | ---- | --- |
| `AGE` | Edad del paciente (años) | Numérica entera | Factor de riesgo basal |
| `SEX` | Sexo (1 = hombre, 2 = mujer) | Numérica binaria | Diferencia por sexo |
| `BMI` | Índice de masa corporal | Numérica continua | Obesidad (mayor correlación con Y) |
| `BP` | Presión arterial media | Numérica continua | Riesgo cardiovascular |
| `S1` | Medida sérica (colesterol sérico total, tc) | Numérica continua | Perfil lipídico |
| `S2` | Medida sérica (lipoproteínas de baja densidad, ldl) | Numérica continua | Perfil lipídico |
| `S3` | Medida sérica (lipoproteínas de alta densidad, hdl) | Numérica continua | Perfil lipídico (correlación negativa) |
| `S4` | Medida sérica (colesterol total / HDL, tch) | Numérica continua | Perfil lipídico |
| `S5` | Medida sérica (log triglicéridos séricos, ltg) | Numérica continua | Perfil lipídico |
| `S6` | Medida sérica (nivel de glucosa en sangre, glu) | Numérica continua | Perfil glucémico |

`SEX` (1/2) se trata como **variable numérica binaria**: no se aplica
One-Hot Encoding. **Ninguna variable fue descartada.**

## 5. Variable objetivo

`Y` — **progresión cuantitativa de la enfermedad un año después del inicio**.
Rango observado: 25 a 346, media 152.1, mediana 140. Es una variable
continua. El proyecto es **académico y predictivo**: la predicción no
constituye un diagnóstico médico.

## 6. Análisis exploratorio

- **Registros:** 442 · **Variables:** 11 (10 descriptores + objetivo).
- **Nulos:** 0 en todas las columnas.
- **Duplicados:** 0.
- **Escalas:** las variables tienen escalas muy diferentes (S5 ~4 frente a
  S1 ~190), lo que justifica la estandarización.
- **Outliers (IQR):** escasos en todas las variables (máx. 2.04%); son
  pacientes reales con valores extremos y **se conservan**.

## 7. Correlación con la variable objetivo

```
Variable  Correlación con Y
----------------------------
BMI        0.586
S5         0.566
BP         0.441
S4         0.430
S3        -0.395
```

`BMI` y `S5` son los predictores más correlacionados; `S3` correlaciona
negativamente.

## 8. Multicolinealidad (VIF)

```
Variable  VIF
--------------
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

Las medidas séricas S1-S6 están muy correlacionadas entre sí (VIF de S1 =
577, S5 = 277, S2 = 245). Consecuencia: los coeficientes individuales son
poco estables, pero la **predicción global es válida** y **no se eliminan
variables**.

## 9. Preprocesamiento

| Paso | Técnica | Justificación |
| ---- | ------- | ------------- |
| Carga | `pd.read_csv(..., sep="\t")` | El dataset es tabulado, no CSV con coma |
| Imputación | `SimpleImputer(strategy="median")` | No hay nulos; por robustez |
| Escalado | `StandardScaler` | Variables en escalas muy distintas |
| Train/Test | 80% / 20%, `random_state=42` | Sin data leakage (transformadores solo con train) |

Pipeline: `diabetes.tab.txt → imputación → StandardScaler → [PolynomialFeatures] → LinearRegression`.

## 10. Flujo del modelo

```
diabetes.tab.txt (tabulado)
 ↓
EDA (nulos, duplicados, outliers, correlación, VIF)
 ↓
Preprocesamiento: SimpleImputer(mediana) → StandardScaler (10 variables)
 ↓
[PolynomialFeatures grado 2/3 — solo modelos polinomiales]
 ↓
LinearRegression (mínimos cuadrados)
 ↓
Evaluación (MAE, MSE, RMSE, R² — train vs test)
 ↓
Guardado con joblib (models/diabetes/*.pkl)
 ↓
Streamlit carga el pipeline y predice (no reentrena)
```

## 11. Regresión Lineal Múltiple

**Concepto:** modela la progresión como combinación lineal de las variables
basales.

**Fórmula:**

```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

**Aplicación:** con las 10 variables basales estandarizadas.

## 12. Regresión Polinomial

**Concepto:** regresión lineal sobre características polinomiales (potencias
e interacciones).

**Fórmula (una variable):**

```
ŷ = β₀ + β₁·x + β₂·x² + ... + βₙ·xⁿ
```

**Grados utilizados:** 2 (66 características) y 3 (286 características). Con
solo 442 registros (~353 de entrenamiento), el grado 3 tiene un riesgo muy
alto de sobreajuste.

## 13. Modelos implementados

1. `LinearRegression` — `[preprocesador → LinearRegression]`.
2. `PolynomialFeatures(degree=2, include_bias=False)` + `LinearRegression`.
3. `PolynomialFeatures(degree=3, include_bias=False)` + `LinearRegression`.

Entrenados con `python -m cases.diabetes.train` y guardados con `joblib`.

## 14. Métricas

- **MAE:** promedio de |real − predicho|. Menor = mejor.
- **MSE:** promedio de (real − predicho)². Penaliza errores grandes.
- **RMSE:** raíz del MSE, en las mismas unidades que `Y`. Menor = mejor.
- **R²:** proporción de varianza explicada. Máximo 1; negativo implica un
  modelo peor que predecir la media.

## 15. Resultados

Tabla real obtenida en el entrenamiento (test 20%, 89 registros):

```
Modelo                  MAE      MSE       RMSE     R²
---------------------------------------------------------
Regresión Lineal        42.79    2,900.19  53.85    0.453   ← mejor
Polinomial grado 2      43.58    3,096.03  55.64    0.416
Polinomial grado 3     162.32   80,980.76 284.57  -14.28   ← sobreajuste
```

## 16. Sobreajuste (train vs test)

| Modelo | R² train | R² test | RMSE train | RMSE test |
| ------ | -------- | ------- | ---------- | --------- |
| Lineal | 0.528 | **0.453** | 53.56 | **53.85** |
| Grado 2 | 0.606 | 0.416 | 48.92 | 55.64 |
| Grado 3 | 0.877 | −14.28 | 27.38 | 284.57 |

- **Lineal:** brecha pequeña (0.528 → 0.453): buen ajuste sin sobreajuste.
- **Grado 2:** no mejora la generalización frente a la lineal.
- **Grado 3:** **sobreajuste severo** — R² train 0.877 pero test **−14.28**
  (RMSE 284.57). Memoriza el entrenamiento (286 características) y colapsa
  en prueba.

## 17. Selección del mejor modelo

**La Regresión Lineal es el mejor modelo** (R² test 0.453, RMSE 53.85). La
polinomial grado 2 aporta menos capacidad predictiva y el grado 3 falla por
sobreajuste.

## 18. Coeficientes

Coeficientes de la Regresión Lineal sobre variables **estandarizadas**
(intercepto 153.74):

```
Variable  Coeficiente
-----------------------
AGE         +1.75
SEX        -11.51
BMI        +25.61
BP         +16.83
S1         -44.45
S2         +24.64
S3          +7.68
S4         +13.14
S5         +35.16
S6          +2.35
```

- **Mayor peso positivo:** `S5` (+35.16) y `BMI` (+25.61) — consistentes con
  su alta correlación con la progresión.
- **Mayor peso negativo:** `S1` (−44.45) y `SEX` (−11.51). El efecto negativo
  de `S1` debe leerse con cautela por la multicolinealidad con el resto de
  medidas séricas.

## 19. Predicción

El usuario introduce las 10 variables basales en la aplicación Streamlit
(con rangos reales del dataset) y selecciona un modelo. La app carga el
pipeline guardado con `joblib` (no reentrena) y muestra la **progresión
estimada de la diabetes** en formato `X.XX`, con un aviso de que es un
ejercicio académico y no un diagnóstico médico.

## 20. Predicción de ejemplo

Paciente típico (medianas de cada variable):

```
AGE=50  SEX=1  BMI=25.7  BP=93.0  S1=186.0  S2=113.0
S3=48.0 S4=4.0 S5=4.62   S6=91.0

Modelo                     Predicción
--------------------------------------
Regresión Lineal            155.20
Polinomial grado 2           36.71
Polinomial grado 3        -7,458.74
```

El valor absurdo del grado 3 (−7,458) evidencia el sobreajuste: fuera del
rango de los datos, el polinomio explota.

## 21. Conclusiones

1. El dataset es **tabulado** (`.tab.txt`), 442 registros, sin nulos ni
   duplicados.
2. `BMI` y `S5` son los predictores más correlacionados con la progresión.
3. La multicolinealidad entre S1-S6 es severa (VIF > 200); se documenta y no
   se eliminan variables.
4. **La Regresión Lineal es el mejor modelo** (R² test 0.453).
5. La **polinomial grado 3 sobreajusta** de forma severa (R² test −14.28);
   el grado 2 no supera a la lineal.
6. El proyecto es académico y predictivo: las predicciones no constituyen un
   diagnóstico médico.

## 22. Preguntas para la exposición — Conceptos

**¿Por qué el dataset se llama `.tab.txt` y no `.csv`?** Porque sus columnas
se separan con **tabulaciones**, no con comas. Se lee con `pd.read_csv(..., sep="\t")`.
Un CSV normal fallaría al cargarlo.

**¿Qué significa que `Y` es la progresión de la enfermedad?** Es una medida
cuantitativa del avance de la diabetes un año después del inicio del estudio;
a mayor valor, mayor progresión. Es la variable continua a predecir.

**¿Qué es el VIF y qué implica un valor de 577?** El VIF mide cuánto se
infla la varianza de un coeficiente por la correlación con otras variables.
Un VIF de 577 en `S1` indica multicolinealidad extrema: no se puede aislar el
efecto individual de `S1`.

**¿Qué es el sobreajuste?** Cuando el modelo memoriza el entrenamiento (R²
train 0.877 en grado 3) pero falla en datos nuevos (R² test −14.28). Con 286
características y ~353 filas de entrenamiento, el modelo ajusta ruido.

## 23. Preguntas para la exposición — Resultados

**¿Por qué `BMI` y `S5` son los predictores más importantes?** Tienen la
mayor correlación con `Y` (0.586 y 0.566) y los mayores coeficientes
estandarizados positivos (+25.61 y +35.16): la obesidad y los triglicéridos
se asocian a mayor progresión.

**¿Por qué la regresión lineal es la mejor si solo explica 45% de la
varianza?** Porque las polinomiales no mejoran la generalización y el grado 3
colapsa. Un R² de 0.453 es razonable: la progresión de la diabetes depende de
factores no capturados por las 10 variables basales.

**¿Por qué `S1` tiene coeficiente negativo si se asocia a lípidos?** Por la
**multicolinealidad**: S1-S6 están altamente correlacionados; el modelo
reparte la información entre ellos y el signo individual pierde significado.

**¿Qué muestra el ejemplo de predicción con el grado 3?** Que un modelo
sobreajustado produce valores absurdos (−7,458) fuera del rango de los datos
(25-346). Demuestra por qué se prefiere el modelo lineal.

## 24. Preguntas para la exposición — Aplicación

**¿La aplicación reentrena el modelo?** No. Los pipelines se guardan con
`joblib` y la app carga los `.pkl` para predecir; es rápida y apta para
Streamlit Cloud.

**¿Qué se introduce para predecir?** Las 10 variables basales (edad, sexo,
IMC, presión arterial y S1-S6) mediante controles limitados a los rangos
reales del dataset, y se elige el modelo.

**¿El resultado es un diagnóstico?** No. Es una estimación estadística de un
ejercicio académico; la interfaz muestra un aviso de que no sustituye el
criterio médico.
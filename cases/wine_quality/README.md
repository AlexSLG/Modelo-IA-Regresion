# Wine Quality — Modelos de Regresión

**Caso 2** de la plataforma *Modelos de Inteligencia Artificial*.

Estimación de la calidad del vino mediante **Regresión Lineal Múltiple** y
**Regresión Polinomial** a partir de sus propiedades físico-químicas.

---

## 1. Descripción del problema

El problema consiste en **estimar la calidad de un vino** (puntuación de 0 a
10) a partir de sus propiedades físico-químicas: acidez, azúcar residual,
cloruros, sulfitos, alcohol, etc. Es un problema de **regresión supervisada**:
el modelo aprende la relación entre los descriptores y la calidad.

La aplicación se integra en la plataforma *Modelos de Inteligencia
Artificial* (Streamlit) junto con el Caso 1 (California Housing).

## 2. Objetivo

Estimar la calidad del vino (`quality`) y comparar el desempeño de:

- **Regresión Lineal Múltiple** (modelo base).
- **Regresión Polinomial** con grados 2 y 3.

## 3. Dataset

- **Nombre:** `data/WineQT.csv` (subconjunto de vino tinto del dataset *Wine Quality*).
- **Registros:** 1,143 vinos.
- **Variables:** 13 columnas en el CSV (11 propiedades físico-químicas +
  `quality` + `Id`). La columna `Id` es un índice secuencial y se descarta.
- **Variable objetivo:** `quality`.

## 4. Descriptores

| Descriptor | Descripción | Tipo | Uso |
| ---------- | ----------- | ---- | --- |
| `fixed acidity` | Acidez fija del vino (ácidos no volátiles), g/dm³ | Numérica continua | Acidez y sabor |
| `volatile acidity` | Acidez volátil (ácido acético), g/dm³ | Numérica continua | Valores altos se asocian a menor calidad |
| `citric acid` | Ácido cítrico, g/dm³ | Numérica continua | Frescura y acidez |
| `residual sugar` | Azúcar residual tras la fermentación, g/dm³ | Numérica continua | Dulzor |
| `chlorides` | Cloruros (sal), g/dm³ | Numérica continua | Sabor salado |
| `free sulfur dioxide` | Dióxido de azufre libre, mg/dm³ | Numérica continua | Conservante/antioxidante |
| `total sulfur dioxide` | Dióxido de azufre total, mg/dm³ | Numérica continua | Conservante total |
| `density` | Densidad del vino, g/cm³ | Numérica continua | Relacionada con alcohol y azúcar |
| `pH` | pH del vino | Numérica continua | Acidez real |
| `sulphates` | Sulfatos, g/dm³ | Numérica continua | Aroma y calidad percibida |
| `alcohol` | Contenido de alcohol, % vol | Numérica continua | Predictor más correlacionado |

## 5. Variable objetivo

`quality` — calidad del vino, puntuación entera de 0 a 10. En este dataset
toma valores de **3 a 8** (6 niveles). Aunque es **discreta**, el ejercicio
exige modelos de regresión, por lo que se trata como **variable numérica
continua**; el modelo produce predicciones decimales (p. ej. 5.72).

## 6. Análisis exploratorio

- **Registros:** 1,143 · **Variables:** 12 (11 descriptores + objetivo).
- **Estadísticos de `quality`:** media 5.66, mediana 6, desv. estándar 0.81,
  mínimo 3, máximo 8. Frecuencias: 3→6, 4→33, 5→483, 6→462, 7→143, 8→16.
- Las propiedades tienen **escalas muy diferentes** (densidad ~0.997 frente
  a SO₂ total hasta 289), lo que justifica la estandarización.

## 7. Valores nulos

No hay valores nulos en ninguna columna. De forma preventiva, el pipeline
incluye `SimpleImputer(median)`, que no modifica los datos (no-op).

## 8. Duplicados

No existen registros duplicados (0 de 1,143).

## 9. Outliers

Por IQR, las variables con más extremos son `residual sugar` (9.62%),
`chlorides` (6.74%), `fixed acidity` (3.85%), `sulphates` (3.76%) y
`total sulfur dioxide` (3.50%). Corresponden a **vinos reales** con
características atípicas (vinos dulces, muy salados o muy ácidos). **Se
conservan** por ser observaciones válidas; el modelo usa todos los datos.

## 10. Correlación

- Con `quality`: `alcohol` (+0.485), `volatile acidity` (−0.407),
  `sulphates` (+0.258), `citric acid` (+0.241).
- Entre predictores (multicolinealidad): `fixed acidity`–`citric acid`
  (0.67), `fixed acidity`–`density` (0.68), `fixed acidity`–`pH` (−0.69),
  `free SO₂`–`total SO₂` (0.66).

## 11. VIF

```
Variable                  VIF
--------------------------------
fixed acidity             74.55
volatile acidity          17.04
citric acid                9.22
residual sugar             5.01
chlorides                  6.72
free sulfur dioxide        6.30
total sulfur dioxide       6.22
density                 1435.10
pH                      1043.34
sulphates                 21.51
alcohol                  123.96
```

Los VIF de `density`, `pH`, `alcohol`, `fixed acidity`, `sulphates` y
`volatile acidity` superan 10 (multicolinealidad severa). Consecuencia: los
coeficientes individuales son poco estables y se interpretan con cautela,
pero la predicción global es válida. **No se eliminan variables.**

## 12. Preprocesamiento

| Paso | Técnica | Justificación |
| ---- | ------- | ------------- |
| Limpieza | Descartar `Id` | Índice secuencial, no propiedad del vino |
| Imputación | `SimpleImputer(median)` | No hay nulos; por robustez |
| Escalado | `StandardScaler` | Escalas muy distintas entre propiedades |
| Train/Test | 80/20, `random_state=42` | Sin data leakage (transformadores ajustados solo con train) |

Pipeline: `Dataset → limpieza → imputación → StandardScaler → [PolynomialFeatures] → LinearRegression`.

## 13. Flujo del modelo

```
WineQT.csv
 ↓
Limpieza: descartar Id (índice secuencial)
 ↓
EDA (nulos, duplicados, outliers, correlación, VIF)
 ↓
Preprocesamiento: SimpleImputer(mediana) → StandardScaler (11 propiedades)
 ↓
[PolynomialFeatures grado 2/3 — solo modelos polinomiales]
 ↓
LinearRegression (mínimos cuadrados)
 ↓
Evaluación (MAE, MSE, RMSE, R² — train vs test)
 ↓
Guardado con joblib (models/wine_quality/*.pkl)
 ↓
Streamlit carga el pipeline y predice (no reentrena)
```

## 14. Regresión Lineal Múltiple

**Concepto:** modela la calidad como combinación lineal de las propiedades.

**Funcionamiento:** estima los coeficientes β por mínimos cuadrados
ordinarios minimizando el error cuadrático.

**Fórmula:**

```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

**Aplicación:** con las 11 propiedades estandarizadas.

## 15. Regresión Polinomial

**Concepto:** regresión lineal sobre características polinomiales
(potencias e interacciones).

**Funcionamiento:** `PolynomialFeatures` genera nuevas columnas (x², x³,
xᵢ·xⱼ) y luego `LinearRegression` aprende los coeficientes.

**Fórmula (una variable):**

```
ŷ = β₀ + β₁·x + β₂·x² + ... + βₙ·xⁿ
```

**Grados utilizados:** 2 (78 características) y 3 (364 características). Con
solo 1,143 registros, el grado 3 tiene alto riesgo de sobreajuste.

## 16. Modelos

1. `LinearRegression` — `[preprocesador → LinearRegression]`.
2. `PolynomialFeatures(degree=2, include_bias=False)` + `LinearRegression`.
3. `PolynomialFeatures(degree=3, include_bias=False)` + `LinearRegression`.

Entrenados con `python -m cases.wine_quality.train` y guardados con `joblib`.

## 17. Métricas

- **MAE:** promedio de |real − predicho|. Menor = mejor.
- **MSE:** promedio de (real − predicho)². Penaliza errores grandes.
- **RMSE:** raíz del MSE, en unidades de calidad. Menor = mejor.
- **R²:** proporción de varianza explicada. Mayor (máx. 1) = mejor; negativo
  implica un modelo peor que predecir la media.

## 18. Resultados

Tabla real obtenida en el entrenamiento (test 20%, 229 registros):

```
Modelo                  MAE    MSE     RMSE    R²
---------------------------------------------------
Regresión Lineal        0.477  0.380   0.616   0.317
Polinomial grado 2      0.487  0.400   0.633   0.281
Polinomial grado 3      0.866  4.156   2.039  -6.469
```

## 19. Comparación

| Modelo | R² train | R² test | RMSE train | RMSE test |
| ------ | -------- | ------- | ---------- | --------- |
| Lineal | 0.382 | **0.317** | 0.644 | **0.616** |
| Grado 2 | 0.459 | 0.281 | 0.603 | 0.633 |
| Grado 3 | 0.715 | −6.469 | 0.438 | 2.039 |

El **modelo lineal** obtiene el mejor desempeño en prueba.

## 20. Sobreajuste

- **Lineal:** R² train 0.382 → test 0.317. Brecha pequeña: **buen ajuste**
  sin sobreajuste (la calidad es difícil de predecir con solo variables
  químicas, de ahí un R² moderado).
- **Grado 2:** R² train 0.459 → test 0.281. La expansión no mejora la
  generalización.
- **Grado 3:** R² train 0.715 → test **−6.469**. **Sobreajuste severo**: el
  modelo memoriza el entrenamiento (364 características) y colapsa en prueba.

## 21. Coeficientes

Coeficientes de la Regresión Lineal sobre variables **estandarizadas**
(intercepto 5.656):

```
Variable                 Coeficiente
--------------------------------------
fixed acidity            +0.087
volatile acidity         -0.239
citric acid              -0.066
residual sugar           +0.005
chlorides                -0.086
free sulfur dioxide      +0.019
total sulfur dioxide     -0.073
density                  -0.059
pH                       -0.038
sulphates                +0.162
alcohol                  +0.286
```

- **Mayor peso positivo:** `alcohol` (+0.286) — más alcohol, mayor calidad.
- **Mayor peso negativo:** `volatile acidity` (−0.239) — vinos más volátiles
  se asocian a menor calidad.
- Al estar estandarizadas, los coeficientes son comparables entre sí. Se
  interpretan con cautela por la multicolinealidad (VIF alto).

## 22. Predicción

El usuario introduce las 11 propiedades en la aplicación Streamlit (con
rangos reales del dataset) y selecciona un modelo. La app carga el pipeline
guardado con `joblib` (no reentrena) y muestra la calidad estimada en
formato `X.XX / 10`, con el valor continuo y su redondeo opcional.

## 23. Conclusiones

1. `alcohol` y `volatile acidity` son los descriptores con mayor influencia.
2. La multicolinealidad (VIF de density=1435, pH=1043) es severa; se documenta
   y no se eliminan variables.
3. Los outliers son variaciones químicas reales; se conservan.
4. **La Regresión Lineal es el mejor modelo** (R² 0.317 en prueba).
5. La **polinomial grado 3 sobreajusta** (R² test −6.47); el grado 2 no aporta
   frente a la lineal.
6. La calidad es discreta pero se modela como continua por el planteamiento
   de regresión del ejercicio.

## 24. Preguntas para la exposición — Conceptos

**¿Qué es la calidad del vino y por qué se modela como continua?** Es una
puntuación entera de 0 a 10; en este dataset toma valores 3-8. El ejercicio
exige modelos de regresión, por lo que se trata como variable continua y el
modelo devuelve predicciones decimales (p. ej. 5.72).

**¿Qué es el R² y qué significa un valor negativo?** El R² mide la proporción
de varianza explicada. Un valor negativo (como −6.469 del grado 3) significa
que el modelo es peor que simplemente predecir la media: falla totalmente en
datos no vistos.

**¿Qué es el sobreajuste?** Cuando el modelo memoriza el conjunto de
entrenamiento (R² train alto) pero no generaliza a datos nuevos (R² test muy
bajo). Con 364 características del grado 3 y solo 1,143 registros, ocurre de
forma severa.

**¿Qué es el VIF?** El factor de inflación de la varianza, que mide la
multicolinealidad de un predictor. Valores > 10 (aquí hasta 1,435 en
`density`) indican que la variable está muy correlacionada con otras.

## 25. Preguntas para la exposición — Resultados

**¿Por qué `alcohol` es el predictor más importante?** Tiene la mayor
correlación positiva con la calidad (r = 0.485) y el mayor coeficiente
estandarizado (+0.286): más alcohol suele implicar vinos mejor valorados.

**¿Por qué `volatile acidity` reduce la calidad?** El ácido acético produce
aromas desagradables; tiene correlación negativa (−0.407) y coeficiente
negativo (−0.239).

**¿Por qué la regresión lineal supera a la polinomial?** La relación entre
las propiedades químicas y la calidad es esencialmente lineal en los datos;
las expansiones polinomiales (78 y 364 características) añaden complejidad
sin mejorar y el grado 3 colapsa por sobreajuste.

**¿Por qué no se eliminan variables con VIF alto?** La multicolinealidad
afecta la interpretación de coeficientes individuales, pero la predicción
global del modelo sigue siendo válida; eliminar variables reduciría
información.

## 26. Preguntas para la exposición — Aplicación

**¿La aplicación entrena en tiempo real?** No. Los pipelines se guardan con
`joblib` y la app carga los `.pkl` para predecir, lo que la hace rápida y
apta para la nube.

**¿Cómo se introducen los datos para predecir?** En la sección Predicción se
ingresan las 11 propiedades físico-químicas con controles limitados a los
rangos reales del dataset y se selecciona el modelo.

**¿Qué muestra el resultado?** La calidad estimada en formato `X.XX / 10`,
junto con el modelo seleccionado y los valores ingresados.
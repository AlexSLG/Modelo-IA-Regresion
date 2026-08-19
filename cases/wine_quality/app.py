"""Interfaz Streamlit del caso Wine Quality.

Contiene la función ``render()`` que invoca el menú principal. Las
secciones disponibles son: Información, Descriptores, Modelos,
Métricas y Predicción. Los modelos se cargan de disco (nunca se
reentrenan) y el formulario usa rangos reales del dataset.
"""

import json

import pandas as pd
import streamlit as st

from .config import MODEL_KEYS, RESULTS_DIR
from .predict import predict
from .preprocessing import load_data

DESCRIPTORS = [
    (
        "fixed acidity",
        "Acidez fija del vino (ácidos no volátiles), en g/dm³.",
        "Numérica continua",
        "Contribuye a la acidez y al sabor.",
    ),
    (
        "volatile acidity",
        "Acidez volátil (principalmente ácido acético), en g/dm³.",
        "Numérica continua",
        "Un valor alto se asocia con vinos de menor calidad.",
    ),
    (
        "citric acid",
        "Cantidad de ácido cítrico, en g/dm³.",
        "Numérica continua",
        "Aporta frescura y acidez.",
    ),
    (
        "residual sugar",
        "Azúcar residual tras la fermentación, en g/dm³.",
        "Numérica continua",
        "Dulzor del vino.",
    ),
    (
        "chlorides",
        "Cantidad de cloruros (sal), en g/dm³.",
        "Numérica continua",
        "Influye en el sabor salado.",
    ),
    (
        "free sulfur dioxide",
        "Dióxido de azufre libre, en mg/dm³.",
        "Numérica continua",
        "Conservante y antioxidante.",
    ),
    (
        "total sulfur dioxide",
        "Dióxido de azufre total, en mg/dm³.",
        "Numérica continua",
        "Cantidad global de conservante.",
    ),
    (
        "density",
        "Densidad del vino, en g/cm³.",
        "Numérica continua",
        "Relacionada con el contenido de alcohol y azúcar.",
    ),
    (
        "pH",
        "pH del vino (acidez real).",
        "Numérica continua",
        "Indica cuán ácido o básico es el vino.",
    ),
    (
        "sulphates",
        "Cantidad de sulfatos, en g/dm³.",
        "Numérica continua",
        "Influencia en el aroma y la calidad percibida.",
    ),
    (
        "alcohol",
        "Contenido de alcohol, en % vol.",
        "Numérica continua",
        "Uno de los predictores más correlacionados con la calidad.",
    ),
]


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """Carga la tabla de métricas guardada durante el entrenamiento."""
    return pd.read_csv(RESULTS_DIR / "metrics.csv")


@st.cache_data(show_spinner=False)
def load_ranges() -> dict:
    """Carga los rangos reales del dataset para el formulario."""
    with open(RESULTS_DIR / "input_ranges.json", encoding="utf-8") as f:
        return json.load(f)


def go_home() -> None:
    """Vuelve al menú principal."""
    st.session_state["page"] = "home"
    st.rerun()


def render_informacion() -> None:
    st.header("Información del problema")
    st.write(
        "El objetivo es **estimar la calidad de un vino** (puntuación "
        "0-10) a partir de sus propiedades físico-químicas — acidez, "
        "azúcar residual, cloruros, alcohol, etc. — utilizando modelos "
        "de regresión."
    )
    df = load_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{df.shape[0]:,}")
    c2.metric("Variables", df.shape[1])
    c3.metric("Variable objetivo", "quality")
    c4.metric("Rango de calidad", f"{int(df.quality.min())} – {int(df.quality.max())}")

    st.subheader("Modelos implementados")
    st.write(
        "- **Regresión Lineal Múltiple**: relación lineal entre las "
        "propiedades del vino y su calidad.\n"
        "- **Regresión Polinomial**: añade términos de grado 2 y 3 para "
        "capturar relaciones no lineales."
    )
    st.subheader("Sobre la variable objetivo")
    st.write(
        "La calidad es una variable **discreta** (valores enteros de 3 a 8 "
        "en este dataset). Por exigencia del ejercicio se trata como una "
        "variable numérica continua mediante regresión; el modelo produce "
        "una predicción decimal que se interpreta en la escala 0-10."
    )


def render_descriptores() -> None:
    st.header("Descriptores")
    st.write(
        "Propiedades físico-químicas utilizadas por el modelo. Todas se "
        "mantienen; la columna `Id` (índice del dataset) se descartó por no "
        "ser una propiedad del vino."
    )
    st.table(pd.DataFrame(DESCRIPTORS, columns=["Variable", "Descripción", "Tipo", "Función"]))
    st.subheader("Variable objetivo")
    st.markdown(
        "**`quality`** — Calidad del vino (puntuación de 0 a 10, en este "
        "dataset de 3 a 8). Es la variable que se desea estimar."
    )


def render_modelos() -> None:
    st.header("Modelos")
    st.subheader("Regresión Lineal Múltiple")
    st.write(
        "Modela la calidad como una combinación lineal de las propiedades "
        "del vino: `y = b0 + b1*x1 + ... + bn*xn`. Es el modelo base e "
        "interpretable; no captura relaciones no lineales."
    )
    st.subheader("Regresión Polinomial")
    st.write(
        "Extiende la regresión lineal añadiendo potencias e interacciones "
        "de las propiedades mediante `PolynomialFeatures`. Se evaluaron los "
        "grados **2** (78 características) y **3** (364 características). "
        "Con solo 1,143 registros, el grado 3 presenta un alto riesgo de "
        "sobreajuste, que se compara con las métricas de entrenamiento y "
        "prueba."
    )
    st.subheader("Pipeline común")
    st.code(
        "WineQT.csv\n"
        "  -> descartar columna Id (índice)\n"
        "  -> SimpleImputer(mediana)          # no hay nulos; por robustez\n"
        "  -> StandardScaler                  # estandariza las propiedades\n"
        "  -> [PolynomialFeatures (grado 2/3)]\n"
        "  -> LinearRegression\n"
        "  -> Predicción de calidad",
        language="text",
    )


def render_metricas() -> None:
    st.header("Métricas")
    metrics = load_metrics()
    st.subheader("Tabla comparativa (test)")
    show = metrics[["Modelo", "MAE_test", "MSE_test", "RMSE_test", "R2_test"]]
    show.columns = ["Modelo", "MAE", "MSE", "RMSE", "R²"]
    st.dataframe(show.round(3), use_container_width=True)

    st.subheader("Sobreajuste (train vs test)")
    show2 = metrics[["Modelo", "R2_train", "R2_test", "RMSE_train", "RMSE_test"]]
    show2.columns = ["Modelo", "R² train", "R² test", "RMSE train", "RMSE test"]
    st.dataframe(show2.round(3), use_container_width=True)
    best = metrics.loc[metrics["R2_test"].idxmax(), "Modelo"]
    st.write(
        f"El modelo con mejor rendimiento en prueba es **{MODEL_KEYS[best]}** "
        "(R² test más alto y RMSE test más bajo)."
    )

    st.subheader("Gráficos")
    st.image(str(RESULTS_DIR / "metrics_comparison.png"), caption="Comparación de RMSE y R²")
    st.image(str(RESULTS_DIR / "correlation_matrix.png"), caption="Matriz de correlación")
    st.image(str(RESULTS_DIR / "target_distribution.png"), caption="Distribución de la calidad")

    model_plot = st.selectbox(
        "Gráficos por modelo",
        list(MODEL_KEYS.keys()),
        format_func=lambda k: MODEL_KEYS[k],
    )
    st.image(str(RESULTS_DIR / f"actual_vs_predicted_{model_plot}.png"),
             caption=f"Calidad real vs predicha — {MODEL_KEYS[model_plot]}")
    st.image(str(RESULTS_DIR / f"residuals_{model_plot}.png"),
             caption=f"Residuos — {MODEL_KEYS[model_plot]}")


def render_prediccion() -> None:
    st.header("Predicción")
    ranges = load_ranges()
    numeric = ranges["numeric"]

    model_key = st.selectbox(
        "Modelo a utilizar",
        list(MODEL_KEYS.keys()),
        format_func=lambda k: MODEL_KEYS[k],
    )

    st.markdown("##### Características del vino")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            fixed_acidity = st.slider(
                "Acidez fija (g/dm³)", min_value=numeric["fixed acidity"]["min"],
                max_value=numeric["fixed acidity"]["max"],
                value=numeric["fixed acidity"]["median"], step=0.1, format="%.2f")
            volatile_acidity = st.slider(
                "Acidez volátil (g/dm³)", min_value=numeric["volatile acidity"]["min"],
                max_value=numeric["volatile acidity"]["max"],
                value=numeric["volatile acidity"]["median"], step=0.01, format="%.2f")
            citric_acid = st.slider(
                "Ácido cítrico (g/dm³)", min_value=numeric["citric acid"]["min"],
                max_value=numeric["citric acid"]["max"],
                value=numeric["citric acid"]["median"], step=0.01, format="%.2f")
            residual_sugar = st.slider(
                "Azúcar residual (g/dm³)", min_value=numeric["residual sugar"]["min"],
                max_value=numeric["residual sugar"]["max"],
                value=numeric["residual sugar"]["median"], step=0.1, format="%.2f")
            chlorides = st.slider(
                "Cloruros (g/dm³)", min_value=numeric["chlorides"]["min"],
                max_value=numeric["chlorides"]["max"],
                value=numeric["chlorides"]["median"], step=0.001, format="%.3f")
            free_so2 = st.slider(
                "Dióxido de azufre libre (mg/dm³)",
                min_value=numeric["free sulfur dioxide"]["min"],
                max_value=numeric["free sulfur dioxide"]["max"],
                value=numeric["free sulfur dioxide"]["median"], step=1.0, format="%.0f")
        with c2:
            total_so2 = st.slider(
                "Dióxido de azufre total (mg/dm³)",
                min_value=numeric["total sulfur dioxide"]["min"],
                max_value=numeric["total sulfur dioxide"]["max"],
                value=numeric["total sulfur dioxide"]["median"], step=1.0, format="%.0f")
            density = st.slider(
                "Densidad (g/cm³)", min_value=numeric["density"]["min"],
                max_value=numeric["density"]["max"],
                value=numeric["density"]["median"], step=0.0001, format="%.4f")
            ph = st.slider(
                "pH", min_value=numeric["pH"]["min"],
                max_value=numeric["pH"]["max"],
                value=numeric["pH"]["median"], step=0.01, format="%.2f")
            sulphates = st.slider(
                "Sulfatos (g/dm³)", min_value=numeric["sulphates"]["min"],
                max_value=numeric["sulphates"]["max"],
                value=numeric["sulphates"]["median"], step=0.01, format="%.2f")
            alcohol = st.slider(
                "Alcohol (% vol)", min_value=numeric["alcohol"]["min"],
                max_value=numeric["alcohol"]["max"],
                value=numeric["alcohol"]["median"], step=0.1, format="%.1f")
        submitted = st.form_submit_button("Predecir calidad")

    if submitted:
        features = {
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_so2,
            "total sulfur dioxide": total_so2,
            "density": density,
            "pH": ph,
            "sulphates": sulphates,
            "alcohol": alcohol,
        }
        with st.spinner("Calculando predicción..."):
            value = predict(model_key, features)

        st.divider()
        st.markdown("### Resultado")
        st.markdown(
            f"#### Calidad estimada del vino\n\n"
            f"# {value:.2f} / 10",
            help="Predicción continua del modelo sobre la escala de calidad 0-10.",
        )
        st.caption(
            f"Valor continuo predicho por {MODEL_KEYS[model_key]}. "
            f"Redondeo: ≈ {round(value)}. "
            "El modelo de regresión genera una predicción numérica continua; "
            "la calidad del dataset original es discreta (3-8)."
        )
        st.write("**Modelo seleccionado:**", MODEL_KEYS[model_key])
        st.write("**Características introducidas:**")
        st.json(features)


def render() -> None:
    """Punto de entrada de la interfaz del caso (llamado por el menú)."""
    st.sidebar.title("Caso 2 — Wine Quality")
    if st.sidebar.button("← Volver al menú principal", use_container_width=True):
        go_home()
    section = st.sidebar.radio(
        "Secciones",
        ["Información", "Descriptores", "Modelos", "Métricas", "Predicción"],
    )

    st.title("Wine Quality")
    st.caption("Estimación de la calidad del vino a partir de sus propiedades físico-químicas.")

    if section == "Información":
        render_informacion()
    elif section == "Descriptores":
        render_descriptores()
    elif section == "Modelos":
        render_modelos()
    elif section == "Métricas":
        render_metricas()
    elif section == "Predicción":
        render_prediccion()
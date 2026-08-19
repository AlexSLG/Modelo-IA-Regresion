"""Interfaz Streamlit del caso California Housing.

Contiene la función ``render()`` que invoca el menú principal. Las
secciones disponibles son: Información, Descriptores, Modelos,
Métricas y Predicción. Los modelos se cargan de disco (nunca se
reentrenan) y el formulario usa rangos reales del dataset.
"""

import json

import pandas as pd
import streamlit as st

from .config import MODELS_DIR, MODEL_KEYS, RESULTS_DIR
from .predict import predict
from .preprocessing import load_data

DESCRIPTORS = [
    (
        "longitude",
        "Longitud geográfica del distrito.",
        "Numérica (continua)",
        "Ubicación geográfica (interacciona con latitude).",
    ),
    (
        "latitude",
        "Latitud geográfica del distrito.",
        "Numérica (continua)",
        "Ubicación geográfica.",
    ),
    (
        "housing_median_age",
        "Edad mediana de las viviendas del distrito (años).",
        "Numérica (entera)",
        "Antigüedad del parque inmobiliario.",
    ),
    (
        "total_rooms",
        "Número total de habitaciones del distrito.",
        "Numérica (entera)",
        "Tamaño del parque inmobiliario.",
    ),
    (
        "total_bedrooms",
        "Número total de dormitorios del distrito (contiene nulos).",
        "Numérica (entera)",
        "Tamaño del parque inmobiliario.",
    ),
    (
        "population",
        "Población total del distrito.",
        "Numérica (entera)",
        "Tamaño demográfico del distrito.",
    ),
    (
        "households",
        "Número de hogares del distrito.",
        "Numérica (entera)",
        "Número de viviendas ocupadas.",
    ),
    (
        "median_income",
        "Ingreso medio de los habitantes del distrito.",
        "Numérica (continua)",
        "Poder adquisitivo (predictor más correlacionado con el objetivo).",
    ),
    (
        "ocean_proximity",
        "Proximidad al océano (categorías: <1H OCEAN, INLAND, ISLAND, "
        "NEAR BAY, NEAR OCEAN).",
        "Categórica",
        "Zona geográfica (se codifica con One-Hot Encoding).",
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
        "El objetivo es predecir el **valor medio de una vivienda** (en USD) "
        "de un distrito de California a partir de características "
        "geográficas, demográficas y económicas."
    )
    df = load_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{df.shape[0]:,}")
    c2.metric("Variables", df.shape[1])
    c3.metric("Variable objetivo", "median_house_value")
    c4.metric("Rango de valores (USD)", f"{df.median_house_value.min():,.0f} – {df.median_house_value.max():,.0f}")

    st.subheader("Modelos implementados")
    st.write(
        "- **Regresión Lineal Múltiple**: relación lineal entre los "
        "descriptores y el valor de la vivienda.\n"
        "- **Regresión Polinomial**: añade términos de grado 2 y 3 para "
        "capturar relaciones no lineales."
    )
    st.subheader("Descripción del dataset")
    st.write(
        "El dataset contiene 20,640 distritos censales de California. La "
        "única variable categórica es `ocean_proximity` (5 categorías) y "
        "solo `total_bedrooms` presenta valores nulos (~1%), que se "
        "imputan con la mediana. Los modelos se evaluaron con 80% de "
        "entrenamiento y 20% de prueba."
    )


def render_descriptores() -> None:
    st.header("Descriptores")
    st.write(
        "Variables utilizadas por el modelo después del análisis y "
        "preprocesamiento. Todas se mantienen: ninguna fue descartada."
    )
    st.table(pd.DataFrame(DESCRIPTORS, columns=["Variable", "Descripción", "Tipo", "Función"]))
    st.subheader("Variable objetivo")
    st.markdown(
        "**`median_house_value`** — Valor medio de la vivienda (en USD) "
        "en el distrito. Es la variable que se desea predecir."
    )


def render_modelos() -> None:
    st.header("Modelos")
    st.subheader("Regresión Lineal Múltiple")
    st.write(
        "Modela la variable objetivo como una combinación lineal de los "
        "descriptores: `y = b0 + b1*x1 + ... + bn*xn`. Es el modelo base y "
        "permite interpretar el efecto de cada variable, aunque no captura "
        "relaciones no lineales."
    )
    st.subheader("Regresión Polinomial")
    st.write(
        "Extiende la regresión lineal añadiendo potencias e interacciones "
        "de las variables mediante `PolynomialFeatures`. Se evaluaron los "
        "grados **2 y 3**. Los grados altos pueden provocar sobreajuste, "
        "por lo que se comparan con la métrica R² de entrenamiento y prueba."
    )
    st.subheader("Pipeline común")
    st.code(
        "Datos\n"
        "  -> SimpleImputer(mediana)          # imputa nulos de total_bedrooms\n"
        "  -> OneHotEncoder                   # codifica ocean_proximity\n"
        "  -> StandardScaler                  # estandariza las numéricas\n"
        "  -> [PolynomialFeatures (grado 2/3)]\n"
        "  -> LinearRegression\n"
        "  -> Predicción",
        language="text",
    )


def render_metricas() -> None:
    st.header("Métricas")
    metrics = load_metrics()
    st.subheader("Tabla comparativa (test)")
    show = metrics[["Modelo", "MAE_test", "MSE_test", "RMSE_test", "R2_test"]]
    show.columns = ["Modelo", "MAE", "MSE", "RMSE", "R²"]
    st.dataframe(show.round(2), use_container_width=True)

    st.subheader("Sobreajuste (train vs test)")
    show2 = metrics[["Modelo", "R2_train", "R2_test", "RMSE_train", "RMSE_test"]]
    show2.columns = ["Modelo", "R² train", "R² test", "RMSE train", "RMSE test"]
    st.dataframe(show2.round(2), use_container_width=True)
    best = metrics.loc[metrics["R2_test"].idxmax(), "Modelo"]
    st.write(
        f"El modelo con mejor rendimiento en prueba es **{MODEL_KEYS[best]}** "
        "(R² test más alto y RMSE test más bajo)."
    )

    st.subheader("Gráficos")
    st.image(str(RESULTS_DIR / "metrics_comparison.png"), caption="Comparación de RMSE y R²")
    st.image(str(RESULTS_DIR / "correlation_matrix.png"), caption="Matriz de correlación")
    st.image(str(RESULTS_DIR / "target_distribution.png"), caption="Distribución de la variable objetivo")

    model_plot = st.selectbox(
        "Gráficos por modelo",
        list(MODEL_KEYS.keys()),
        format_func=lambda k: MODEL_KEYS[k],
    )
    st.image(str(RESULTS_DIR / f"actual_vs_predicted_{model_plot}.png"),
             caption=f"Reales vs predichos — {MODEL_KEYS[model_plot]}")
    st.image(str(RESULTS_DIR / f"residuals_{model_plot}.png"),
             caption=f"Residuos — {MODEL_KEYS[model_plot]}")


def render_prediccion() -> None:
    st.header("Predicción")
    ranges = load_ranges()
    numeric = ranges["numeric"]
    categories = ranges["categorical"]["ocean_proximity"]

    model_key = st.selectbox(
        "Modelo a utilizar",
        list(MODEL_KEYS.keys()),
        format_func=lambda k: MODEL_KEYS[k],
    )

    st.markdown("##### Características del distrito")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            longitude = st.number_input(
                "Longitude", min_value=numeric["longitude"]["min"],
                max_value=numeric["longitude"]["max"],
                value=numeric["longitude"]["median"], step=0.01, format="%.2f")
            latitude = st.number_input(
                "Latitude", min_value=numeric["latitude"]["min"],
                max_value=numeric["latitude"]["max"],
                value=numeric["latitude"]["median"], step=0.01, format="%.2f")
            housing_median_age = st.number_input(
                "Housing median age", min_value=int(numeric["housing_median_age"]["min"]),
                max_value=int(numeric["housing_median_age"]["max"]),
                value=int(numeric["housing_median_age"]["median"]), step=1)
            total_rooms = st.number_input(
                "Total rooms", min_value=int(numeric["total_rooms"]["min"]),
                max_value=int(numeric["total_rooms"]["max"]),
                value=int(numeric["total_rooms"]["median"]), step=1)
            total_bedrooms = st.number_input(
                "Total bedrooms", min_value=int(numeric["total_bedrooms"]["min"]),
                max_value=int(numeric["total_bedrooms"]["max"]),
                value=int(numeric["total_bedrooms"]["median"]), step=1)
        with c2:
            population = st.number_input(
                "Population", min_value=int(numeric["population"]["min"]),
                max_value=int(numeric["population"]["max"]),
                value=int(numeric["population"]["median"]), step=1)
            households = st.number_input(
                "Households", min_value=int(numeric["households"]["min"]),
                max_value=int(numeric["households"]["max"]),
                value=int(numeric["households"]["median"]), step=1)
            median_income = st.number_input(
                "Median income", min_value=float(numeric["median_income"]["min"]),
                max_value=float(numeric["median_income"]["max"]),
                value=float(numeric["median_income"]["median"]), step=0.01, format="%.2f")
            ocean_proximity = st.selectbox("Ocean proximity", categories)
        submitted = st.form_submit_button("Predecir")

    if submitted:
        features = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        }
        with st.spinner("Calculando predicción..."):
            value = predict(model_key, features)

        st.divider()
        st.markdown("### Resultado")
        st.markdown(
            f"#### Valor estimado de la vivienda\n\n"
            f"# $ {value:,.0f}",
            help="Predicción del modelo sobre el valor medio de la vivienda.",
        )
        st.write(f"**Modelo seleccionado:** {MODEL_KEYS[model_key]}")
        st.write("**Características introducidas:**")
        st.json(features)
        st.caption(
            "La predicción se calcula aplicando el pipeline entrenado "
            "(imputación, codificación y estandarización) sobre los datos "
            "ingresados. El valor es una estimación del valor medio del "
            "distrito y puede superar el máximo observado en el dataset "
            "en zonas de alto ingreso."
        )


def render() -> None:
    """Punto de entrada de la interfaz del caso (llamado por el menú)."""
    st.sidebar.title("Caso 1 — California Housing")
    if st.sidebar.button("← Volver al menú principal", use_container_width=True):
        go_home()
    section = st.sidebar.radio(
        "Secciones",
        ["Información", "Descriptores", "Modelos", "Métricas", "Predicción"],
    )

    st.title("California Housing")
    st.caption("Predicción del valor medio de viviendas en distritos de California.")

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
"""Interfaz Streamlit del caso Diabetes.

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
    ("AGE", "Edad (años).", "Numérica entera", "Factor de riesgo basal."),
    ("SEX", "Sexo (1 = hombre, 2 = mujer).", "Numérica binaria", "Diferencia por sexo."),
    ("BMI", "Índice de masa corporal.", "Numérica continua", "Obesidad (mayor correlación con Y)."),
    ("BP", "Presión arterial media.", "Numérica continua", "Riesgo cardiovascular."),
    ("S1", "Medida sérica S1 (colesterol sérico total, tc).", "Numérica continua", "Perfil lipídico."),
    ("S2", "Medida sérica S2 (lipoproteínas de baja densidad, ldl).", "Numérica continua", "Perfil lipídico."),
    ("S3", "Medida sérica S3 (lipoproteínas de alta densidad, hdl).", "Numérica continua", "Perfil lipídico (correlación negativa)."),
    ("S4", "Medida sérica S4 (colesterol total / HDL, tch).", "Numérica continua", "Perfil lipídico."),
    ("S5", "Medida sérica S5 (log triglicéridos séricos, ltg).", "Numérica continua", "Perfil lipídico."),
    ("S6", "Medida sérica S6 (nivel de glucosa en sangre, glu).", "Numérica continua", "Perfil glucémico."),
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
        "El objetivo es **predecir cuantitativamente la progresión de la "
        "diabetes un año después del inicio**, a partir de 10 variables "
        "basales del paciente (edad, sexo, IMC, presión arterial y seis "
        "medidas séricas), utilizando modelos de regresión."
    )
    df = load_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{df.shape[0]:,}")
    c2.metric("Variables", df.shape[1])
    c3.metric("Variable objetivo", "Y")
    c4.metric("Rango de progresión", f"{int(df.Y.min())} – {int(df.Y.max())}")

    st.subheader("Modelos implementados")
    st.write(
        "- **Regresión Lineal Múltiple**: relación lineal entre las "
        "variables basales y la progresión.\n"
        "- **Regresión Polinomial**: añade términos de grado 2 y 3 para "
        "capturar relaciones no lineales."
    )
    st.subheader("Sobre la variable objetivo")
    st.write(
        "`Y` es una medida cuantitativa (continua) de la progresión de la "
        "enfermedad. Este proyecto es **académico y predictivo**: el "
        "resultado no debe interpretarse como un diagnóstico médico."
    )


def render_descriptores() -> None:
    st.header("Descriptores")
    st.write(
        "Las 10 variables basales utilizadas por el modelo. Todas se "
        "conservan; el dataset no tiene variables descartadas."
    )
    st.table(pd.DataFrame(DESCRIPTORS, columns=["Variable", "Descripción", "Tipo", "Función"]))
    st.subheader("Variable objetivo")
    st.markdown(
        "**`Y`** — Progresión cuantitativa de la diabetes un año después "
        "del inicio. Es la variable que se desea predecir."
    )


def render_modelos() -> None:
    st.header("Modelos")
    st.subheader("Regresión Lineal Múltiple")
    st.write(
        "Modela la progresión como combinación lineal de las variables "
        "basales: `y = b0 + b1*x1 + ... + bn*xn`. Es el modelo base e "
        "interpretable."
    )
    st.subheader("Regresión Polinomial")
    st.write(
        "Extiende la regresión lineal añadiendo potencias e interacciones "
        "mediante `PolynomialFeatures`. Grados evaluados: **2** (66 "
        "características) y **3** (286 características). Con solo 442 "
        "registros, el grado 3 presenta alto riesgo de sobreajuste."
    )
    st.subheader("Pipeline común")
    st.code(
        "diabetes.tab.txt (separado por tabulaciones)\n"
        "  -> SimpleImputer(mediana)          # no hay nulos; por robustez\n"
        "  -> StandardScaler                  # estandariza las variables basales\n"
        "  -> [PolynomialFeatures (grado 2/3)]\n"
        "  -> LinearRegression\n"
        "  -> Predicción de progresión",
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
             caption=f"Progresión real vs predicha — {MODEL_KEYS[model_plot]}")
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

    st.markdown("##### Datos basales del paciente")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider(
                "Edad — AGE (años)", min_value=numeric["AGE"]["min"],
                max_value=numeric["AGE"]["max"],
                value=numeric["AGE"]["median"], step=1.0, format="%.0f")
            sex = st.selectbox("Sexo — SEX", [1, 2],
                               format_func=lambda v: "1 (hombre)" if v == 1 else "2 (mujer)")
            bmi = st.slider(
                "Índice de masa corporal — BMI", min_value=numeric["BMI"]["min"],
                max_value=numeric["BMI"]["max"],
                value=numeric["BMI"]["median"], step=0.1, format="%.1f")
            bp = st.slider(
                "Presión arterial media — BP", min_value=numeric["BP"]["min"],
                max_value=numeric["BP"]["max"],
                value=numeric["BP"]["median"], step=1.0, format="%.1f")
            s1 = st.slider(
                "S1 (colesterol sérico total)", min_value=numeric["S1"]["min"],
                max_value=numeric["S1"]["max"],
                value=numeric["S1"]["median"], step=1.0, format="%.0f")
            s2 = st.slider(
                "S2 (LDL)", min_value=numeric["S2"]["min"],
                max_value=numeric["S2"]["max"],
                value=numeric["S2"]["median"], step=0.1, format="%.1f")
        with c2:
            s3 = st.slider(
                "S3 (HDL)", min_value=numeric["S3"]["min"],
                max_value=numeric["S3"]["max"],
                value=numeric["S3"]["median"], step=0.1, format="%.1f")
            s4 = st.slider(
                "S4 (colesterol total / HDL)", min_value=numeric["S4"]["min"],
                max_value=numeric["S4"]["max"],
                value=numeric["S4"]["median"], step=0.1, format="%.1f")
            s5 = st.slider(
                "S5 (log triglicéridos séricos)", min_value=numeric["S5"]["min"],
                max_value=numeric["S5"]["max"],
                value=numeric["S5"]["median"], step=0.01, format="%.3f")
            s6 = st.slider(
                "S6 (glucosa en sangre)", min_value=numeric["S6"]["min"],
                max_value=numeric["S6"]["max"],
                value=numeric["S6"]["median"], step=1.0, format="%.0f")
        submitted = st.form_submit_button("Predecir progresión")

    if submitted:
        features = {
            "AGE": age,
            "SEX": sex,
            "BMI": bmi,
            "BP": bp,
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "S4": s4,
            "S5": s5,
            "S6": s6,
        }
        with st.spinner("Calculando predicción..."):
            value = predict(model_key, features)

        st.divider()
        st.markdown("### Resultado")
        st.markdown(
            f"#### Progresión estimada de la diabetes\n\n"
            f"# {value:.2f}",
            help="Predicción continua del modelo sobre la progresión de la enfermedad.",
        )
        st.write(f"**Modelo seleccionado:** {MODEL_KEYS[model_key]}")
        st.write("**Variables ingresadas:**")
        st.json(features)
        st.caption(
            "Este proyecto es académico y predictivo: el valor es una "
            "estimación estadística y **no constituye un diagnóstico médico**."
        )


def render() -> None:
    """Punto de entrada de la interfaz del caso (llamado por el menú)."""
    st.sidebar.title("Caso 3 — Diabetes")
    if st.sidebar.button("← Volver al menú principal", use_container_width=True):
        go_home()
    section = st.sidebar.radio(
        "Secciones",
        ["Información", "Descriptores", "Modelos", "Métricas", "Predicción"],
    )

    st.title("Diabetes")
    st.caption("Predicción de la progresión de la diabetes a partir de variables basales.")

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
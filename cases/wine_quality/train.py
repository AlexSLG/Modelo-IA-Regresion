"""Entrenamiento y guardado de los modelos del caso Wine Quality.

Ejecutable con:
    python -m cases.wine_quality.train

Entrena los tres pipelines (lineal, polinomial grado 2 y grado 3),
evalúa sobre train/test y guarda modelos, métricas, gráficos y metadatos
en ``models/`` y ``results/``. Los modelos se guardan con joblib y la
aplicación Streamlit solo los carga (no reentrena).
"""

import json

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from .config import (
    CASE_ID,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET,
    TEST_SIZE,
)
from .evaluate import (
    plot_actual_vs_predicted,
    plot_correlation_matrix,
    plot_metrics_comparison,
    plot_residuals,
    plot_target_distribution,
    metrics_table,
)
from .preprocessing import build_preprocessor, get_feature_names, load_data


def build_models(preprocessor) -> dict:
    """Construye los tres pipelines de regresión del caso."""
    return {
        "linear_regression": Pipeline(
            steps=[
                ("pre", preprocessor),
                ("lr", LinearRegression()),
            ]
        ),
        "polynomial_degree_2": Pipeline(
            steps=[
                ("pre", preprocessor),
                # Expande las propiedades a términos de grado 2 (78 features).
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("lr", LinearRegression()),
            ]
        ),
        "polynomial_degree_3": Pipeline(
            steps=[
                ("pre", preprocessor),
                # Grado 3 (364 features): mayor riesgo de sobreajuste.
                ("poly", PolynomialFeatures(degree=3, include_bias=False)),
                ("lr", LinearRegression()),
            ]
        ),
    }


def compute_input_ranges(df: pd.DataFrame) -> dict:
    """Rangos y valores típicos reales para el formulario de predicción."""
    numeric = df[NUMERIC_FEATURES].agg(["min", "max", "mean", "median"]).round(3)
    return {
        "numeric": {
            col: {
                "min": float(numeric.loc["min", col]),
                "max": float(numeric.loc["max", col]),
                "mean": float(numeric.loc["mean", col]),
                "median": float(numeric.loc["median", col]),
            }
            for col in NUMERIC_FEATURES
        },
        "target": {
            "name": TARGET,
            "min": int(df[TARGET].min()),
            "max": int(df[TARGET].max()),
            "levels": sorted(df[TARGET].unique().tolist()),
        },
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # División 80/20 sin data leakage: los transformadores solo se ajustan con train.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODELS_DIR / f"{name}.pkl")
        print(f"Entrenado y guardado: {name}")

    # Nombres de las características finales (después del preprocesado).
    preprocessor.fit(X_train, y_train)
    feature_names = get_feature_names(preprocessor)
    (MODELS_DIR / "feature_names.json").write_text(
        json.dumps({"model_id": CASE_ID, "features": feature_names}, indent=2),
        encoding="utf-8",
    )

    # Rangos reales del dataset para el formulario de predicción.
    (RESULTS_DIR / "input_ranges.json").write_text(
        json.dumps(compute_input_ranges(df), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Tabla comparativa de métricas (train y test).
    table = metrics_table(models, X_train, y_train, X_test, y_test)
    table.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(table.round(3).to_string(index=False))

    # Visualizaciones.
    plot_correlation_matrix(df, RESULTS_DIR / "correlation_matrix.png")
    plot_target_distribution(df, RESULTS_DIR / "target_distribution.png")
    plot_metrics_comparison(table, RESULTS_DIR / "metrics_comparison.png")
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        plot_actual_vs_predicted(
            y_test, y_pred,
            f"Calidad real vs predicha — {name}",
            RESULTS_DIR / f"actual_vs_predicted_{name}.png",
        )
        plot_residuals(
            y_test, y_pred,
            f"Residuos — {name}",
            RESULTS_DIR / f"residuals_{name}.png",
        )

    print("\nModelos y resultados guardados en:", MODELS_DIR, "y", RESULTS_DIR)


if __name__ == "__main__":
    main()
"""Predicción con los modelos entrenados del caso Wine Quality.

La aplicación Streamlit carga los pipelines guardados con joblib y
solo realiza predicciones; nunca vuelve a entrenar el modelo.
"""

import pandas as pd
import joblib

from .config import MODELS_DIR, MODEL_KEYS


def load_model(model_key: str):
    """Carga el pipeline entrenado correspondiente a ``model_key``."""
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Modelo desconocido: {model_key}. Válidos: {list(MODEL_KEYS)}")
    path = MODELS_DIR / f"{model_key}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo {path}. Ejecuta primero: "
            "python -m cases.wine_quality.train"
        )
    return joblib.load(path)


def predict(model_key: str, features: dict) -> float:
    """Predice la calidad del vino a partir de un diccionario de entrada.

    ``features`` debe contener las 11 propiedades físico-químicas del
    caso. Devuelve la calidad estimada (valor continuo, típicamente
    entre 3 y 8).
    """
    pipeline = load_model(model_key)
    # El orden de las columnas debe coincidir con el del entrenamiento.
    x = pd.DataFrame([features])
    prediction = pipeline.predict(x)[0]
    return float(prediction)
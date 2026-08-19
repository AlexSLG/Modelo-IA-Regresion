"""Preprocesamiento del caso Wine Quality.

Construye el preprocesador de los modelos: imputación (no-op, el
dataset no tiene nulos) y estandarización de las propiedades
físico-químicas del vino.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_FILE, ID_COLUMN, NUMERIC_FEATURES


def load_data() -> pd.DataFrame:
    """Carga el dataset real y descarta la columna de índice ``Id``."""
    df = pd.read_csv(DATA_FILE)
    return df.drop(columns=[ID_COLUMN])


def build_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador compartido por los modelos.

    Solo hay variables numéricas: se imputan nulos con la mediana (aunque
    el dataset no presenta valores faltantes) y se estandarizan para que
    todas las propiedades tengan la misma escala.
    """
    numeric_pipeline = Pipeline(
        steps=[
            # Imputa la mediana por robustez; el dataset no tiene nulos.
            ("imputer", SimpleImputer(strategy="median")),
            # Estandariza las propiedades (media 0, desviación 1).
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Devuelve los nombres de las características tras el preprocesado."""
    return NUMERIC_FEATURES
"""Preprocesamiento del caso California Housing.

Construye el preprocesador común a todos los modelos:
imputación de valores nulos, codificación de la variable categórica
y estandarización de las variables numéricas.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    DATA_FILE,
    NUMERIC_FEATURES,
)


def load_data() -> pd.DataFrame:
    """Carga el dataset original del caso."""
    return pd.read_csv(DATA_FILE)


def build_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador compartido por todos los modelos.

    - Numéricas: imputación de nulos con la mediana y estandarización.
    - Categóricas: One-Hot Encoding (se descarta la primera categoría
      para evitar la trampa de las variables dummy).
    """
    numeric_pipeline = Pipeline(
        steps=[
            # Imputa los nulos de total_bedrooms (~1%) con la mediana.
            ("imputer", SimpleImputer(strategy="median")),
            # Estandariza para evitar escalas muy diferentes (media 0, varianza 1).
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Convierte la variable categórica en variables binarias.
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Devuelve los nombres de las características tras el preprocesado."""
    numeric_names = NUMERIC_FEATURES
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categorical_names = onehot.get_feature_names_out(CATEGORICAL_FEATURES)
    return numeric_names + list(categorical_names)
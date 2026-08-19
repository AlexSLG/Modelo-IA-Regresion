"""Preprocesamiento del caso Diabetes.

Carga el archivo ``diabetes.tab.txt`` (separado por tabulaciones) y
construye el preprocesador de los modelos: imputación (no-op, no hay
nulos) y estandarización de las variables basales.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_FILE, DATA_SEPARATOR, NUMERIC_FEATURES


def load_data() -> pd.DataFrame:
    """Carga el dataset real respetando su formato tabulado."""
    return pd.read_csv(DATA_FILE, sep=DATA_SEPARATOR)


def build_preprocessor() -> ColumnTransformer:
    """Construye el preprocesador compartido por los modelos.

    Solo hay variables numéricas: se imputan nulos con la mediana (aunque
    el dataset no presenta valores faltantes) y se estandarizan para que
    todas las variables basales tengan la misma escala.
    """
    numeric_pipeline = Pipeline(
        steps=[
            # Imputa la mediana por robustez; el dataset no tiene nulos.
            ("imputer", SimpleImputer(strategy="median")),
            # Estandariza las variables (media 0, desviación 1).
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
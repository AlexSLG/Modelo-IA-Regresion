"""Configuración central del caso California Housing.

Define rutas relativas al proyecto (nunca absolutas), el identificador
del caso, la variable objetivo, los descriptores y el ``random_state``
para garantizar resultados reproducibles.
"""

from pathlib import Path

# Raíz del proyecto: tres niveles hacia arriba desde este archivo.
BASE_DIR = Path(__file__).resolve().parents[2]

CASE_ID = "california_housing"
CASE_TITLE = "California Housing"

DATA_FILE = BASE_DIR / "data" / "housing(1).csv"
MODELS_DIR = BASE_DIR / "models" / CASE_ID
RESULTS_DIR = BASE_DIR / "results" / CASE_ID

# Reproducibilidad de la división train/test.
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET = "median_house_value"

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]

CATEGORICAL_FEATURES = ["ocean_proximity"]

MODEL_KEYS = {
    "linear_regression": "Regresión Lineal Múltiple",
    "polynomial_degree_2": "Regresión Polinomial (grado 2)",
    "polynomial_degree_3": "Regresión Polinomial (grado 3)",
}
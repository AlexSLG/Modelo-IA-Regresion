"""Configuración central del caso Wine Quality.

Define rutas relativas al proyecto (nunca absolutas), la variable
objetivo, los descriptores y el ``random_state`` para reproducibilidad.
"""

from pathlib import Path

# Raíz del proyecto: tres niveles hacia arriba desde este archivo.
BASE_DIR = Path(__file__).resolve().parents[2]

CASE_ID = "wine_quality"
CASE_TITLE = "Wine Quality"

DATA_FILE = BASE_DIR / "data" / "WineQT.csv"
MODELS_DIR = BASE_DIR / "models" / CASE_ID
RESULTS_DIR = BASE_DIR / "results" / CASE_ID

# Reproducibilidad de la división train/test.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# La variable objetivo es la calidad del vino (entera, 3-8). Se trata
# como variable numérica continua porque el ejercicio exige regresión.
TARGET = "quality"

# La columna 'Id' es un índice secuencial, no una propiedad química;
# se descarta al cargar los datos.
ID_COLUMN = "Id"

NUMERIC_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

# Este caso no tiene variables categóricas, por lo que no se aplica One-Hot.
CATEGORICAL_FEATURES = []

MODEL_KEYS = {
    "linear_regression": "Regresión Lineal Múltiple",
    "polynomial_degree_2": "Regresión Polinomial (grado 2)",
    "polynomial_degree_3": "Regresión Polinomial (grado 3)",
}
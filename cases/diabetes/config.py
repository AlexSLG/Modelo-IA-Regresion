"""Configuración central del caso Diabetes.

Define rutas relativas al proyecto (nunca absolutas), la variable
objetivo, los descriptores y el ``random_state`` para reproducibilidad.
"""

from pathlib import Path

# Raíz del proyecto: tres niveles hacia arriba desde este archivo.
BASE_DIR = Path(__file__).resolve().parents[2]

CASE_ID = "diabetes"
CASE_TITLE = "Diabetes"

# El dataset NO es un CSV con coma: es un archivo de texto separado por
# tabulaciones (.tab.txt). Se lee con sep="\t" en preprocessing.
DATA_FILE = BASE_DIR / "data" / "diabetes.tab.txt"
DATA_SEPARATOR = "\t"
MODELS_DIR = BASE_DIR / "models" / CASE_ID
RESULTS_DIR = BASE_DIR / "results" / CASE_ID

# Reproducibilidad de la división train/test.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Progresión cuantitativa de la enfermedad un año después del inicio.
TARGET = "Y"

NUMERIC_FEATURES = [
    "AGE",
    "SEX",
    "BMI",
    "BP",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
]

# El caso no tiene variables categóricas con nombres (SEX es 1/2 y se
# trata como numérica), por lo que no se aplica One-Hot.
CATEGORICAL_FEATURES = []

MODEL_KEYS = {
    "linear_regression": "Regresión Lineal Múltiple",
    "polynomial_degree_2": "Regresión Polinomial (grado 2)",
    "polynomial_degree_3": "Regresión Polinomial (grado 3)",
}
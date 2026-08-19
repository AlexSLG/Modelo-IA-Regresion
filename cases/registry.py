"""Registro central de casos de estudio.

Para incorporar un nuevo caso (p. ej. Wine Quality o Diabetes) basta con:
1. Crear ``cases/<id>/app.py`` con una función ``render()``.
2. Añadir una entrada a :data:`CASES` con ``status="available"``.

El menú principal de ``app.py`` lee este registro, por lo que no es
necesario rediseñar la aplicación.
"""

from importlib import import_module

CASES = [
    {
        "id": "california_housing",
        "name": "California Housing",
        "title": "Caso 1 — California Housing",
        "description": (
            "Predicción del valor medio de viviendas en distritos de "
            "California mediante modelos de regresión."
        ),
        "button": "Ingresar al Caso 1",
        "module": "cases.california_housing.app",
        "status": "available",
    },
    {
        "id": "wine_quality",
        "name": "Wine Quality",
        "title": "Caso 2 — Wine Quality",
        "description": (
            "Estimación de la calidad de vinos a partir de sus propiedades "
            "físico-químicas (acidez, azúcar, cloruros, alcohol)."
        ),
        "button": "Ingresar al Caso 2",
        "module": "cases.wine_quality.app",
        "status": "available",
    },
    {
        "id": "diabetes",
        "name": "Diabetes",
        "title": "Caso 3 — Diabetes",
        "description": (
            "Predicción de la progresión de la diabetes a partir de "
            "variables basales de los pacientes (edad, IMC, presión, etc.)."
        ),
        "button": "Ingresar al Caso 3",
        "module": "cases.diabetes.app",
        "status": "available",
    },
]


def get_case(case_id: str) -> dict:
    """Devuelve la entrada de registro de un caso por su identificador."""
    return next(case for case in CASES if case["id"] == case_id)


def render_case(case_id: str) -> None:
    """Importa y renderiza la interfaz del caso seleccionado."""
    case = get_case(case_id)
    module = import_module(case["module"])
    module.render()
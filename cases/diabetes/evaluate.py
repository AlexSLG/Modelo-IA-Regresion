"""Evaluación de los modelos del caso Diabetes.

Calcula las métricas de regresión (MAE, MSE, RMSE, R²) y genera las
visualizaciones académicas del caso, adaptadas a la variable objetivo
``Y`` (progresión de la diabetes).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Estilo profesional para las figuras.
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def compute_metrics(y_true, y_pred) -> dict:
    """Calcula MAE, MSE, RMSE y R² para un par reales/predichos."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def metrics_table(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Construye la tabla comparativa de métricas en entrenamiento y prueba."""
    rows = []
    for name, pipeline in models.items():
        train_metrics = compute_metrics(y_train, pipeline.predict(X_train))
        test_metrics = compute_metrics(y_test, pipeline.predict(X_test))
        rows.append(
            {
                "Modelo": name,
                "R2_train": train_metrics["R2"],
                "RMSE_train": train_metrics["RMSE"],
                "R2_test": test_metrics["R2"],
                "RMSE_test": test_metrics["RMSE"],
                "MAE_test": test_metrics["MAE"],
                "MSE_test": test_metrics["MSE"],
            }
        )
    return pd.DataFrame(rows)


def plot_correlation_matrix(df: pd.DataFrame, path: Path) -> None:
    """Matriz de correlación de las variables basales."""
    corr = df.select_dtypes(include="number").corr()
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"shrink": 0.8})
    plt.title("Matriz de correlación — Diabetes")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_target_distribution(df: pd.DataFrame, path: Path) -> None:
    """Distribución de la variable objetivo (progresión)."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Y"], bins=30, kde=True, color="#2E86AB")
    plt.title("Distribución de la progresión de la diabetes (Y)")
    plt.xlabel("Progresión un año después del inicio")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_actual_vs_predicted(y_test, y_pred, title: str, path: Path) -> None:
    """Valores reales frente a valores predichos."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, s=25, color="#2E86AB")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "--", color="#C0392B", label="Línea ideal (y = x)")
    plt.xlabel("Progresión real")
    plt.ylabel("Progresión predicha")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_residuals(y_test, y_pred, title: str, path: Path) -> None:
    """Gráfico de residuos (error = real - predicho)."""
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, s=25, color="#2E86AB")
    plt.axhline(0, color="#C0392B", linestyle="--")
    plt.xlabel("Progresión predicha")
    plt.ylabel("Residuos")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame, path: Path) -> None:
    """Comparación visual de las métricas entre modelos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x="Modelo", y="RMSE_test", data=metrics_df, ax=axes[0], color="#2E86AB")
    axes[0].set_title("RMSE en prueba por modelo")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(x="Modelo", y="R2_test", data=metrics_df, ax=axes[1], color="#27AE60")
    axes[1].set_title("R² en prueba por modelo")
    axes[1].set_ylabel("R²")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
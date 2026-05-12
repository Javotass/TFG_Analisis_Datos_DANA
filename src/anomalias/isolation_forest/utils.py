"""Utilidades compartidas para detectores Isolation Forest."""

from __future__ import annotations

import numpy as np


def validar_dimensiones(data, required: set[str]) -> None:
    """Valida que el DataArray contenga las dimensiones requeridas."""
    missing = required - set(data.dims)
    if missing:
        raise ValueError(f"Faltan dimensiones en data: {missing}")


def imputar_nan_por_mediana_columna(X: np.ndarray) -> np.ndarray:
    """Imputa NaN por la mediana de cada columna en una matriz 2-D."""
    col_medians = np.nanmedian(X, axis=0)
    return np.where(np.isnan(X), col_medians[np.newaxis, :], X)


def combinar_mascaras(
    mascara_if: np.ndarray,
    mascara_aux: np.ndarray,
    usar_criterio_aux: bool,
    combinar_criterios: str,
) -> np.ndarray:
    """Combina mascara IF y mascara auxiliar con regla OR/AND."""
    if not usar_criterio_aux:
        return mascara_if

    criterio = combinar_criterios.strip().lower()
    if criterio not in {"or", "and"}:
        raise ValueError("'combinar_criterios' debe ser 'or' o 'and'")
    if criterio == "or":
        return mascara_if | mascara_aux
    return mascara_if & mascara_aux

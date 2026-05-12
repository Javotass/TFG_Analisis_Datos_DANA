"""
Deteccion temporal de anomalias con ecuaciones en diferencias de primer orden.

Modelo base:
    x_t = a * x_(t-1) + b

La serie x_t se construye como la media espacial de la variable meteorologica
en cada timestep. Se estiman a y b por minimos cuadrados, se calcula la serie
predicha y se detectan anomalias por z-score del residuo.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _clasificar_estabilidad(a: float) -> str:
    """Clasifica la dinamica segun el valor de a (tutorial de diferencias)."""
    if abs(a) < 1:
        if 0 < a < 1:
            return "estable_monotona"
        if -1 < a < 0:
            return "estable_oscilatoria"
        return "estable"
    if a > 1:
        return "inestable_monotona"
    if a < -1:
        return "inestable_oscilatoria"
    if np.isclose(a, 1.0):
        return "caso_a_igual_1"
    if np.isclose(a, -1.0):
        return "frontera_oscilatoria"
    return "frontera_estabilidad"


def detectar_anomalias_diferencias_primer_orden_temporal(
    data: xr.DataArray,
    umbral_zscore_residuo: float = 2.5,
    min_puntos_ajuste: int = 10,
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """
    Detecta anomalias temporales ajustando un modelo x_t = a*x_(t-1) + b.

    Parameters
    ----------
    data : xr.DataArray
        Array con dimensiones (valid_time, latitude, longitude).
    umbral_zscore_residuo : float
        Umbral de |z-score| sobre los residuos para marcar anomalias.
    min_puntos_ajuste : int
        Minimo de pares validos (x_(t-1), x_t) para estimar a y b.

    Returns
    -------
    scores : xr.DataArray (valid_time,)
        Score temporal: -|z_residuo| (mas negativo = mas anomalo).
    mascara : xr.DataArray bool (valid_time,)
        True en timesteps anomalos segun el umbral de z-score.
    estadisticas : dict
        Parametros estimados, estabilidad, equilibrio y resumen de anomalias.
    """
    required = {"valid_time", "latitude", "longitude"}
    missing = required - set(data.dims)
    if missing:
        raise ValueError(f"Faltan dimensiones en data: {missing}")

    serie_media_da = data.mean(dim=["latitude", "longitude"])
    y = serie_media_da.values.astype(float)
    times = data.valid_time.values

    if y.size < 3:
        raise ValueError("Se requieren al menos 3 timesteps para ajustar el modelo")

    x_prev = y[:-1]
    x_curr = y[1:]
    mask_valid = np.isfinite(x_prev) & np.isfinite(x_curr)
    n_valid = int(mask_valid.sum())

    if n_valid < min_puntos_ajuste:
        raise ValueError(
            "No hay suficientes pares validos para ajustar x_t = a*x_(t-1)+b: "
            f"{n_valid} < {min_puntos_ajuste}"
        )

    A = np.column_stack((x_prev[mask_valid], np.ones(n_valid)))
    target = x_curr[mask_valid]
    coef, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    a = float(coef[0])
    b = float(coef[1])

    y_pred = np.full_like(y, np.nan, dtype=float)
    y_pred[1:] = a * y[:-1] + b

    resid = y - y_pred
    resid[0] = np.nan

    resid_mu = float(np.nanmean(resid))
    resid_std = float(np.nanstd(resid))
    if resid_std > 0:
        z_resid = (resid - resid_mu) / resid_std
    else:
        z_resid = np.zeros_like(resid)
        z_resid[0] = np.nan

    mascara_vals = np.abs(z_resid) >= umbral_zscore_residuo
    mascara_vals = np.where(np.isnan(z_resid), False, mascara_vals)

    scores_vals = -np.abs(z_resid)
    scores_vals = np.where(np.isnan(scores_vals), 0.0, scores_vals)

    if np.isclose(1.0 - a, 0.0):
        equilibrio = np.nan
    else:
        equilibrio = b / (1.0 - a)

    tipo_estabilidad = _clasificar_estabilidad(a)
    es_estable = bool(abs(a) < 1)

    scores_da = xr.DataArray(
        scores_vals,
        dims=["valid_time"],
        coords={"valid_time": times},
        name="fod_temporal_score",
        attrs={
            "long_name": "Score por residuo en ecuacion de primer orden",
            "variable": str(data.name),
            "modelo": "x_t = a*x_(t-1) + b",
            "a": a,
            "b": b,
            "equilibrio": float(equilibrio) if np.isfinite(equilibrio) else np.nan,
            "tipo_estabilidad": tipo_estabilidad,
            "umbral_zscore_residuo": umbral_zscore_residuo,
        },
    )
    scores_da = scores_da.assign_coords(
        serie_media=("valid_time", y),
        serie_pred=("valid_time", y_pred),
        residuo=("valid_time", resid),
        zscore_residuo=("valid_time", z_resid),
    )

    mascara_da = xr.DataArray(
        mascara_vals.astype(bool),
        dims=["valid_time"],
        coords={"valid_time": times},
        name="fod_temporal_anomalia",
        attrs={
            "long_name": "Mascara temporal de anomalias por residuo AR(1)",
            "variable": str(data.name),
            "umbral_zscore_residuo": umbral_zscore_residuo,
            "a": a,
            "b": b,
            "tipo_estabilidad": tipo_estabilidad,
        },
    )

    n_time = int(len(times))
    n_eval = int(np.isfinite(z_resid).sum())
    n_anom = int(mascara_vals.sum())
    pct = round(100.0 * n_anom / n_eval, 3) if n_eval > 0 else 0.0

    estadisticas = {
        "n_total_timesteps": n_time,
        "n_timesteps_evaluados": n_eval,
        "num_anomalias": n_anom,
        "porcentaje_anomalias": pct,
        "umbral_zscore_residuo": umbral_zscore_residuo,
        "a": a,
        "b": b,
        "equilibrio": float(equilibrio) if np.isfinite(equilibrio) else None,
        "estable": es_estable,
        "tipo_estabilidad": tipo_estabilidad,
        "residuo_media": resid_mu,
        "residuo_std": resid_std,
        "score_min": float(np.nanmin(scores_vals)),
        "score_max": float(np.nanmax(scores_vals)),
        "score_media": float(np.nanmean(scores_vals)),
        "min_puntos_ajuste": min_puntos_ajuste,
    }

    return scores_da, mascara_da, estadisticas

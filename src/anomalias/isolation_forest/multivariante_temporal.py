"""Isolation Forest multivariante temporal."""

from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.ensemble import IsolationForest

from .utils import imputar_nan_por_mediana_columna, combinar_mascaras


def detectar_anomalias_isolation_forest_temporal_multivariante(
    variables: dict[str, xr.DataArray],
    n_estimators: int = 300,
    contamination: float = 0.05,
    random_state: int = 42,
    n_jobs: int = -1,
    usar_criterio_media: bool = True,
    umbral_zscore_media: float = 2.0,
    combinar_criterios: str = "or",
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """Aplica Isolation Forest temporal multivariante."""
    if not variables:
        raise ValueError("El diccionario 'variables' esta vacio")

    ref_da = next(iter(variables.values()))
    times = ref_da.valid_time.values
    n_time = len(times)
    n_lat = len(ref_da.latitude)
    n_lon = len(ref_da.longitude)

    X_parts = []
    n_features_total = 0
    for _, da in variables.items():
        arr = da.values
        X_var = arr.reshape(n_time, n_lat * n_lon)
        X_var = imputar_nan_por_mediana_columna(X_var)
        X_parts.append(X_var)
        n_features_total += n_lat * n_lon

    X = np.concatenate(X_parts, axis=1)

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    clf.fit(X)

    raw_scores = clf.score_samples(X)
    etiquetas = clf.predict(X)

    media_por_timestep = np.nanmean(X, axis=1)
    media_global = float(np.nanmean(media_por_timestep))
    std_global = float(np.nanstd(media_por_timestep))
    if std_global > 0:
        zscores_media = (media_por_timestep - media_global) / std_global
    else:
        zscores_media = np.zeros_like(media_por_timestep)

    mascara_if = etiquetas == -1
    mascara_media = np.abs(zscores_media) >= umbral_zscore_media
    mascara_final = combinar_mascaras(
        mascara_if,
        mascara_media,
        usar_criterio_aux=usar_criterio_media,
        combinar_criterios=combinar_criterios,
    )

    scores_da = xr.DataArray(
        raw_scores,
        dims=["valid_time"],
        coords={"valid_time": times},
        name="if_temporal_multi_score",
        attrs={
            "long_name": "Isolation Forest temporal multivariate score",
            "variables": list(variables.keys()),
            "description": "Score por timestep sobre combinacion de variables",
        },
    )

    mascara_da = xr.DataArray(
        mascara_final,
        dims=["valid_time"],
        coords={"valid_time": times},
        name="if_temporal_multi_anomalia",
        attrs={
            "long_name": "Isolation Forest temporal multivariate mask",
            "variables": list(variables.keys()),
            "usar_criterio_media": usar_criterio_media,
            "umbral_zscore_media": umbral_zscore_media,
            "combinar_criterios": combinar_criterios,
        },
    )
    mascara_da = mascara_da.assign_coords(
        if_mask=("valid_time", mascara_if),
        media_mask=("valid_time", mascara_media),
    )

    n_anomalias = int(mascara_final.sum())
    n_anomalias_if = int(mascara_if.sum())
    n_anomalias_media = int(mascara_media.sum())
    estadisticas = {
        "n_total_timesteps": n_time,
        "num_anomalias": n_anomalias,
        "porcentaje_anomalias": round(100 * n_anomalias / n_time, 3),
        "num_anomalias_if": n_anomalias_if,
        "num_anomalias_media": n_anomalias_media,
        "porcentaje_anomalias_if": round(100 * n_anomalias_if / n_time, 3),
        "porcentaje_anomalias_media": round(100 * n_anomalias_media / n_time, 3),
        "score_min": float(np.nanmin(raw_scores)),
        "score_max": float(np.nanmax(raw_scores)),
        "score_media": float(np.nanmean(raw_scores)),
        "media_global_temporal": media_global,
        "std_global_temporal": std_global,
        "usar_criterio_media": usar_criterio_media,
        "umbral_zscore_media": umbral_zscore_media,
        "combinar_criterios": combinar_criterios,
        "n_estimators": n_estimators,
        "contamination": contamination,
        "variables": list(variables.keys()),
        "n_features_total": n_features_total,
    }

    return scores_da, mascara_da, estadisticas

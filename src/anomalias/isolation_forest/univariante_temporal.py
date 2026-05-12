"""Isolation Forest univariante temporal."""

from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.ensemble import IsolationForest

from .utils import validar_dimensiones, imputar_nan_por_mediana_columna, combinar_mascaras


def detectar_anomalias_isolation_forest_temporal(
    data: xr.DataArray,
    n_estimators: int = 200,
    contamination: float = 0.05,
    random_state: int = 42,
    n_jobs: int = -1,
    usar_criterio_media: bool = True,
    umbral_zscore_media: float = 2.0,
    combinar_criterios: str = "or",
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """Aplica Isolation Forest temporal y devuelve score, mascara y estadisticas."""
    validar_dimensiones(data, {"valid_time", "latitude", "longitude"})

    times = data.valid_time.values
    n_time = len(times)
    n_lat = len(data.latitude)
    n_lon = len(data.longitude)

    arr = data.values
    X = arr.reshape(n_time, n_lat * n_lon)
    X_clean = imputar_nan_por_mediana_columna(X)

    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    clf.fit(X_clean)

    raw_scores = clf.score_samples(X_clean)
    etiquetas = clf.predict(X_clean)
    offset = float(clf.offset_)

    media_por_timestep = np.nanmean(X_clean, axis=1)
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
        name="if_temporal_score",
        attrs={
            "long_name": "Isolation Forest temporal anomaly score",
            "variable": str(data.name),
            "description": "Score por timestep; mas negativo = patron espacial mas atipico",
        },
    )

    mascara_da = xr.DataArray(
        mascara_final,
        dims=["valid_time"],
        coords={"valid_time": times},
        name="if_temporal_anomalia",
        attrs={
            "long_name": "Isolation Forest temporal anomaly mask",
            "variable": str(data.name),
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
        "offset_decision": offset,
        "media_global_temporal": media_global,
        "std_global_temporal": std_global,
        "usar_criterio_media": usar_criterio_media,
        "umbral_zscore_media": umbral_zscore_media,
        "combinar_criterios": combinar_criterios,
        "n_estimators": n_estimators,
        "contamination": contamination,
        "n_features_espaciales": n_lat * n_lon,
    }

    return scores_da, mascara_da, estadisticas

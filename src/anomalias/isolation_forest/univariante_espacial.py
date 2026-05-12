"""Isolation Forest univariante espacial."""

from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.ensemble import IsolationForest

from .utils import validar_dimensiones, imputar_nan_por_mediana_columna, combinar_mascaras


def detectar_anomalias_isolation_forest(
    data: xr.DataArray,
    n_estimators: int = 200,
    contamination: float = 0.02,
    random_state: int = 42,
    n_jobs: int = -1,
    usar_criterio_media: bool = True,
    umbral_zscore_media: float = 2.0,
    combinar_criterios: str = "or",
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """Aplica Isolation Forest univariante y devuelve score, mascara y estadisticas."""
    validar_dimensiones(data, {"valid_time", "latitude", "longitude"})

    lats = data.latitude.values
    lons = data.longitude.values
    n_lat, n_lon = len(lats), len(lons)
    n_time = len(data.valid_time)

    arr = data.values
    X = arr.transpose(1, 2, 0).reshape(n_lat * n_lon, n_time)
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

    media_local = np.nanmean(X_clean, axis=1)
    media_global = float(np.nanmean(media_local))
    std_global = float(np.nanstd(media_local))
    if std_global > 0:
        zscores_media = (media_local - media_global) / std_global
    else:
        zscores_media = np.zeros_like(media_local)

    mascara_if_flat = etiquetas == -1
    mascara_media_flat = np.abs(zscores_media) >= umbral_zscore_media
    mascara_final_flat = combinar_mascaras(
        mascara_if_flat,
        mascara_media_flat,
        usar_criterio_aux=usar_criterio_media,
        combinar_criterios=combinar_criterios,
    )

    scores_2d = raw_scores.reshape(n_lat, n_lon)
    anomalas_2d = mascara_final_flat.reshape(n_lat, n_lon)

    scores_da = xr.DataArray(
        scores_2d,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="if_score",
        attrs={"long_name": "Isolation Forest anomaly score", "variable": str(data.name)},
    )

    mascara_da = xr.DataArray(
        anomalas_2d,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="if_anomalia",
        attrs={
            "long_name": "Isolation Forest anomaly mask (hybrid optional)",
            "variable": str(data.name),
            "usar_criterio_media": usar_criterio_media,
            "umbral_zscore_media": umbral_zscore_media,
            "combinar_criterios": combinar_criterios,
        },
    )
    mascara_da = mascara_da.assign_coords(
        if_mask=(("latitude", "longitude"), mascara_if_flat.reshape(n_lat, n_lon)),
        media_mask=(("latitude", "longitude"), mascara_media_flat.reshape(n_lat, n_lon)),
    )

    n_anomalias = int(anomalas_2d.sum())
    n_anomalias_if = int(mascara_if_flat.sum())
    n_anomalias_media = int(mascara_media_flat.sum())
    n_total = n_lat * n_lon
    estadisticas = {
        "n_total_puntos": n_total,
        "num_anomalias": n_anomalias,
        "porcentaje_anomalias": round(100 * n_anomalias / n_total, 3),
        "num_anomalias_if": n_anomalias_if,
        "num_anomalias_media": n_anomalias_media,
        "porcentaje_anomalias_if": round(100 * n_anomalias_if / n_total, 3),
        "porcentaje_anomalias_media": round(100 * n_anomalias_media / n_total, 3),
        "score_min": float(np.nanmin(raw_scores)),
        "score_max": float(np.nanmax(raw_scores)),
        "score_media": float(np.nanmean(raw_scores)),
        "media_global_espacial": media_global,
        "std_global_espacial": std_global,
        "usar_criterio_media": usar_criterio_media,
        "umbral_zscore_media": umbral_zscore_media,
        "combinar_criterios": combinar_criterios,
        "n_estimators": n_estimators,
        "contamination": contamination,
        "n_timesteps_features": n_time,
    }

    return scores_da, mascara_da, estadisticas

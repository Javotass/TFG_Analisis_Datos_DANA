"""Isolation Forest multivariante espacial."""

from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.ensemble import IsolationForest

from .utils import imputar_nan_por_mediana_columna


def detectar_anomalias_isolation_forest_multivariante(
    variables: dict[str, xr.DataArray],
    n_estimators: int = 300,
    contamination: float = 0.02,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """Aplica Isolation Forest multivariante espacial."""
    if not variables:
        raise ValueError("El diccionario 'variables' está vacío")

    ref_da = next(iter(variables.values()))
    lats = ref_da.latitude.values
    lons = ref_da.longitude.values
    n_lat, n_lon = len(lats), len(lons)

    X_parts = []
    n_time_total = 0
    for _, da in variables.items():
        arr = da.values
        n_time = arr.shape[0]
        X_var = arr.transpose(1, 2, 0).reshape(n_lat * n_lon, n_time)
        X_var = imputar_nan_por_mediana_columna(X_var)
        X_parts.append(X_var)
        n_time_total += n_time

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

    scores_2d = raw_scores.reshape(n_lat, n_lon)
    anomalas_2d = (etiquetas == -1).reshape(n_lat, n_lon)

    scores_da = xr.DataArray(
        scores_2d,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="if_multi_score",
        attrs={"long_name": "Isolation Forest multivariate score", "variables": list(variables.keys())},
    )

    mascara_da = xr.DataArray(
        anomalas_2d,
        dims=["latitude", "longitude"],
        coords={"latitude": lats, "longitude": lons},
        name="if_multi_anomalia",
        attrs={"long_name": "Isolation Forest multivariate anomaly mask", "variables": list(variables.keys())},
    )

    n_anomalias = int(anomalas_2d.sum())
    n_total = n_lat * n_lon
    estadisticas = {
        "n_total_puntos": n_total,
        "num_anomalias": n_anomalias,
        "porcentaje_anomalias": round(100 * n_anomalias / n_total, 3),
        "score_min": float(np.nanmin(raw_scores)),
        "score_max": float(np.nanmax(raw_scores)),
        "score_media": float(np.nanmean(raw_scores)),
        "n_estimators": n_estimators,
        "contamination": contamination,
        "variables": list(variables.keys()),
        "n_features_total": n_time_total,
    }

    return scores_da, mascara_da, estadisticas

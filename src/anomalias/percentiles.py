"""
Detección de anomalías por Percentiles — solo precipitación (tp).

La precipitación tiene distribución zero-inflated (80 %+ de valores = 0 mm),
por lo que Isolation Forest no distingue bien lluvia intensa de moderada.
Aquí cada punto (lat, lon) tiene su propio umbral P99, evitando sesgos
climatológicos entre zonas.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def detectar_anomalias_percentiles(
    data: xr.DataArray,
    percentil: int = 99,
    min_threshold: float = 1e-3,
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """
    Detecta días de precipitación extrema usando el percentil P99 local.

    Para cada celda (lat, lon) calcula un umbral propio basado en su historia
    temporal, evitando sesgos por diferencias climatológicas entre zonas.

    Parameters
    ----------
    data : xr.DataArray
        Precipitación con dimensiones (valid_time, latitude, longitude) en mm.
    percentil : int
        Percentil a usar como umbral (por defecto 99).
    min_threshold : float
        Valor mínimo en mm por encima del cual se considera lluvia real.
        Evita que ceros con ruido numérico sean marcados como anómalos.

    Returns
    -------
    mascara : xr.DataArray bool (valid_time, latitude, longitude)
        True en los pares (tiempo, lat, lon) donde tp supera el umbral local.
    umbrales : xr.DataArray (latitude, longitude)
        Mapa de umbrales P{percentil} por punto espacial.
    estadisticas : dict
        Resumen del análisis.
    """
    required = {'valid_time', 'latitude', 'longitude'}
    missing = required - set(data.dims)
    if missing:
        raise ValueError(f"Faltan dimensiones en data: {missing}")

    lats = data.latitude.values
    lons = data.longitude.values
    n_time = len(data.valid_time)
    n_lat, n_lon = len(lats), len(lons)

    # data.values → (n_time, n_lat, n_lon)
    arr = data.values
    umbral_arr = np.nanpercentile(arr, percentil, axis=0)   # (n_lat, n_lon)
    # Mínimo para filtrar ceros con ruido numérico que no son lluvia real
    umbral_arr = np.maximum(umbral_arr, min_threshold)

    # Un punto es anómalo si supera su propio umbral local (no el global)
    anomalas_arr = arr > umbral_arr[np.newaxis, :, :]       # (n_time, n_lat, n_lon)

    umbrales_da = xr.DataArray(
        umbral_arr,
        dims=['latitude', 'longitude'],
        coords={'latitude': lats, 'longitude': lons},
        name='tp_umbral_p99',
        attrs={
            'long_name': f'Umbral P{percentil} local de precipitacion',
            'units': 'mm',
            'percentil': percentil,
        },
    )

    mascara_da = xr.DataArray(
        anomalas_arr,
        dims=['valid_time', 'latitude', 'longitude'],
        coords={
            'valid_time': data.valid_time.values,
            'latitude': lats,
            'longitude': lons,
        },
        name='tp_anomalia_p99',
        attrs={
            'long_name': f'Anomalia de precipitacion (tp > P{percentil} local)',
            'percentil': percentil,
        },
    )

    n_anomalias = int(anomalas_arr.sum())
    n_total = n_time * n_lat * n_lon
    dias_con_anomalia = int((anomalas_arr.sum(axis=(1, 2)) > 0).sum())
    puntos_con_anomalia = int((anomalas_arr.sum(axis=0) > 0).sum())

    estadisticas = {
        'n_total_observaciones': n_total,
        'num_anomalias': n_anomalias,
        'porcentaje_anomalias': round(100 * n_anomalias / n_total, 3),
        'dias_con_anomalia': dias_con_anomalia,
        'puntos_con_anomalia': puntos_con_anomalia,
        'percentil': percentil,
        'min_threshold': min_threshold,
        'umbral_min': float(np.nanmin(umbral_arr)),
        'umbral_max': float(np.nanmax(umbral_arr)),
        'umbral_media': float(np.nanmean(umbral_arr)),
        'n_timesteps': n_time,
        'n_puntos_espaciales': n_lat * n_lon,
    }

    return mascara_da, umbrales_da, estadisticas

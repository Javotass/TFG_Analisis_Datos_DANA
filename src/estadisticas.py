"""Módulo para cálculo de estadísticas descriptivas."""

import numpy as np
import pandas as pd
import xarray as xr


def calcular_dimensiones(ds: xr.Dataset) -> dict:
    """Extrae las dimensiones del dataset."""
    return {dim: int(size) for dim, size in ds.dims.items()}


def calcular_variables(ds: xr.Dataset) -> dict:
    """Extrae información de las variables del dataset."""
    variables = {}
    for var in ds.data_vars:
        attrs = ds[var].attrs
        variables[var] = {
            "nombre": attrs.get('long_name', 'N/A'),
            "unidades": attrs.get('units', 'N/A'),
            "dimensiones": list(ds[var].dims)
        }
    return variables


def calcular_coordenadas(ds: xr.Dataset) -> dict:
    """Extrae información de las coordenadas del dataset."""
    return {
        "latitud_min": float(ds.latitude.min().values),
        "latitud_max": float(ds.latitude.max().values),
        "longitud_min": float(ds.longitude.min().values),
        "longitud_max": float(ds.longitude.max().values),
        "tiempo_inicio": str(pd.to_datetime(ds.valid_time.min().values).date()),
        "tiempo_fin": str(pd.to_datetime(ds.valid_time.max().values).date()),
        "total_registros": int(len(ds.valid_time))
    }


def calcular_estadisticas_temperatura(ds: xr.Dataset) -> dict:
    """Calcula estadísticas de temperatura."""
    if 't2m' not in ds:
        return None
    
    temp_mean = ds['t2m'] - 273.15
    stats = temp_mean.mean(dim=['latitude', 'longitude'])
    
    return {
        "unidad": "°C",
        "media": round(float(stats.mean().values), 2),
        "minima": round(float(stats.min().values), 2),
        "maxima": round(float(stats.max().values), 2),
        "desviacion_estandar": round(float(stats.std().values), 2)
    }


def calcular_estadisticas_viento(ds: xr.Dataset) -> dict:
    """Calcula estadísticas de velocidad del viento."""
    if 'u10' not in ds or 'v10' not in ds:
        return None
    
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
    wind_stats = wind_speed.mean(dim=['latitude', 'longitude'])
    
    return {
        "unidad": "m/s",
        "media": round(float(wind_stats.mean().values), 2),
        "minima": round(float(wind_stats.min().values), 2),
        "maxima": round(float(wind_stats.max().values), 2),
        "desviacion_estandar": round(float(wind_stats.std().values), 2)
    }


def calcular_estadisticas_precipitacion(ds: xr.Dataset) -> dict:
    """Calcula estadísticas de precipitación."""
    if 'tp' not in ds:
        return None
    
    precip = ds['tp'] * 1000  # Convertir m a mm
    precip_stats = precip.mean(dim=['latitude', 'longitude'])
    
    return {
        "unidad": "mm",
        "media": round(float(precip_stats.mean().values), 4),
        "minima": round(float(precip_stats.min().values), 4),
        "maxima": round(float(precip_stats.max().values), 4),
        "total_acumulado": round(float(precip.sum().values), 2)
    }


def calcular_estadisticas_presion(ds: xr.Dataset) -> dict:
    """Calcula estadísticas de presión superficial."""
    if 'sp' not in ds:
        return None
    
    pressure = ds['sp'] / 100  # Convertir Pa a hPa
    pressure_stats = pressure.mean(dim=['latitude', 'longitude'])
    
    return {
        "unidad": "hPa",
        "media": round(float(pressure_stats.mean().values), 2),
        "minima": round(float(pressure_stats.min().values), 2),
        "maxima": round(float(pressure_stats.max().values), 2),
        "desviacion_estandar": round(float(pressure_stats.std().values), 2)
    }


def calcular_todas_estadisticas(ds: xr.Dataset) -> dict:
    """Calcula todas las estadísticas del dataset."""
    estadisticas = {}
    
    temp_stats = calcular_estadisticas_temperatura(ds)
    if temp_stats:
        estadisticas["temperatura_2m"] = temp_stats
    
    viento_stats = calcular_estadisticas_viento(ds)
    if viento_stats:
        estadisticas["velocidad_viento_10m"] = viento_stats
    
    precip_stats = calcular_estadisticas_precipitacion(ds)
    if precip_stats:
        estadisticas["precipitacion"] = precip_stats
    
    presion_stats = calcular_estadisticas_presion(ds)
    if presion_stats:
        estadisticas["presion_superficial"] = presion_stats
    
    return estadisticas


def calcular_valores_faltantes(ds: xr.Dataset) -> dict:
    """Detecta valores faltantes en el dataset."""
    valores_faltantes = {}
    for var in ds.data_vars:
        null_count = ds[var].isnull().sum().values
        total = ds[var].size
        pct = (null_count / total) * 100
        valores_faltantes[var] = {
            "cantidad": int(null_count),
            "total": int(total),
            "porcentaje": round(float(pct), 2)
        }
    return valores_faltantes


def calcular_correlaciones(ds: xr.Dataset) -> dict:
    """Calcula la matriz de correlación entre variables."""
    df_corr = pd.DataFrame()
    
    if 't2m' in ds:
        df_corr['Temperatura'] = (ds['t2m'] - 273.15).mean(dim=['latitude', 'longitude']).values
    
    if 'u10' in ds and 'v10' in ds:
        df_corr['Velocidad Viento'] = np.sqrt(ds['u10']**2 + ds['v10']**2).mean(dim=['latitude', 'longitude']).values
    
    if 'tp' in ds:
        df_corr['Precipitación'] = (ds['tp'] * 1000).mean(dim=['latitude', 'longitude']).values
    
    if 'sp' in ds:
        df_corr['Presión'] = (ds['sp'] / 100).mean(dim=['latitude', 'longitude']).values
    
    if df_corr.empty:
        return {}
    
    return df_corr.corr().to_dict()

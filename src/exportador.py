"""Módulo para exportar datos a CSV y JSON."""

import json
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime


def exportar_serie_temporal_csv(ds: xr.Dataset, ruta_csv: str) -> str:
    """Exporta las series temporales promediadas a CSV."""
    df_series = pd.DataFrame()
    df_series['valid_time'] = pd.to_datetime(ds.valid_time.values)
    
    if 't2m' in ds:
        df_series['temperatura_2m'] = (ds['t2m'] - 273.15).mean(dim=['latitude', 'longitude']).values
    
    if 'u10' in ds and 'v10' in ds:
        df_series['velocidad_viento'] = np.sqrt(ds['u10']**2 + ds['v10']**2).mean(dim=['latitude', 'longitude']).values
    
    if 'tp' in ds:
        df_series['precipitacion'] = (ds['tp'] * 1000).mean(dim=['latitude', 'longitude']).values
    
    if 'sp' in ds:
        df_series['presion_superficial'] = (ds['sp'] / 100).mean(dim=['latitude', 'longitude']).values
    
    df_series = df_series.set_index('valid_time')
    df_series.to_csv(ruta_csv)
    
    return ruta_csv


def guardar_resultados_json(resultados: dict, ruta_json: str) -> str:
    """Guarda los resultados del análisis en un archivo JSON."""
    resultados["estado"] = "completado"
    resultados["fecha_finalizacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(ruta_json, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    
    return ruta_json

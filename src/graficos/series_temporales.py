"""Módulo para generación de gráficos de series temporales."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def generar_serie_temperatura(ds: xr.Dataset, output_dir: Path) -> str:
    """
    Genera gráfico de serie temporal de temperatura.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 't2m' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(16, 6))
    temp = (ds['t2m'] - 273.15).mean(dim=['latitude', 'longitude'])
    temp.plot(ax=ax, color='orangered', linewidth=0.8)
    ax.set_title('Serie Temporal - Temperatura a 2m', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temperatura (°C)')
    ax.set_xlabel('Fecha')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'serie_temperatura.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_serie_viento(ds: xr.Dataset, output_dir: Path) -> str:
    """
    Genera gráfico de serie temporal de velocidad del viento.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'u10' not in ds or 'v10' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(16, 6))
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2).mean(dim=['latitude', 'longitude'])
    wind_speed.plot(ax=ax, color='steelblue', linewidth=0.8)
    ax.set_title('Serie Temporal - Velocidad del Viento a 10m', fontsize=14, fontweight='bold')
    ax.set_ylabel('Velocidad (m/s)')
    ax.set_xlabel('Fecha')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'serie_viento.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_serie_precipitacion(ds: xr.Dataset, output_dir: Path) -> str:
    """
    Genera gráfico de serie temporal de precipitación.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'tp' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(16, 6))
    precip = (ds['tp'] * 1000).mean(dim=['latitude', 'longitude'])
    precip.plot(ax=ax, color='darkblue', linewidth=0.8)
    ax.set_title('Serie Temporal - Precipitación Total', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precipitación (mm)')
    ax.set_xlabel('Fecha')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'serie_precipitacion.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_serie_presion(ds: xr.Dataset, output_dir: Path) -> str:
    """
    Genera gráfico de serie temporal de presión superficial.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'sp' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(16, 6))
    pressure = (ds['sp'] / 100).mean(dim=['latitude', 'longitude'])
    pressure.plot(ax=ax, color='purple', linewidth=0.8)
    ax.set_title('Serie Temporal - Presión Superficial', fontsize=14, fontweight='bold')
    ax.set_ylabel('Presión (hPa)')
    ax.set_xlabel('Fecha')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'serie_presion.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_series_temporales(ds: xr.Dataset, output_dir: Path) -> List[str]:
    """
    Genera gráficos de series temporales de todas las variables (cada uno por separado).
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida para los gráficos.
        
    Returns:
        Lista de rutas de archivos guardados.
    """
    rutas = []
    
    # Temperatura
    ruta = generar_serie_temperatura(ds, output_dir)
    if ruta:
        rutas.append(ruta)
    
    # Velocidad del viento
    ruta = generar_serie_viento(ds, output_dir)
    if ruta:
        rutas.append(ruta)
    
    # Precipitación
    ruta = generar_serie_precipitacion(ds, output_dir)
    if ruta:
        rutas.append(ruta)
    
    # Presión
    ruta = generar_serie_presion(ds, output_dir)
    if ruta:
        rutas.append(ruta)
    
    return rutas

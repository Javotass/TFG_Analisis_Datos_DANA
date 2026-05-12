"""Módulo para generación de histogramas de distribución."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def generar_distribucion_temperatura(ds: xr.Dataset, output_dir: Path) -> str:
    """
    Genera histograma de distribución de temperatura.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 't2m' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    temp_flat = (ds['t2m'] - 273.15).values.flatten()
    temp_flat = temp_flat[~np.isnan(temp_flat)]
    
    ax.hist(temp_flat, bins=50, color='orangered', alpha=0.7, edgecolor='black')
    ax.axvline(temp_flat.mean(), color='darkred', linestyle='--', linewidth=2,
               label=f'Media: {temp_flat.mean():.2f}°C')
    ax.set_title('Distribución de Temperatura 2m', fontsize=14, fontweight='bold')
    ax.set_xlabel('Temperatura (°C)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'distribucion_temperatura.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_distribucion_viento(ds: xr.Dataset, output_dir: Path) -> str:
    """
    Genera histograma de distribución de velocidad del viento.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'u10' not in ds or 'v10' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    wind_flat = np.sqrt(ds['u10']**2 + ds['v10']**2).values.flatten()
    wind_flat = wind_flat[~np.isnan(wind_flat)]
    
    ax.hist(wind_flat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(wind_flat.mean(), color='darkblue', linestyle='--', linewidth=2,
               label=f'Media: {wind_flat.mean():.2f} m/s')
    ax.set_title('Distribución de Velocidad del Viento 10m', fontsize=14, fontweight='bold')
    ax.set_xlabel('Velocidad (m/s)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'distribucion_viento.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_distribucion_precipitacion(ds: xr.Dataset, output_dir: Path) -> str:
    """
    Genera histograma de distribución de precipitación.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'tp' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    precip_flat = (ds['tp'] * 1000).values.flatten()
    precip_flat = precip_flat[~np.isnan(precip_flat)]
    precip_flat = precip_flat[precip_flat > 0]
    
    if len(precip_flat) == 0:
        plt.close()
        return None
    
    ax.hist(precip_flat, bins=50, color='darkblue', alpha=0.7, edgecolor='black')
    ax.axvline(precip_flat.mean(), color='navy', linestyle='--', linewidth=2,
               label=f'Media: {precip_flat.mean():.4f} mm')
    ax.set_title('Distribución de Precipitación Total (valores > 0)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Precipitación (mm)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'distribucion_precipitacion.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_distribucion_presion(ds: xr.Dataset, output_dir: Path) -> str:
    """
    Genera histograma de distribución de presión superficial.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'sp' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    pressure_flat = (ds['sp'] / 100).values.flatten()
    pressure_flat = pressure_flat[~np.isnan(pressure_flat)]
    
    ax.hist(pressure_flat, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(pressure_flat.mean(), color='indigo', linestyle='--', linewidth=2,
               label=f'Media: {pressure_flat.mean():.2f} hPa')
    ax.set_title('Distribución de Presión Superficial', fontsize=14, fontweight='bold')
    ax.set_xlabel('Presión (hPa)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'distribucion_presion.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_distribuciones(ds: xr.Dataset, output_dir: Path) -> List[str]:
    """
    Genera histogramas de distribución de variables (cada uno por separado).
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Lista de rutas de archivos guardados.
    """
    rutas = []
    
    # Temperatura
    ruta = generar_distribucion_temperatura(ds, output_dir)
    if ruta:
        rutas.append(ruta)
    
    # Velocidad del viento
    ruta = generar_distribucion_viento(ds, output_dir)
    if ruta:
        rutas.append(ruta)
    
    # Precipitación
    ruta = generar_distribucion_precipitacion(ds, output_dir)
    if ruta:
        rutas.append(ruta)
    
    # Presión
    ruta = generar_distribucion_presion(ds, output_dir)
    if ruta:
        rutas.append(ruta)
    
    return rutas

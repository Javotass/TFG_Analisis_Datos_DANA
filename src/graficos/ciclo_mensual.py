"""Módulo para generación de gráficos de ciclo mensual."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import calendar


def generar_ciclo_mensual_temperatura(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> str:
    """
    Genera gráfico del ciclo mensual de temperatura.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio.
        fecha_fin: Fecha de fin.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 't2m' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    meses_nombres = {i: calendar.month_abbr[i] for i in range(1, 13)}
    
    temp_monthly = (ds['t2m'] - 273.15).mean(dim=['latitude', 'longitude']).groupby('valid_time.month').mean()
    months_in_data = temp_monthly.month.values
    
    ax.plot(months_in_data, temp_monthly.values, marker='o', linewidth=2, markersize=8, color='orangered')
    ax.set_title(f'Ciclo Mensual - Temperatura 2m\n({fecha_inicio} - {fecha_fin})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temperatura (°C)')
    ax.set_xlabel('Mes')
    ax.set_xticks(months_in_data)
    ax.set_xticklabels([meses_nombres[m] for m in months_in_data])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'ciclo_mensual_temperatura.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_ciclo_mensual_viento(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> str:
    """
    Genera gráfico del ciclo mensual de velocidad del viento.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio.
        fecha_fin: Fecha de fin.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'u10' not in ds or 'v10' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    meses_nombres = {i: calendar.month_abbr[i] for i in range(1, 13)}
    
    wind_monthly = np.sqrt(ds['u10']**2 + ds['v10']**2).mean(dim=['latitude', 'longitude']).groupby('valid_time.month').mean()
    months_in_data = wind_monthly.month.values
    
    ax.plot(months_in_data, wind_monthly.values, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.set_title(f'Ciclo Mensual - Velocidad del Viento\n({fecha_inicio} - {fecha_fin})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Velocidad (m/s)')
    ax.set_xlabel('Mes')
    ax.set_xticks(months_in_data)
    ax.set_xticklabels([meses_nombres[m] for m in months_in_data])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'ciclo_mensual_viento.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_ciclo_mensual_precipitacion(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> str:
    """
    Genera gráfico del ciclo mensual de precipitación.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio.
        fecha_fin: Fecha de fin.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'tp' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    meses_nombres = {i: calendar.month_abbr[i] for i in range(1, 13)}
    
    precip_monthly = (ds['tp'] * 1000).mean(dim=['latitude', 'longitude']).groupby('valid_time.month').sum()
    months_in_data = precip_monthly.month.values
    
    ax.bar(months_in_data, precip_monthly.values, color='darkblue', alpha=0.7)
    ax.set_title(f'Ciclo Mensual - Precipitación Acumulada\n({fecha_inicio} - {fecha_fin})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precipitación (mm)')
    ax.set_xlabel('Mes')
    ax.set_xticks(months_in_data)
    ax.set_xticklabels([meses_nombres[m] for m in months_in_data])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    ruta = str(output_dir / 'ciclo_mensual_precipitacion.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_ciclo_mensual_presion(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> str:
    """
    Genera gráfico del ciclo mensual de presión superficial.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio.
        fecha_fin: Fecha de fin.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'sp' not in ds:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    meses_nombres = {i: calendar.month_abbr[i] for i in range(1, 13)}
    
    pressure_monthly = (ds['sp'] / 100).mean(dim=['latitude', 'longitude']).groupby('valid_time.month').mean()
    months_in_data = pressure_monthly.month.values
    
    ax.plot(months_in_data, pressure_monthly.values, marker='o', linewidth=2, markersize=8, color='purple')
    ax.set_title(f'Ciclo Mensual - Presión Superficial\n({fecha_inicio} - {fecha_fin})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Presión (hPa)')
    ax.set_xlabel('Mes')
    ax.set_xticks(months_in_data)
    ax.set_xticklabels([meses_nombres[m] for m in months_in_data])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    ruta = str(output_dir / 'ciclo_mensual_presion.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_ciclo_mensual(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> List[str]:
    """
    Genera gráficos del ciclo mensual (cada uno por separado).
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio.
        fecha_fin: Fecha de fin.
        
    Returns:
        Lista de rutas de archivos guardados.
    """
    rutas = []
    
    # Temperatura
    ruta = generar_ciclo_mensual_temperatura(ds, output_dir, fecha_inicio, fecha_fin)
    if ruta:
        rutas.append(ruta)
    
    # Viento
    ruta = generar_ciclo_mensual_viento(ds, output_dir, fecha_inicio, fecha_fin)
    if ruta:
        rutas.append(ruta)
    
    # Precipitación
    ruta = generar_ciclo_mensual_precipitacion(ds, output_dir, fecha_inicio, fecha_fin)
    if ruta:
        rutas.append(ruta)
    
    # Presión
    ruta = generar_ciclo_mensual_presion(ds, output_dir, fecha_inicio, fecha_fin)
    if ruta:
        rutas.append(ruta)
    
    return rutas

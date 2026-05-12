"""Módulo para generación de matriz de correlación."""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def generar_matriz_correlacion(ds: xr.Dataset, output_dir: Path) -> str:
    """
    Genera la matriz de correlación.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
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
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df_corr.corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={'label': 'Correlación'}, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title('Matriz de Correlación - Variables Meteorológicas', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    ruta = str(output_dir / 'correlacion_variables.svg')
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta

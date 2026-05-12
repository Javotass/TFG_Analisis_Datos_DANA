"""Módulo para generación de mapas de valores medios."""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from typing import List
from shapely.geometry import LineString, MultiLineString, Polygon


def _generar_mapa_base(data: xr.DataArray, output_dir: Path, fecha_inicio: str, fecha_fin: str,
                       titulo: str, nombre_archivo: str, label_colorbar: str, 
                       cmap: str = 'viridis', agregacion: str = 'mean') -> str:
    """
    Función base para generar mapas (evita código duplicado).
    
    Args:
        data: DataArray con los datos a visualizar.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio del período.
        fecha_fin: Fecha de fin del período.
        titulo: Título base del gráfico.
        nombre_archivo: Nombre del archivo PNG a guardar.
        label_colorbar: Etiqueta para la barra de color.
        cmap: Mapa de colores a usar.
        agregacion: Tipo de agregación ('mean' o 'sum').
        
    Returns:
        Ruta del archivo guardado.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Aplicar agregación temporal
    if agregacion == 'mean':
        data_agg = data.mean(dim='valid_time')
    elif agregacion == 'sum':
        data_agg = data.sum(dim='valid_time')
    else:
        raise ValueError(f"Agregación no soportada: {agregacion}")
    
    # Generar plot
    data_agg.plot(ax=ax, cmap=cmap, add_colorbar=True,
                  cbar_kwargs={'label': label_colorbar})
    
    # Configurar título con estadística
    valor_medio = data_agg.mean().values
    ax.set_title(f'{titulo} - Media: {valor_medio:.2f} {label_colorbar.split("(")[-1].strip(")")}\n({fecha_inicio} - {fecha_fin})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    
    # Guardar
    plt.tight_layout()
    ruta = str(output_dir / nombre_archivo)
    plt.savefig(ruta, bbox_inches='tight')
    plt.close()
    
    return ruta


def generar_mapa_temperatura(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> str:
    """
    Genera mapa de temperatura media.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio del período.
        fecha_fin: Fecha de fin del período.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 't2m' not in ds:
        return None
    
    temp_celsius = ds['t2m'] - 273.15
    return _generar_mapa_base(
        data=temp_celsius,
        output_dir=output_dir,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        titulo='Temperatura 2m',
        nombre_archivo='mapa_temperatura.svg',
        label_colorbar='Temperatura (°C)',
        cmap='RdYlBu_r',
        agregacion='mean'
    )


def generar_mapa_viento(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> str:
    """
    Genera mapa de velocidad del viento media.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio del período.
        fecha_fin: Fecha de fin del período.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'u10' not in ds or 'v10' not in ds:
        return None
    
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
    return _generar_mapa_base(
        data=wind_speed,
        output_dir=output_dir,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        titulo='Velocidad Viento',
        nombre_archivo='mapa_viento.svg',
        label_colorbar='Velocidad (m/s)',
        cmap='YlGnBu',
        agregacion='mean'
    )


def generar_mapa_precipitacion(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> str:
    """
    Genera mapa de precipitación acumulada.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio del período.
        fecha_fin: Fecha de fin del período.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'tp' not in ds:
        return None
    
    precip_mm = ds['tp'] * 1000
    ruta = _generar_mapa_base(
        data=precip_mm,
        output_dir=output_dir,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        titulo='Precipitación Acumulada',
        nombre_archivo='mapa_precipitacion.svg',
        label_colorbar='Precipitación (mm)',
        cmap='Blues',
        agregacion='sum'
    )
    # Actualizar título para mostrar "Total" en lugar de "Media"
    return ruta


def generar_mapa_presion(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> str:
    """
    Genera mapa de presión superficial media.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio del período.
        fecha_fin: Fecha de fin del período.
        
    Returns:
        Ruta del archivo guardado o None si no hay datos.
    """
    if 'sp' not in ds:
        return None
    
    pressure_hpa = ds['sp'] / 100
    return _generar_mapa_base(
        data=pressure_hpa,
        output_dir=output_dir,
        fecha_inicio=fecha_inicio,
        fecha_fin=fecha_fin,
        titulo='Presión Superficial',
        nombre_archivo='mapa_presion.svg',
        label_colorbar='Presión (hPa)',
        cmap='viridis',
        agregacion='mean'
    )


def generar_mapas_medios(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> List[str]:
    """
    Genera mapas de valores promedio (cada uno por separado).
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio del período.
        fecha_fin: Fecha de fin del período.
        
    Returns:
        Lista de rutas de archivos guardados.
    """
    rutas = []
    
    # Temperatura
    ruta = generar_mapa_temperatura(ds, output_dir, fecha_inicio, fecha_fin)
    if ruta:
        rutas.append(ruta)
    
    # Velocidad del viento
    ruta = generar_mapa_viento(ds, output_dir, fecha_inicio, fecha_fin)
    if ruta:
        rutas.append(ruta)
    
    # Precipitación
    ruta = generar_mapa_precipitacion(ds, output_dir, fecha_inicio, fecha_fin)
    if ruta:
        rutas.append(ruta)
    
    # Presión
    ruta = generar_mapa_presion(ds, output_dir, fecha_inicio, fecha_fin)
    if ruta:
        rutas.append(ruta)
    
    return rutas


def generar_mapas_animados(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> List[str]:
    """
    Genera mapas animados HTML para cada variable meteorológica.
    
    Args:
        ds: Dataset de xarray.
        output_dir: Directorio de salida.
        fecha_inicio: Fecha de inicio del período.
        fecha_fin: Fecha de fin del período.
        
    Returns:
        Lista de rutas de archivos HTML generados.
    """
    rutas = []
    
    # Mapeo de variables y sus configuraciones
    variables_config = {
        't2m': {
            'param': 'temperatura',
            'title': f'Temperatura 2m ({fecha_inicio} - {fecha_fin})',
            'file_name': 'mapa_animado_temperatura.html',
            'conversion': lambda x: x - 273.15  # K a °C
        },
        'tp': {
            'param': 'precipitacion',
            'title': f'Precipitación Total ({fecha_inicio} - {fecha_fin})',
            'file_name': 'mapa_animado_precipitacion.html',
            'conversion': lambda x: x * 1000  # m a mm
        },
        'sp': {
            'param': 'presion',
            'title': f'Presión Superficial ({fecha_inicio} - {fecha_fin})',
            'file_name': 'mapa_animado_presion.html',
            'conversion': lambda x: x / 100  # Pa a hPa
        }
    }
    
    # Variable compuesta: velocidad del viento
    if 'u10' in ds and 'v10' in ds:
        variables_config['wind'] = {
            'param': 'velocidad_viento',
            'title': f'Velocidad del Viento 10m ({fecha_inicio} - {fecha_fin})',
            'file_name': 'mapa_animado_viento.html',
            'is_composite': True
        }
    
    # Generar cada mapa animado
    for var_key, config in variables_config.items():
        try:
            # Caso especial: velocidad del viento (variable compuesta)
            if config.get('is_composite'):
                if 'u10' in ds and 'v10' in ds:
                    # Calcular velocidad del viento
                    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
                    df = _xarray_to_dataframe(wind_speed, config['param'])
            else:
                # Variables simples
                if var_key not in ds:
                    continue
                
                # Aplicar conversión si existe
                data = ds[var_key]
                if 'conversion' in config:
                    data = config['conversion'](data)
                
                df = _xarray_to_dataframe(data, config['param'])
            
            # Generar mapa animado
            path_to_save = str(output_dir) + '/'
            plot_animated_map(
                data=df,
                path_to_save=path_to_save,
                file_name=config['file_name'],
                param=config['param'],
                title=config['title'],
                coastline_file=None  # Opcional: agregar ruta si se tiene shapefile
            )
            
            ruta_completa = str(output_dir / config['file_name'])
            rutas.append(ruta_completa)
            
        except Exception as e:
            print(f"  [ERROR] No se pudo generar mapa animado para {var_key}: {e}")
            continue
    
    return rutas


def _xarray_to_dataframe(data_array: xr.DataArray, param_name: str) -> pd.DataFrame:
    """
    Convierte un DataArray de xarray al formato requerido por plot_animated_map.
    
    Args:
        data_array: DataArray con dimensiones (valid_time, latitude, longitude).
        param_name: Nombre del parámetro para la columna de valores.
        
    Returns:
        DataFrame con columnas: time, latitude, longitude, [param_name]
    """
    # Convertir a DataFrame
    df = data_array.to_dataframe(name=param_name).reset_index()
    
    # Renombrar columnas al formato esperado
    df = df.rename(columns={'valid_time': 'time'})
    
    # Eliminar NaN si existen
    df = df.dropna(subset=[param_name])
    
    return df


def plot_animated_map(data, path_to_save, file_name, param, title="", coastline_file='./collection/data/ne_10m_coastline.shp'):
    """
    Crea un mapa animado (heatmap) a partir de datos con coordenadas temporales.
    
    Parámetros:
    -----------
    data : pandas.DataFrame
        DataFrame con columnas: 'time', 'latitude', 'longitude' y el parámetro a visualizar
    path_to_save : str
        Ruta donde guardar el archivo HTML generado (debe terminar en '/')
    file_name : str
        Nombre del archivo HTML a crear (ej: 'mapa_animado.html')
    param : str
        Nombre de la columna que contiene los valores a visualizar en el mapa de calor
    title : str, opcional
        Título del gráfico
    coastline_file : str, opcional
        Ruta al archivo shapefile de línea de costa (por defecto usa ne_10m_coastline.shp)
    
    Retorna:
    --------
    None
        Guarda el mapa animado como archivo HTML
    
    Ejemplo de uso:
    --------------
    >>> df = pd.DataFrame({
    ...     'time': ['2024-01-01', '2024-01-02'],
    ...     'latitude': [40.0, 41.0],
    ...     'longitude': [-3.0, -2.0],
    ...     'temperatura': [20.5, 21.3]
    ... })
    >>> plot_animated_map(df, './', 'mi_mapa.html', 'temperatura', title='Temperatura')
    """
    # Eliminar duplicados en la misma ubicación y tiempo
    data = data.drop_duplicates(subset=['time', 'latitude', 'longitude'])

    # Obtener todos los timestamps únicos ordenados
    timestamps = sorted(data['time'].unique())
   
    # Obtener latitudes y longitudes únicas ordenadas
    lats = np.sort(data['latitude'].unique())
    lons = np.sort(data['longitude'].unique())
   
    # Crear frames para la animación (uno por cada timestamp)
    frames = []
    for ts in timestamps:
        # Filtrar datos para este timestamp y crear matriz pivot
        df_t = data[data['time'] == ts].pivot(index='latitude', columns='longitude', values=param).reindex(index=lats, columns=lons)
        z = df_t.values
        
        # Crear frame con heatmap
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=z, 
                x=lons, 
                y=lats, 
                colorscale='RdBu_r',  # Escala de colores rojo-azul invertida
                zmid=0,
                zmin=data[param].min(), 
                zmax=data[param].max(),
                colorbar=dict(len=1.05, y=0.5)
            )],
            name=str(ts)
        ))

    # Crear pasos del slider para cada timestamp
    slider_steps = [dict(
        method='animate',
        args=[[str(ts)], dict(mode='immediate', frame=dict(duration=500, redraw=True), transition=dict(duration=0))],
        label=str(ts)
    ) for ts in timestamps]

    # Crear figura principal
    fig = go.Figure(
        data=frames[0].data,  # Empezar con el primer frame
        frames=frames,
        layout=go.Layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            height=900,
            width=1000,
            autosize=False,
            xaxis=dict(
                range=[lons.min(), lons.max()],
                scaleanchor="y",
                scaleratio=1,
                constrain='domain',
                title="Longitud",
                showgrid=False
            ),
            yaxis=dict(
                range=[lats.min(), lats.max()],
                constrain='domain',
                title="Latitud",
                showgrid=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=100, r=200, t=100, b=150, autoexpand=True),
           
            # Botones de play y pause
            updatemenus=[
                # Botón Play
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.49, y=-0.10,  
                    xanchor="right",
                    yanchor="top",
                    buttons=[dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True, 
                            "mode": "immediate"
                        }]
                    )]
                ),
                # Botón Pause
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.51, y=-0.10,
                    xanchor="left",
                    yanchor="top",
                    buttons=[dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False}, 
                            "mode": "immediate"
                        }]
                    )]
                )
            ],
           
            # Slider temporal
            sliders=[dict(
                active=0,
                y=-0.15,      
                x=0.1,        
                len=0.8,      
                xanchor='left',
                pad=dict(t=40, b=10),
                currentvalue=dict(
                    prefix="Fecha: ",
                    visible=True,
                    xanchor="center"
                ),
                steps=slider_steps
            )]
        )
    )

    # Añadir líneas de costa si se proporciona el archivo
    if coastline_file:
        coast = gpd.read_file(coastline_file)
        
        # Crear polígono del bounding box de los datos
        bbox_poly = Polygon([
            (lons.min(), lats.min()),
            (lons.max(), lats.min()),
            (lons.max(), lats.max()),
            (lons.min(), lats.max())
        ])
        
        # Recortar líneas de costa al área de datos
        coast_strict = coast.geometry.apply(lambda g: g.intersection(bbox_poly))
        coast_strict = coast_strict.explode(index_parts=False)

        # Añadir cada segmento de línea de costa al gráfico
        for geom in coast_strict:
            if geom.is_empty:
                continue
            if isinstance(geom, LineString):
                x, y = list(geom.xy[0]), list(geom.xy[1])
                fig.add_trace(go.Scatter(
                    x=x, y=y, 
                    mode='lines',
                    line=dict(color='black', width=1), 
                    showlegend=False
                ))
            elif isinstance(geom, MultiLineString):
                for line in geom:
                    x, y = list(line.xy[0]), list(line.xy[1])
                    fig.add_trace(go.Scatter(
                        x=x, y=y, 
                        mode='lines',
                        line=dict(color='black', width=1), 
                        showlegend=False
                    ))

    # Guardar como archivo HTML
    fig.write_html(path_to_save + file_name)
    print(f"Mapa animado guardado en: {path_to_save + file_name}")

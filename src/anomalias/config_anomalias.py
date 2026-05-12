"""
Configuracion de parametros para deteccion de anomalias.

Algoritmos:
  - Isolation Forest: t2m, wind_speed, sp (univariante) + multivariante
  - Percentiles p99 : tp (precipitacion zero-inflated)
"""

# ---------------------------------------------------------------------------
# Metadatos de cada variable
# ---------------------------------------------------------------------------
CONFIG_ANOMALIAS = {
    't2m': {
        'nombre': 'Temperatura 2m',
        'unidad': 'grados C',
    },
    'wind_speed': {
        'nombre': 'Velocidad del Viento',
        'unidad': 'm/s',
    },
    'tp': {
        'nombre': 'Precipitacion',
        'unidad': 'mm',
    },
    'sp': {
        'nombre': 'Presion Superficial',
        'unidad': 'hPa',
    },
}

# ---------------------------------------------------------------------------
# Isolation Forest - variables univariantes (t2m, wind_speed, sp)
# Cada punto (lat, lon) es una muestra; sus N timesteps son las features.
# Salida: mapa espacial (lat, lon) con score de anomalia y mascara binaria.
# ---------------------------------------------------------------------------
CONFIG_ISOLATION_FOREST = {
    'n_estimators': 200,
    # 0.05: 5% de los puntos son considerados anomalos (motivado fisicamente:
    # zonas de alta montana en Espana como Pirineos, Sierra Nevada y
    # Sistema Central representan aproximadamente ese porcentaje del dominio).
    'contamination': 0.05,
    'random_state': 42,
    'n_jobs': -1,
    # Criterio complementario: z-score sobre la media espacial fusionado con IF
    # para evitar depender solo de extremos puros.
    'usar_criterio_media': True,
    'umbral_zscore_media': 2.0,
    'combinar_criterios': 'or',
    'multivariante': {
        'n_estimators': 300,
        'contamination': 0.05,
        'random_state': 42,
        'n_jobs': -1,
        'variables': ['t2m', 'wind_speed', 'sp'],
    },
}

# ---------------------------------------------------------------------------
# Percentiles - solo tp (precipitacion zero-inflated)
# Para cada punto (lat, lon) se calcula el p99 sobre sus N dias.
# Un dia es anomalo si tp > umbral_local.
# Salida: cubo booleano (time, lat, lon).
# ---------------------------------------------------------------------------
CONFIG_PERCENTILES = {
    'tp': {
        'percentil': 99,
        'min_threshold': 1e-3,  # mm minimos para considerar lluvia real
    },
}

# ---------------------------------------------------------------------------
# Isolation Forest TEMPORAL - detecta eventos meteorologicos anomalos
# Cada timestep es una muestra; los N puntos espaciales son las features.
# Salida: serie temporal (valid_time,) con mascara de timesteps anomalos.
# ---------------------------------------------------------------------------
CONFIG_IF_TEMPORAL = {
    'n_estimators': 200,
    # 0.05: ~9 timesteps de 182 considerados eventos atipicos.
    # Con solo 6 semanas de datos el umbral es conservador; con datos
    # multianuales se podria reducir a 0.01-0.02.
    'contamination': 0.05,
    'random_state': 42,
    'n_jobs': -1,
    'usar_criterio_media': True,
    'umbral_zscore_media': 2.0,
    'combinar_criterios': 'or',
    'variables': ['t2m', 'wind_speed', 'sp', 'tp'],
    'multivariante': {
        'n_estimators': 300,
        'contamination': 0.05,
        'random_state': 42,
        'n_jobs': -1,
        'variables': ['t2m', 'wind_speed', 'sp'],
    },
}

# ---------------------------------------------------------------------------
# Ecuacion en diferencias de primer orden (temporal)
# x_t = a*x_(t-1) + b
# Se ajusta sobre la media espacial por timestep y se marcan anomalias por
# z-score de residuo.
# ---------------------------------------------------------------------------
CONFIG_FOD = {
    'umbral_zscore_residuo': 2.5,
    # Permite ajustar sensibilidad por variable manteniendo un valor base.
    'umbral_zscore_residuo_por_variable': {
        't2m': 2.0,
    },
    'min_puntos_ajuste': 10,
    'variables': ['t2m', 'wind_speed', 'sp', 'tp'],
}

# ---------------------------------------------------------------------------
# Autoencoder temporal
# Aprende patron normal por reconstruccion sobre features dinamicas por
# timestep y marca anomalias por error alto.
# ---------------------------------------------------------------------------
CONFIG_AUTOENCODER = {
    'variables': ['t2m', 'wind_speed', 'sp', 'tp'],
    'hidden_layer_sizes': [16, 6, 16],
    'contamination': 0.05,
    'random_state': 42,
    'max_iter': 2000,
}

# ---------------------------------------------------------------------------
# Configuracion general
# ---------------------------------------------------------------------------
CONFIG_GENERAL = {
    'visualizacion': {
        'guardar_graficos': True,
        'formato_graficos': 'svg',
        'dpi': 300,
    },
}


def obtener_config_isolation_forest(multivariante=False):
    """Devuelve los parametros de Isolation Forest (multivariante si multivariante=True)."""
    if multivariante:
        return CONFIG_ISOLATION_FOREST['multivariante']
    return {k: v for k, v in CONFIG_ISOLATION_FOREST.items() if k != 'multivariante'}


def obtener_config_isolation_forest_temporal():
    """Devuelve los parametros del Isolation Forest temporal."""
    return dict(CONFIG_IF_TEMPORAL)


def obtener_config_percentiles(variable='tp'):
    """Devuelve los parametros de Percentiles para la variable indicada."""
    if variable not in CONFIG_PERCENTILES:
        raise ValueError(f"Variable '{variable}' no tiene configuracion de percentiles")
    return CONFIG_PERCENTILES[variable]


def obtener_config_fod():
    """Devuelve los parametros de la ecuacion de primer orden temporal."""
    return dict(CONFIG_FOD)


def obtener_config_autoencoder():
    """Devuelve los parametros del autoencoder temporal."""
    return dict(CONFIG_AUTOENCODER)

"""
Modulo de deteccion de anomalias.

Algoritmos implementados
------------------------
Isolation Forest (univariante)
    t2m, wind_speed, sp.
    Cada punto geografico (lat, lon) es una muestra; sus N timesteps son las
    features.  Detecta lugares cuyo comportamiento temporal es atipico.
    Salida: mapa espacial (lat, lon).

Isolation Forest (multivariante)
    Todas las variables juntas (t2m, wind_speed, sp).
    Detecta lugares donde la *combinacion* entre variables es anomala.
    Salida: mapa espacial (lat, lon).

Isolation Forest (temporal)
    Cada timestep es una muestra; los N puntos espaciales son las features.
    Detecta *eventos meteorologicos* cuyo patron espacial es atipico respecto
    al resto de pasos temporales.
    Salida: serie temporal (valid_time,).

Percentiles p99 (tp)
    La precipitacion tiene distribucion zero-inflated; Isolation Forest falla
    al distinguir lluvia intensa de lluvia moderada.
    Para cada punto (lat, lon) se calcula su umbral P99 local.
    Salida: cubo espacio-temporal (valid_time, lat, lon).
"""

from .isolation_forest import (
    detectar_anomalias_isolation_forest,
    detectar_anomalias_isolation_forest_multivariante,
    detectar_anomalias_isolation_forest_temporal,
    detectar_anomalias_isolation_forest_temporal_multivariante,
)

from .percentiles import (
    detectar_anomalias_percentiles,
)

from .diferencias_primer_orden import (
    detectar_anomalias_diferencias_primer_orden_temporal,
)

from .autoencoder_temporal import (
    detectar_anomalias_autoencoder_temporal,
)

from .config_anomalias import (
    CONFIG_ANOMALIAS,
    CONFIG_ISOLATION_FOREST,
    CONFIG_IF_TEMPORAL,
    CONFIG_PERCENTILES,
    CONFIG_FOD,
    CONFIG_AUTOENCODER,
    CONFIG_GENERAL,
    obtener_config_isolation_forest,
    obtener_config_isolation_forest_temporal,
    obtener_config_percentiles,
    obtener_config_fod,
    obtener_config_autoencoder,
)

from .visualizacion import (
    visualizar_mapa_scores_if,
    visualizar_mapa_anomalias_if,
    visualizar_anomalias_tp_percentiles,
    visualizar_resumen_multivariante,
    visualizar_serie_temporal_if,
    visualizar_distribucion_scores_if,
    visualizar_serie_temporal_if_temporal,
    visualizar_serie_temporal_fod,
    visualizar_serie_temporal_autoencoder,
    visualizar_resumen_temporal_multivariante,
)

__all__ = [
    # Isolation Forest
    "detectar_anomalias_isolation_forest",
    "detectar_anomalias_isolation_forest_multivariante",
    "detectar_anomalias_isolation_forest_temporal",
    "detectar_anomalias_isolation_forest_temporal_multivariante",
    "detectar_anomalias_diferencias_primer_orden_temporal",
    "detectar_anomalias_autoencoder_temporal",
    # Percentiles
    "detectar_anomalias_percentiles",
    # Config
    "CONFIG_ANOMALIAS",
    "CONFIG_ISOLATION_FOREST",
    "CONFIG_IF_TEMPORAL",
    "CONFIG_PERCENTILES",
    "CONFIG_FOD",
    "CONFIG_AUTOENCODER",
    "CONFIG_GENERAL",
    "obtener_config_isolation_forest",
    "obtener_config_isolation_forest_temporal",
    "obtener_config_percentiles",
    "obtener_config_fod",
    "obtener_config_autoencoder",
    # Visualizacion
    "visualizar_mapa_scores_if",
    "visualizar_mapa_anomalias_if",
    "visualizar_anomalias_tp_percentiles",
    "visualizar_resumen_multivariante",
    "visualizar_serie_temporal_if",
    "visualizar_distribucion_scores_if",
    "visualizar_serie_temporal_if_temporal",
    "visualizar_serie_temporal_fod",
    "visualizar_serie_temporal_autoencoder",
    "visualizar_resumen_temporal_multivariante",
]

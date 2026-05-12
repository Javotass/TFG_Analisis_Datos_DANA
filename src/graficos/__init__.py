"""Paquete de gráficos para visualización de datos meteorológicos."""

from .series_temporales import (
    generar_series_temporales,
    generar_serie_temperatura,
    generar_serie_viento,
    generar_serie_precipitacion,
    generar_serie_presion
)
from .mapas import (
    generar_mapas_medios,
    generar_mapa_temperatura,
    generar_mapa_viento,
    generar_mapa_precipitacion,
    generar_mapa_presion
)
from .distribuciones import generar_distribuciones
from .correlaciones import generar_matriz_correlacion
from .ciclo_mensual import (
    generar_ciclo_mensual,
    generar_ciclo_mensual_temperatura,
    generar_ciclo_mensual_viento,
    generar_ciclo_mensual_precipitacion,
    generar_ciclo_mensual_presion
)
from .config_graficos import configurar_estilo

__all__ = [
    'generar_series_temporales',
    'generar_serie_temperatura',
    'generar_serie_viento',
    'generar_serie_precipitacion',
    'generar_serie_presion',
    'generar_mapas_medios',
    'generar_mapa_temperatura',
    'generar_mapa_viento',
    'generar_mapa_precipitacion',
    'generar_mapa_presion',
    'generar_distribuciones',
    'generar_matriz_correlacion',
    'generar_ciclo_mensual',
    'generar_ciclo_mensual_temperatura',
    'generar_ciclo_mensual_viento',
    'generar_ciclo_mensual_precipitacion',
    'generar_ciclo_mensual_presion',
    'configurar_estilo'
]

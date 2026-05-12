# filepath: c:\Users\javie\TFG_Analisis_Datos\src\__init__.py
"""
Módulo src para análisis de datos ERA5.

Contiene:
- config_loader: Carga de configuración desde JSON
- estadisticas: Cálculo de estadísticas descriptivas
- visualizaciones: Generación de gráficos
- exportador: Exportación a CSV y JSON
"""

from .config_loader import cargar_configuracion, obtener_rutas, inicializar_resultados
from .estadisticas import (
    calcular_dimensiones,
    calcular_variables,
    calcular_coordenadas,
    calcular_todas_estadisticas,
    calcular_valores_faltantes,
    calcular_correlaciones
)
from .visualizaciones import generar_todos_graficos
from .exportador import exportar_serie_temporal_csv, guardar_resultados_json

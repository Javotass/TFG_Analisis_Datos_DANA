import xarray as xr
from pathlib import Path

from .graficos.config_graficos import configurar_estilo
from .graficos.series_temporales import generar_series_temporales
from .graficos.mapas import generar_mapas_medios, generar_mapas_animados
from .graficos.distribuciones import generar_distribuciones
from .graficos.correlaciones import generar_matriz_correlacion
from .graficos.ciclo_mensual import generar_ciclo_mensual


# Re-exportar las funciones para mantener compatibilidad hacia atrás
__all__ = [
    'configurar_estilo',
    'generar_series_temporales',
    'generar_mapas_medios',
    'generar_distribuciones',
    'generar_matriz_correlacion',
    'generar_ciclo_mensual',
    'generar_todos_graficos'
]


def generar_todos_graficos(ds: xr.Dataset, output_dir: Path, fecha_inicio: str, fecha_fin: str) -> list:
    """Genera todos los gráficos del análisis."""
    configurar_estilo()
    output_dir = Path(output_dir)

    subcarpetas = {
        "series_temporales": output_dir / "series_temporales",
        "mapas": output_dir / "mapas",
        "distribuciones": output_dir / "distribuciones",
        "correlaciones": output_dir / "correlaciones",
        "ciclo_mensual": output_dir / "ciclo_mensual"
    }
    
    graficos = []
    
    print("  Generando series temporales...")
    graficos.extend(generar_series_temporales(ds, subcarpetas["series_temporales"]))
    
    print("  Generando mapas de valores medios...")
    graficos.extend(generar_mapas_medios(ds, subcarpetas["mapas"], fecha_inicio, fecha_fin))
    
    print("  Generando mapas animados...")
    graficos.extend(generar_mapas_animados(ds, subcarpetas["mapas"], fecha_inicio, fecha_fin))
    
    print("  Generando distribuciones...")
    graficos.append(generar_distribuciones(ds, subcarpetas["distribuciones"]))
    
    print("  Generando matriz de correlación...")
    ruta_corr = generar_matriz_correlacion(ds, subcarpetas["correlaciones"])
    if ruta_corr:
        graficos.append(ruta_corr)
    
    print("  Generando ciclo mensual...")
    graficos.extend(generar_ciclo_mensual(ds, subcarpetas["ciclo_mensual"], fecha_inicio, fecha_fin))
    
    return graficos

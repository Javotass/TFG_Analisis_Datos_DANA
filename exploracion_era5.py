import time
import tracemalloc
import xarray as xr
from pathlib import Path

from src import (
    cargar_configuracion,
    obtener_rutas,
    inicializar_resultados,
    calcular_dimensiones,
    calcular_variables,
    calcular_coordenadas,
    calcular_todas_estadisticas,
    calcular_valores_faltantes,
    calcular_correlaciones,
    generar_todos_graficos,
    exportar_serie_temporal_csv,
    guardar_resultados_json
)


def main():
    print("=" * 80)
    print("ANÁLISIS EXPLORATORIO DE DATOS ERA5-LAND")
    print("=" * 80)

    CONFIG_FILE = "config.json"
    print(f"\nCargando configuración desde: {CONFIG_FILE}")

    config = cargar_configuracion(CONFIG_FILE)
    rutas = obtener_rutas(config)
    resultados = inicializar_resultados(config, rutas)

    print(f"Cargando datos desde: {rutas['data_file']}")
    ds = xr.open_dataset(rutas['data_file'])

    print("\n" + "-" * 80)
    print("CALCULANDO ESTADÍSTICAS...")
    print("-" * 80)

    resultados["dimensiones"] = calcular_dimensiones(ds)
    print(f"  Dimensiones: {resultados['dimensiones']}")

    resultados["variables"] = calcular_variables(ds)
    print(f"  Variables: {list(resultados['variables'].keys())}")

    resultados["coordenadas"] = calcular_coordenadas(ds)
    print(f"  Período: {resultados['coordenadas']['tiempo_inicio']} a {resultados['coordenadas']['tiempo_fin']}")

    resultados["estadisticas"] = calcular_todas_estadisticas(ds)
    for var, stats in resultados["estadisticas"].items():
        print(f"  {var}: Media = {stats['media']} {stats['unidad']}")

    resultados["valores_faltantes"] = calcular_valores_faltantes(ds)
    resultados["correlaciones"] = calcular_correlaciones(ds)

    print("\n" + "-" * 80)
    print("GENERANDO VISUALIZACIONES...")
    print("-" * 80)

    output_dir = Path(rutas["graficos_folder"])
    graficos = generar_todos_graficos(
        ds,
        output_dir,
        rutas["fecha_inicio"],
        rutas["fecha_fin"]
    )
    resultados["graficos_generados"] = graficos

    for grafico in graficos:
        print(f"  [OK] {grafico}")

    print("\n" + "-" * 80)
    print("EXPORTANDO DATOS...")
    print("-" * 80)

    csv_file = exportar_serie_temporal_csv(ds, rutas["csv_file"])
    resultados["archivos_exportados"].append(csv_file)
    print(f"  [OK] CSV: {csv_file}")

    json_file = guardar_resultados_json(resultados, rutas["log_file"])
    print(f"  [OK] JSON: {json_file}")

    ds.close()
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO")
    print("=" * 80)
    print(f"Resultados: {rutas['log_file']}")
    print(f"Gráficos: {output_dir.absolute()}")
    print(f"CSV: {rutas['csv_file']}")


if __name__ == "__main__":
    tracemalloc.start()
    _t0 = time.time()
    main()
    _elapsed = time.time() - _t0
    _, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nTiempo de ejecución : {_elapsed:.1f} s")
    print(f"Memoria pico        : {_peak / 1024 / 1024:.1f} MB")
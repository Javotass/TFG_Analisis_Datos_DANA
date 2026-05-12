"""
Script principal para Deteccion de Anomalias.

Algoritmos
----------
Isolation Forest (univariante)
    Variables: t2m, wind_speed, sp.
    Cada punto geografico (lat, lon) es una muestra;
    sus N timesteps son las features.
    Salida: mapa espacial (lat, lon) de scores y mascara binaria.

Isolation Forest (multivariante)
    Variables: t2m + wind_speed + sp concatenadas.
    Detecta lugares con combinaciones anomalas entre variables.
    Salida: mapa espacial (lat, lon).

Percentiles p99
    Variable: tp (precipitacion).
    Para cada punto (lat, lon) se calcula su umbral P99 local.
    Salida: cubo espacio-temporal (valid_time, lat, lon).

Ecuacion en diferencias de primer orden (temporal)
    Modelo: x_t = a*x_(t-1) + b sobre la media espacial por timestep.
    Detecta eventos por residuo estandarizado del modelo.
    Salida: serie temporal (valid_time,) de scores y mascara.

Autoencoder temporal
    Aprende el patron normal de la dinamica temporal por reconstruccion.
    Detecta eventos por error alto de reconstruccion.
    Salida: serie temporal (valid_time,) de scores y mascara.
"""

import json
import csv
import matplotlib
matplotlib.use('Agg')  # backend no-interactivo: evita errores de Tkinter en Windows
import xarray as xr
import numpy as np
from pathlib import Path

from src.anomalias import (
    detectar_anomalias_isolation_forest,
    detectar_anomalias_isolation_forest_multivariante,
    detectar_anomalias_isolation_forest_temporal,
    detectar_anomalias_isolation_forest_temporal_multivariante,
    detectar_anomalias_diferencias_primer_orden_temporal,
    detectar_anomalias_autoencoder_temporal,
    detectar_anomalias_percentiles,
    CONFIG_ANOMALIAS,
    obtener_config_isolation_forest,
    obtener_config_isolation_forest_temporal,
    obtener_config_percentiles,
    obtener_config_fod,
    obtener_config_autoencoder,
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
from src import cargar_configuracion


# ---------------------------------------------------------------------------
# Preparacion de variables
# ---------------------------------------------------------------------------

def preparar_datos(ds: xr.Dataset, variable: str):
    """
    Extrae y transforma una variable del dataset ERA5-Land.

    Conversiones:
      t2m       : Kelvin -> Celsius
      wind_speed: sqrt(u10^2 + v10^2)  [m/s]
      tp        : m -> mm
      sp        : Pa -> hPa

    Returns None si la variable no esta disponible.
    """
    if variable == "t2m":
        if "t2m" not in ds:
            return None
        da = ds["t2m"] - 273.15
        da.name = "t2m"
        return da

    elif variable == "wind_speed":
        if "u10" not in ds or "v10" not in ds:
            return None
        da = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
        da.name = "wind_speed"
        return da

    elif variable == "tp":
        if "tp" not in ds:
            return None
        da = ds["tp"] * 1000.0
        da.name = "tp"
        return da

    elif variable == "sp":
        if "sp" not in ds:
            return None
        da = ds["sp"] / 100.0
        da.name = "sp"
        return da

    return None


# ---------------------------------------------------------------------------
# Deteccion por variable
# ---------------------------------------------------------------------------

def detectar_isolation_forest(
    data: xr.DataArray,
    variable: str,
    cfg: dict,
) -> tuple:
    """Aplica Isolation Forest univariante y devuelve (scores, mascara, stats)."""
    print(f"  > Isolation Forest [{variable.upper()}]...")
    scores, mascara, stats = detectar_anomalias_isolation_forest(
        data,
        n_estimators=cfg["n_estimators"],
        contamination=cfg["contamination"],
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
        usar_criterio_media=cfg.get("usar_criterio_media", True),
        umbral_zscore_media=cfg.get("umbral_zscore_media", 2.0),
        combinar_criterios=cfg.get("combinar_criterios", "or"),
    )
    n = stats["num_anomalias"]
    p = stats["porcentaje_anomalias"]
    print(f"    * Puntos anomalos: {n} / {stats['n_total_puntos']} ({p:.2f}%)")
    if "num_anomalias_if" in stats and "num_anomalias_media" in stats:
        print(
            "    * Desglose -> "
            f"IF: {stats['num_anomalias_if']} ({stats['porcentaje_anomalias_if']:.2f}%), "
            f"z-score media: {stats['num_anomalias_media']} ({stats['porcentaje_anomalias_media']:.2f}%), "
            f"fusion: {stats.get('combinar_criterios', 'or')}"
        )
    return scores, mascara, stats


def detectar_percentiles_tp(
    data_tp: xr.DataArray,
    cfg: dict,
) -> tuple:
    """Aplica Percentiles p99 a tp y devuelve (mascara, umbrales, stats)."""
    print("  > Percentiles P99 [TP]...")
    mascara, umbrales, stats = detectar_anomalias_percentiles(
        data_tp,
        percentil=cfg["percentil"],
        min_threshold=cfg["min_threshold"],
    )
    n = stats["num_anomalias"]
    p = stats["porcentaje_anomalias"]
    d = stats["dias_con_anomalia"]
    print(f"    * Observaciones anomalas : {n} ({p:.2f}%)")
    print(f"    * Dias con algun punto anomalo: {d}")
    return mascara, umbrales, stats


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

def generar_visualizaciones_if(
    data: xr.DataArray,
    scores: xr.DataArray,
    mascara: xr.DataArray,
    variable: str,
    output_dir: Path,
) -> None:
    """Genera y guarda los graficos de Isolation Forest univariante."""
    var_cfg = CONFIG_ANOMALIAS[variable]
    nombre = var_cfg["nombre"]
    unidad = var_cfg["unidad"]
    label = f"{nombre} ({unidad})"

    var_dir = output_dir / variable
    var_dir.mkdir(parents=True, exist_ok=True)

    # Mapa de scores
    visualizar_mapa_scores_if(
        scores, mascara,
        titulo=f"Isolation Forest — {nombre}",
        variable_nombre=label,
        output_path=var_dir / "if_scores.svg",
    )
    print(f"    [OK] if_scores.svg")

    # Mapa de anomalias sobre el campo medio
    visualizar_mapa_anomalias_if(
        data, mascara,
        titulo=f"Anomalias — {nombre}",
        variable_nombre=label,
        output_path=var_dir / "if_anomalias.svg",
    )
    print(f"    [OK] if_anomalias.svg")

    # Serie temporal: normales vs anomalos
    visualizar_serie_temporal_if(
        data, mascara,
        titulo=f"Serie Temporal — {nombre}",
        variable_nombre=label,
        output_path=var_dir / "if_serie_temporal.svg",
    )
    print(f"    [OK] if_serie_temporal.svg")

    # Distribucion de scores (diagnostico de contaminacion)
    visualizar_distribucion_scores_if(
        scores,
        titulo=f"Distribucion de Scores IF — {nombre}",
        variable_nombre=label,
        output_path=var_dir / "if_distribucion_scores.svg",
    )
    print(f"    [OK] if_distribucion_scores.svg")


def detectar_if_temporal(
    data: xr.DataArray,
    variable: str,
    cfg: dict,
) -> tuple:
    """Aplica Isolation Forest temporal y devuelve (scores, mascara, stats).

    Cada timestep es una muestra; el modelo detecta momentos en el tiempo
    con patron espacial atipico (eventos meteorologicos).
    """
    print(f"  > IF Temporal [{variable.upper()}]...")
    scores, mascara, stats = detectar_anomalias_isolation_forest_temporal(
        data,
        n_estimators=cfg["n_estimators"],
        contamination=cfg["contamination"],
        random_state=cfg["random_state"],
        n_jobs=cfg["n_jobs"],
        usar_criterio_media=cfg.get("usar_criterio_media", True),
        umbral_zscore_media=cfg.get("umbral_zscore_media", 2.0),
        combinar_criterios=cfg.get("combinar_criterios", "or"),
    )
    n = stats["num_anomalias"]
    p = stats["porcentaje_anomalias"]
    print(f"    * Timesteps anomalos: {n} / {stats['n_total_timesteps']} ({p:.2f}%)")
    if "num_anomalias_if" in stats and "num_anomalias_media" in stats:
        print(
            "    * Desglose -> "
            f"IF: {stats['num_anomalias_if']} ({stats['porcentaje_anomalias_if']:.2f}%), "
            f"z-score media: {stats['num_anomalias_media']} ({stats['porcentaje_anomalias_media']:.2f}%), "
            f"fusion: {stats.get('combinar_criterios', 'or')}"
        )
    return scores, mascara, stats


def detectar_fod_temporal(
    data: xr.DataArray,
    variable: str,
    cfg: dict,
) -> tuple:
    """Aplica x_t = a*x_(t-1)+b y devuelve (scores, mascara, stats)."""
    umbral_var = cfg.get("umbral_zscore_residuo_por_variable", {}).get(
        variable,
        cfg["umbral_zscore_residuo"],
    )
    print(f"  > Ecuacion diferencias 1er orden [{variable.upper()}]...")
    scores, mascara, stats = detectar_anomalias_diferencias_primer_orden_temporal(
        data,
        umbral_zscore_residuo=umbral_var,
        min_puntos_ajuste=cfg["min_puntos_ajuste"],
    )
    n = stats["num_anomalias"]
    p = stats["porcentaje_anomalias"]
    print(f"    * Timesteps anomalos: {n} / {stats['n_timesteps_evaluados']} ({p:.2f}%)")
    print(
        "    * Parametros -> "
        f"a={stats['a']:.4f}, b={stats['b']:.4f}, "
        f"estable={stats['estable']}, tipo={stats['tipo_estabilidad']}, "
        f"umbral_z={stats['umbral_zscore_residuo']:.2f}"
    )
    if stats.get("equilibrio") is not None:
        print(f"    * Equilibrio estimado: {stats['equilibrio']:.4f}")
    return scores, mascara, stats


def generar_visualizaciones_if_temporal(
    data: xr.DataArray,
    scores: xr.DataArray,
    mascara: xr.DataArray,
    variable: str,
    output_dir: Path,
    stats: dict | None = None,
) -> None:
    """Genera y guarda el grafico de IF temporal para una variable."""
    var_cfg = CONFIG_ANOMALIAS[variable]
    nombre = var_cfg["nombre"]
    unidad = var_cfg["unidad"]
    label = f"{nombre} ({unidad})"

    var_dir = output_dir / variable
    var_dir.mkdir(parents=True, exist_ok=True)

    umbral = stats.get("offset_decision") if stats else None

    visualizar_serie_temporal_if_temporal(
        data, scores, mascara,
        titulo=f"IF Temporal — {nombre}",
        variable_nombre=label,
        output_path=var_dir / "if_temporal_eventos.svg",
        umbral_decision=umbral,
    )
    print(f"    [OK] if_temporal_eventos.svg")


def generar_visualizaciones_fod_temporal(
    data: xr.DataArray,
    scores: xr.DataArray,
    mascara: xr.DataArray,
    variable: str,
    output_dir: Path,
) -> None:
    """Genera y guarda el grafico temporal del metodo de diferencias."""
    var_cfg = CONFIG_ANOMALIAS[variable]
    nombre = var_cfg["nombre"]
    unidad = var_cfg["unidad"]
    label = f"{nombre} ({unidad})"

    var_dir = output_dir / variable
    var_dir.mkdir(parents=True, exist_ok=True)

    visualizar_serie_temporal_fod(
        data, scores, mascara,
        titulo=f"Ecuacion en diferencias 1er orden — {nombre}",
        variable_nombre=label,
        output_path=var_dir / "fod_temporal_eventos.svg",
    )
    print(f"    [OK] fod_temporal_eventos.svg")


def detectar_autoencoder_temporal(
    data: xr.DataArray,
    variable: str,
    cfg: dict,
) -> tuple:
    """Aplica autoencoder temporal y devuelve (scores, mascara, stats)."""
    print(f"  > Autoencoder temporal [{variable.upper()}]...")
    hls = tuple(cfg["hidden_layer_sizes"])
    scores, mascara, stats = detectar_anomalias_autoencoder_temporal(
        data,
        hidden_layer_sizes=hls,
        contamination=cfg["contamination"],
        random_state=cfg["random_state"],
        max_iter=cfg["max_iter"],
    )
    n = stats["num_anomalias"]
    p = stats["porcentaje_anomalias"]
    print(f"    * Timesteps anomalos: {n} / {stats['n_timesteps_evaluados']} ({p:.2f}%)")
    print(
        "    * Parametros -> "
        f"H={stats['hidden_layer_sizes']}, contamination={stats['contamination']}, "
        f"threshold_error={stats['threshold_error']:.6f}"
    )
    return scores, mascara, stats


def generar_visualizaciones_autoencoder_temporal(
    data: xr.DataArray,
    scores: xr.DataArray,
    mascara: xr.DataArray,
    variable: str,
    output_dir: Path,
) -> None:
    """Genera y guarda el grafico temporal del autoencoder."""
    var_cfg = CONFIG_ANOMALIAS[variable]
    nombre = var_cfg["nombre"]
    unidad = var_cfg["unidad"]
    label = f"{nombre} ({unidad})"

    var_dir = output_dir / variable
    var_dir.mkdir(parents=True, exist_ok=True)

    visualizar_serie_temporal_autoencoder(
        data, scores, mascara,
        titulo=f"Autoencoder temporal — {nombre}",
        variable_nombre=label,
        output_path=var_dir / "ae_temporal_eventos.svg",
    )
    print(f"    [OK] ae_temporal_eventos.svg")


def generar_visualizaciones_tp(
    data_tp: xr.DataArray,
    mascara: xr.DataArray,
    umbrales: xr.DataArray,
    output_dir: Path,
) -> None:
    """Genera y guarda los graficos de Percentiles para tp."""
    tp_dir = output_dir / "tp"
    tp_dir.mkdir(parents=True, exist_ok=True)

    visualizar_anomalias_tp_percentiles(
        data_tp, mascara, umbrales,
        titulo="Precipitacion Extrema — Percentil P99",
        output_path=tp_dir / "percentil_p99.svg",
    )
    print("    [OK] percentil_p99.svg")


def generar_visualizaciones_multivariante(
    scores: xr.DataArray,
    mascara: xr.DataArray,
    output_dir: Path,
) -> None:
    """Genera y guarda el grafico del modelo multivariante."""
    multi_dir = output_dir / "multivariante"
    multi_dir.mkdir(parents=True, exist_ok=True)

    visualizar_resumen_multivariante(
        scores, mascara,
        titulo="Isolation Forest Multivariante (t2m + wind_speed + sp)",
        output_path=multi_dir / "if_multivariante.svg",
    )
    print("    [OK] if_multivariante.svg")


# ---------------------------------------------------------------------------
# Exportacion de resultados
# ---------------------------------------------------------------------------

def exportar_estadisticas(resultados: dict, output_dir: Path) -> None:
    """Guarda las estadisticas de todos los modelos en un JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ruta = output_dir / "estadisticas.json"

    # Convertir valores numpy a Python nativos para JSON
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(_clean(resultados), f, indent=2, ensure_ascii=False)

    print(f"   * Estadisticas guardadas en: {ruta}")


def _to_iso_timestamp(ts) -> str:
    """Convierte timestamps numpy/xarray a texto ISO legible."""
    try:
        return np.datetime_as_string(ts, unit="m")
    except Exception:
        return str(ts)


def exportar_comparativa_anomalias_temporales(
    resultados_if_temporal: dict,
    resultados_fod: dict,
    resultados_ae: dict,
    output_dir: Path,
) -> dict:
    """Exporta lista por metodo y comparativa para clasificar anomalias absolutas.

    Una anomalia se clasifica como "absoluta" cuando coincide en los 3 metodos:
    IF temporal, FOD temporal y Autoencoder temporal.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    variables_comunes = sorted(
        set(resultados_if_temporal.keys())
        & set(resultados_fod.keys())
        & set(resultados_ae.keys())
    )

    filas_globales = []
    resumen = {}

    for var in variables_comunes:
        da_if = resultados_if_temporal[var]["mascara"]
        da_fod = resultados_fod[var]["mascara"]
        da_ae = resultados_ae[var]["mascara"]

        tiempos = da_if.valid_time.values
        m_if = da_if.values.astype(bool)
        m_fod = da_fod.values.astype(bool)
        m_ae = da_ae.values.astype(bool)

        n = min(len(tiempos), len(m_if), len(m_fod), len(m_ae))
        tiempos = tiempos[:n]
        m_if = m_if[:n]
        m_fod = m_fod[:n]
        m_ae = m_ae[:n]

        m_union = m_if | m_fod | m_ae
        m_abs = m_if & m_fod & m_ae

        filas_var = []
        for i in range(n):
            if not m_union[i]:
                continue

            n_metodos = int(m_if[i]) + int(m_fod[i]) + int(m_ae[i])
            if n_metodos == 3:
                clasificacion = "absoluta"
            elif n_metodos == 2:
                clasificacion = "consenso_2_de_3"
            else:
                clasificacion = "unico"

            fila = {
                "variable": var,
                "fecha": _to_iso_timestamp(tiempos[i]),
                "if_temporal": int(m_if[i]),
                "fod_temporal": int(m_fod[i]),
                "autoencoder_temporal": int(m_ae[i]),
                "n_metodos": n_metodos,
                "clasificacion": clasificacion,
            }
            filas_var.append(fila)
            filas_globales.append(fila)

        ruta_var = output_dir / f"anomalias_comparativa_{var}.csv"
        with open(ruta_var, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "variable",
                    "fecha",
                    "if_temporal",
                    "fod_temporal",
                    "autoencoder_temporal",
                    "n_metodos",
                    "clasificacion",
                ],
            )
            writer.writeheader()
            writer.writerows(filas_var)

        n_if = int(m_if.sum())
        n_fod = int(m_fod.sum())
        n_ae = int(m_ae.sum())
        n_union = int(m_union.sum())
        n_abs = int(m_abs.sum())
        resumen[var] = {
            "if_temporal": n_if,
            "fod_temporal": n_fod,
            "autoencoder_temporal": n_ae,
            "union_total": n_union,
            "anomalias_absolutas": n_abs,
            "porcentaje_absolutas_sobre_union": round(100.0 * n_abs / n_union, 3) if n_union else 0.0,
        }

    ruta_global = output_dir / "anomalias_comparativa_global.csv"
    with open(ruta_global, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variable",
                "fecha",
                "if_temporal",
                "fod_temporal",
                "autoencoder_temporal",
                "n_metodos",
                "clasificacion",
            ],
        )
        writer.writeheader()
        writer.writerows(filas_globales)

    ruta_resumen = output_dir / "anomalias_comparativa_resumen.json"
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    return {
        "variables": variables_comunes,
        "resumen": resumen,
        "ruta_global": str(ruta_global),
        "ruta_resumen": str(ruta_resumen),
    }


# ---------------------------------------------------------------------------
# Exportacion espacial
# ---------------------------------------------------------------------------

def exportar_anomalias_espaciales_csv(
    resultados_if: dict,
    resultados_multi: dict,
    output_dir: Path,
) -> str:
    """Exporta las coordenadas (lat, lon) de los puntos anomalos del IF espacial a CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filas = []

    for var, res in resultados_if.items():
        mascara = res["mascara"]
        scores = res["scores"]
        lats = mascara.latitude.values
        lons = mascara.longitude.values
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if bool(mascara.values[i, j]):
                    filas.append({
                        "variable": var,
                        "latitude": round(float(lat), 4),
                        "longitude": round(float(lon), 4),
                        "if_score": round(float(scores.values[i, j]), 6),
                        "criterio_if": int(mascara.if_mask.values[i, j]),
                        "criterio_media": int(mascara.media_mask.values[i, j]),
                    })

    if resultados_multi:
        mascara_m = resultados_multi["mascara"]
        scores_m = resultados_multi["scores"]
        lats = mascara_m.latitude.values
        lons = mascara_m.longitude.values
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if bool(mascara_m.values[i, j]):
                    filas.append({
                        "variable": "multivariante",
                        "latitude": round(float(lat), 4),
                        "longitude": round(float(lon), 4),
                        "if_score": round(float(scores_m.values[i, j]), 6),
                        "criterio_if": 1,
                        "criterio_media": 0,
                    })

    ruta = output_dir / "anomalias_espaciales_if.csv"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variable", "latitude", "longitude", "if_score", "criterio_if", "criterio_media"],
        )
        writer.writeheader()
        writer.writerows(filas)

    return str(ruta)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Ejecuta la deteccion de anomalias completa."""

    print("=" * 80)
    print("DETECCION DE ANOMALIAS")
    print("=" * 80)
    print("Algoritmos:")
    print("  - Isolation Forest (univariante) : t2m, wind_speed, sp")
    print("  - Isolation Forest (multivariante): t2m + wind_speed + sp")
    print("  - Percentiles P99               : tp  (precipitacion)")
    print("  - Ecuacion 1er orden (temporal) : t2m, wind_speed, sp, tp")
    print("  - Autoencoder temporal          : t2m, wind_speed, sp, tp")
    print("=" * 80)

    # -- Configuracion --------------------------------------------------------
    config = cargar_configuracion("config.json")
    datos_folder = Path(config.get("output_folder", "datos"))
    archivo_entrada = config.get("archivo_entrada", "").strip()
    if archivo_entrada:
        ruta_nc = Path(archivo_entrada)
        if not ruta_nc.is_absolute():
            ruta_nc = Path(ruta_nc)
        if not ruta_nc.exists():
            print(f"ERROR: El archivo_entrada no existe: {ruta_nc}")
            return
    else:
        archivos_nc = list(datos_folder.glob("era5_land_*.nc"))

        if not archivos_nc:
            print(f"ERROR: No se encontro ningun archivo NetCDF en {datos_folder}/")
            print("Ejecuta primero: python descarga_era5.py")
            return

        ruta_nc = max(archivos_nc, key=lambda p: p.stat().st_mtime)
    print(f"\n1. Archivo de datos: {ruta_nc.name}")

    # -- Directorios de salida ------------------------------------------------
    output_base = Path("resultados_anomalias/fase1")
    output_graficos = output_base / "graficos"
    output_graficos.mkdir(parents=True, exist_ok=True)
    print(f"   Directorio de salida: {output_graficos}")

    # -- Dataset --------------------------------------------------------------
    print("\n2. Cargando dataset ERA5-Land...")
    with xr.open_dataset(ruta_nc) as ds:
        n_pasos = len(ds.valid_time)
        print(f"   * {n_pasos} pasos temporales, variables: {list(ds.data_vars)}")

        # -- Preparar variables -----------------------------------------------
        print("\n3. Preparando variables...")
        datos = {}
        for var in ["t2m", "wind_speed", "tp", "sp"]:
            da = preparar_datos(ds, var)
            if da is not None:
                # Cargar en memoria para desvincular del fichero ya cerrado.
                datos[var] = da.load()
                print(f"   [OK] {var}: shape {da.shape}")
            else:
                print(f"   [--] {var}: no disponible")

    # -- Isolation Forest univariante -----------------------------------------
    print("\n4. Isolation Forest — univariante")
    print("-" * 60)

    cfg_if = obtener_config_isolation_forest(multivariante=False)
    variables_if = [v for v in ["t2m", "wind_speed", "sp"] if v in datos]

    resultados_if = {}
    for var in variables_if:
        scores, mascara, stats = detectar_isolation_forest(datos[var], var, cfg_if)
        resultados_if[var] = {"scores": scores, "mascara": mascara, "stats": stats}
        generar_visualizaciones_if(datos[var], scores, mascara, var, output_graficos)

    # -- Isolation Forest multivariante ---------------------------------------
    print("\n5. Isolation Forest — multivariante")
    print("-" * 60)

    cfg_multi = obtener_config_isolation_forest(multivariante=True)
    vars_multi = {v: datos[v] for v in cfg_multi["variables"] if v in datos}

    if len(vars_multi) >= 2:
        print(f"  > Variables: {list(vars_multi.keys())}")
        scores_m, mascara_m, stats_m = detectar_anomalias_isolation_forest_multivariante(
            vars_multi,
            n_estimators=cfg_multi["n_estimators"],
            contamination=cfg_multi["contamination"],
            random_state=cfg_multi["random_state"],
            n_jobs=cfg_multi["n_jobs"],
        )
        n_m = stats_m["num_anomalias"]
        p_m = stats_m["porcentaje_anomalias"]
        print(f"    * Puntos anomalos: {n_m} / {stats_m['n_total_puntos']} ({p_m:.2f}%)")
        generar_visualizaciones_multivariante(scores_m, mascara_m, output_graficos)
        resultados_multi = {"scores": scores_m, "mascara": mascara_m, "stats": stats_m}
    else:
        print("  ! No hay suficientes variables para el modelo multivariante.")
        resultados_multi = {}

    # -- Percentiles p99 — tp -------------------------------------------------
    print("\n6. Percentiles P99 — precipitacion (tp)")
    print("-" * 60)

    resultados_tp = {}
    if "tp" in datos:
        cfg_pct = obtener_config_percentiles("tp")
        mascara_tp, umbrales_tp, stats_tp = detectar_percentiles_tp(datos["tp"], cfg_pct)
        generar_visualizaciones_tp(datos["tp"], mascara_tp, umbrales_tp, output_graficos)
        resultados_tp = {"mascara": mascara_tp, "umbrales": umbrales_tp, "stats": stats_tp}
    else:
        print("  ! Variable tp no disponible.")

    # -- Isolation Forest temporal — detecta eventos --------------------------
    print("\n7. Isolation Forest — temporal (deteccion de eventos)")
    print("-" * 60)
    print("   Enfoque: cada timestep es una muestra, datos espaciales son features.")
    print("   Detecta momentos temporales con patron espacial atipico.")

    cfg_tem = obtener_config_isolation_forest_temporal()
    vars_temporal = [v for v in cfg_tem["variables"] if v in datos]

    resultados_temporal = {}
    for var in vars_temporal:
        scores_t, mascara_t, stats_t = detectar_if_temporal(datos[var], var, cfg_tem)
        resultados_temporal[var] = {"scores": scores_t, "mascara": mascara_t, "stats": stats_t}
        generar_visualizaciones_if_temporal(datos[var], scores_t, mascara_t, var, output_graficos, stats=stats_t)

    # -- Isolation Forest temporal multivariante --------------------------------
    print("\n8. Isolation Forest — temporal multivariante")
    print("-" * 60)
    print("   Combina los campos espaciales de todas las variables por timestep.")
    print("   Detecta eventos donde la combinacion de variables es anomala.")

    cfg_tem_multi = cfg_tem["multivariante"]
    vars_tem_multi = {v: datos[v] for v in cfg_tem_multi["variables"] if v in datos}
    resultados_temporal_multi = {}
    if len(vars_tem_multi) >= 2:
        print(f"  > Variables: {list(vars_tem_multi.keys())}")
        scores_tm, mascara_tm, stats_tm = detectar_anomalias_isolation_forest_temporal_multivariante(
            vars_tem_multi,
            n_estimators=cfg_tem_multi["n_estimators"],
            contamination=cfg_tem_multi["contamination"],
            random_state=cfg_tem_multi["random_state"],
            n_jobs=cfg_tem_multi["n_jobs"],
            usar_criterio_media=cfg_tem.get("usar_criterio_media", True),
            umbral_zscore_media=cfg_tem.get("umbral_zscore_media", 2.0),
            combinar_criterios=cfg_tem.get("combinar_criterios", "or"),
        )
        n_tm = stats_tm["num_anomalias"]
        p_tm = stats_tm["porcentaje_anomalias"]
        print(f"    * Timesteps anomalos: {n_tm} / {stats_tm['n_total_timesteps']} ({p_tm:.2f}%)")
        if "num_anomalias_if" in stats_tm and "num_anomalias_media" in stats_tm:
            print(
                "    * Desglose -> "
                f"IF: {stats_tm['num_anomalias_if']} ({stats_tm['porcentaje_anomalias_if']:.2f}%), "
                f"z-score media: {stats_tm['num_anomalias_media']} ({stats_tm['porcentaje_anomalias_media']:.2f}%), "
                f"fusion: {stats_tm.get('combinar_criterios', 'or')}"
            )

        multi_tem_dir = output_graficos / "multivariante"
        multi_tem_dir.mkdir(parents=True, exist_ok=True)
        visualizar_resumen_temporal_multivariante(
            vars_tem_multi, scores_tm, mascara_tm,
            titulo="IF Temporal Multivariante (t2m + wind_speed + sp)",
            output_path=multi_tem_dir / "if_temporal_multivariante.svg",
        )
        print("    [OK] if_temporal_multivariante.svg")
        resultados_temporal_multi = {"scores": scores_tm, "mascara": mascara_tm, "stats": stats_tm}
    else:
        print("  ! No hay suficientes variables disponibles.")

    # -- Ecuacion en diferencias de primer orden ------------------------------
    print("\n9. Ecuacion en diferencias de primer orden — temporal")
    print("-" * 60)
    print("   Modelo: x_t = a*x_(t-1) + b sobre la media espacial por timestep.")

    cfg_fod = obtener_config_fod()
    vars_fod = [v for v in cfg_fod["variables"] if v in datos]
    resultados_fod = {}
    for var in vars_fod:
        scores_fod, mascara_fod, stats_fod = detectar_fod_temporal(datos[var], var, cfg_fod)
        resultados_fod[var] = {
            "scores": scores_fod,
            "mascara": mascara_fod,
            "stats": stats_fod,
        }
        generar_visualizaciones_fod_temporal(datos[var], scores_fod, mascara_fod, var, output_graficos)

    # -- Autoencoder temporal -------------------------------------------------
    print("\n10. Autoencoder temporal")
    print("-" * 60)
    print("   Deteccion por error de reconstruccion sobre features dinamicas.")

    cfg_ae = obtener_config_autoencoder()
    vars_ae = [v for v in cfg_ae["variables"] if v in datos]
    resultados_ae = {}
    for var in vars_ae:
        scores_ae, mascara_ae, stats_ae = detectar_autoencoder_temporal(datos[var], var, cfg_ae)
        resultados_ae[var] = {
            "scores": scores_ae,
            "mascara": mascara_ae,
            "stats": stats_ae,
        }
        generar_visualizaciones_autoencoder_temporal(datos[var], scores_ae, mascara_ae, var, output_graficos)

    # -- Exportar estadisticas ------------------------------------------------
    print("\n11. Exportando estadisticas...")

    stats_export = {}
    for var, res in resultados_if.items():
        stats_export[f"isolation_forest_{var}"] = res["stats"]
    if resultados_multi:
        stats_export["isolation_forest_multivariante"] = resultados_multi["stats"]
    if resultados_tp:
        stats_export["percentiles_tp"] = resultados_tp["stats"]
    for var, res in resultados_temporal.items():
        stats_export[f"isolation_forest_temporal_{var}"] = res["stats"]
    if resultados_temporal_multi:
        stats_export["isolation_forest_temporal_multivariante"] = resultados_temporal_multi["stats"]
    for var, res in resultados_fod.items():
        stats_export[f"diferencias_primer_orden_temporal_{var}"] = res["stats"]
    for var, res in resultados_ae.items():
        stats_export[f"autoencoder_temporal_{var}"] = res["stats"]

    exportar_estadisticas(stats_export, output_graficos)

    # -- Exportar coordenadas espaciales anomalas -----------------------------
    output_comparativa = output_base / "comparativa_temporal"
    ruta_espacial = exportar_anomalias_espaciales_csv(
        resultados_if,
        resultados_multi,
        output_comparativa,
    )
    print(f"   * CSV espacial: {ruta_espacial}")

    # -- Comparativa temporal entre metodos ----------------------------------
    print("\n12. Comparando metodos temporales (IF, FOD, AE)...")
    output_comparativa = output_base / "comparativa_temporal"
    comparativa = exportar_comparativa_anomalias_temporales(
        resultados_temporal,
        resultados_fod,
        resultados_ae,
        output_comparativa,
    )
    print(f"   * CSV global: {comparativa['ruta_global']}")
    print(f"   * Resumen JSON: {comparativa['ruta_resumen']}")

    # -- Resumen final --------------------------------------------------------
    print(f"\n{'='*80}")
    print("RESUMEN")
    print(f"{'='*80}")

    print("\nIsolation Forest (univariante):")
    for var, res in resultados_if.items():
        st = res["stats"]
        print(f"  {CONFIG_ANOMALIAS[var]['nombre']:25s}: "
              f"{st['num_anomalias']:4d} puntos anomalos "
              f"({st['porcentaje_anomalias']:.2f}%)")

    if resultados_multi:
        st = resultados_multi["stats"]
        print(f"\nIsolation Forest (multivariante):")
        print(f"  {'Combinado':25s}: "
              f"{st['num_anomalias']:4d} puntos anomalos "
              f"({st['porcentaje_anomalias']:.2f}%)")

    if resultados_tp:
        st = resultados_tp["stats"]
        print(f"\nPercentiles P99 (tp):")
        print(f"  {'Precipitacion':25s}: "
              f"{st['num_anomalias']:6d} obs. anomalas "
              f"({st['porcentaje_anomalias']:.2f}%), "
              f"{st['dias_con_anomalia']} dias afectados")

    if resultados_temporal:
        print(f"\nIsolation Forest (temporal — univariante):")
        for var, res in resultados_temporal.items():
            st = res["stats"]
            nombre = CONFIG_ANOMALIAS[var]['nombre']
            print(f"  {nombre:25s}: "
                  f"{st['num_anomalias']:4d} timesteps anomalos "
                  f"({st['porcentaje_anomalias']:.2f}%)")

    if resultados_temporal_multi:
        st = resultados_temporal_multi["stats"]
        print(f"\nIsolation Forest (temporal — multivariante):")
        print(f"  {'Combinado':25s}: "
              f"{st['num_anomalias']:4d} timesteps anomalos "
              f"({st['porcentaje_anomalias']:.2f}%)")

    if resultados_fod:
        print(f"\nEcuacion en diferencias de 1er orden (temporal):")
        for var, res in resultados_fod.items():
            st = res["stats"]
            nombre = CONFIG_ANOMALIAS[var]['nombre']
            eq = st['equilibrio']
            eq_txt = "NA" if eq is None else f"{eq:.3f}"
            print(f"  {nombre:25s}: "
                  f"{st['num_anomalias']:4d} timesteps anomalos "
                  f"({st['porcentaje_anomalias']:.2f}%), "
                  f"a={st['a']:.3f}, eq={eq_txt}, {st['tipo_estabilidad']}")

    if resultados_ae:
        print(f"\nAutoencoder temporal:")
        for var, res in resultados_ae.items():
            st = res["stats"]
            nombre = CONFIG_ANOMALIAS[var]['nombre']
            print(f"  {nombre:25s}: "
                  f"{st['num_anomalias']:4d} timesteps anomalos "
                  f"({st['porcentaje_anomalias']:.2f}%), "
                  f"thr={st['threshold_error']:.5f}")

    if comparativa.get("resumen"):
        print(f"\nComparativa temporal (anomalia absoluta = coincidencia en 3 metodos):")
        for var in comparativa["variables"]:
            info = comparativa["resumen"][var]
            nombre = CONFIG_ANOMALIAS.get(var, {}).get("nombre", var)
            print(
                f"  {nombre:25s}: "
                f"absolutas={info['anomalias_absolutas']:3d}, "
                f"union={info['union_total']:3d}, "
                f"{info['porcentaje_absolutas_sobre_union']:.2f}%"
            )

    print(f"\n[OK] Graficos guardados en: {output_graficos.absolute()}")
    print(f"[OK] Comparativa temporal en: {output_comparativa.absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()

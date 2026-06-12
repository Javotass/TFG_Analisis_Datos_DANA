"""
Análisis de sensibilidad de hiperparámetros para detección de anomalías.

Ejecuta cada algoritmo con 3-4 configuraciones distintas sobre la variable t2m
y reporta: nº de timesteps anómalos, si el 29-30 oct aparece, y la tabla
resumen lista para incluir en la sección 4.4 del TFG.

Uso:
    python analisis_sensibilidad.py
"""

import csv
import json
import warnings
import matplotlib
matplotlib.use('Agg')

import numpy as np
import xarray as xr
from pathlib import Path

from src.anomalias import (
    detectar_anomalias_isolation_forest_temporal,
    detectar_anomalias_diferencias_primer_orden_temporal,
    detectar_anomalias_autoencoder_temporal,
)
from src import cargar_configuracion
from deteccion_anomalias_fase1 import preparar_datos

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fechas de referencia del evento DANA (oct 2024)
# ---------------------------------------------------------------------------
FECHAS_DANA = {"2024-10-29", "2024-10-30", "2024-10-31"}


def _fecha_iso(ts) -> str:
    try:
        return str(np.datetime_as_string(ts, unit="D"))
    except Exception:
        return str(ts)[:10]


def detecta_dana(mascara) -> bool:
    """True si algún timestep de la máscara cae en las fechas DANA."""
    tiempos = mascara.valid_time.values
    for i, t in enumerate(tiempos):
        if bool(mascara.values[i]) and _fecha_iso(t) in FECHAS_DANA:
            return True
    return False


def fechas_anomalas(mascara) -> list[str]:
    tiempos = mascara.valid_time.values
    return [_fecha_iso(t) for i, t in enumerate(tiempos) if bool(mascara.values[i])]


# ---------------------------------------------------------------------------
# Cargar datos una sola vez
# ---------------------------------------------------------------------------

def cargar_datos():
    config = cargar_configuracion("config.json")
    datos_folder = Path(config.get("output_folder", "datos"))
    archivo_entrada = config.get("archivo_entrada", "").strip()

    if archivo_entrada:
        ruta_nc = Path(archivo_entrada)
    else:
        archivos = list(datos_folder.glob("era5_land_*.nc"))
        if not archivos:
            raise FileNotFoundError(f"No se encontró ningún .nc en {datos_folder}/")
        ruta_nc = max(archivos, key=lambda p: p.stat().st_mtime)

    print(f"Cargando: {ruta_nc.name}")
    with xr.open_dataset(ruta_nc) as ds:
        n_t = len(ds.valid_time)
        print(f"  {n_t} timesteps, {list(ds.data_vars)}")
        datos = {}
        for var in ["t2m", "wind_speed", "tp", "sp"]:
            da = preparar_datos(ds, var)
            if da is not None:
                datos[var] = da.load()
    return datos, n_t


# ---------------------------------------------------------------------------
# Experimentos por algoritmo
# ---------------------------------------------------------------------------

VARIABLE = "wind_speed"  # variable representativa para los experimentos

# Isolation Forest temporal — variantes
IF_TEMPORAL_CONFIGS = [
    {"label": "contam=0.01, n=200", "contamination": 0.01, "n_estimators": 200},
    {"label": "contam=0.03, n=200", "contamination": 0.03, "n_estimators": 200},
    {"label": "contam=0.05, n=200", "contamination": 0.05, "n_estimators": 200},  # base
    {"label": "contam=0.10, n=200", "contamination": 0.10, "n_estimators": 200},
    {"label": "contam=0.05, n=100", "contamination": 0.05, "n_estimators": 100},
    {"label": "contam=0.05, n=300", "contamination": 0.05, "n_estimators": 300},
]

# AR(1) / FOD — variantes de umbral z-score
FOD_CONFIGS = [
    {"label": "z=1.5", "umbral_zscore_residuo": 1.5},
    {"label": "z=2.0", "umbral_zscore_residuo": 2.0},
    {"label": "z=2.5", "umbral_zscore_residuo": 2.5},  # base
    {"label": "z=3.0", "umbral_zscore_residuo": 3.0},
    {"label": "z=3.5", "umbral_zscore_residuo": 3.5},
]

# Autoencoder — variantes de arquitectura y contaminación
AE_CONFIGS = [
    {"label": "H=[8,4,8], contam=0.05",   "hidden_layer_sizes": [8, 4, 8],   "contamination": 0.05},
    {"label": "H=[16,6,16], contam=0.05", "hidden_layer_sizes": [16, 6, 16], "contamination": 0.05},  # base
    {"label": "H=[32,10,32], contam=0.05","hidden_layer_sizes": [32, 10, 32],"contamination": 0.05},
    {"label": "H=[16,6,16], contam=0.03", "hidden_layer_sizes": [16, 6, 16], "contamination": 0.03},
    {"label": "H=[16,6,16], contam=0.08", "hidden_layer_sizes": [16, 6, 16], "contamination": 0.08},
]


def run_if_temporal(data, n_t):
    filas = []
    print(f"\n{'='*60}")
    print("IF TEMPORAL — sensibilidad")
    print(f"{'='*60}")
    for cfg in IF_TEMPORAL_CONFIGS:
        es_base = "base" if "contam=0.05, n=200" == cfg["label"] else ""
        scores, mascara, stats = detectar_anomalias_isolation_forest_temporal(
            data,
            n_estimators=cfg["n_estimators"],
            contamination=cfg["contamination"],
            random_state=42,
            n_jobs=-1,
            usar_criterio_media=True,
            umbral_zscore_media=2.0,
            combinar_criterios="or",
        )
        n_anom = stats["num_anomalias"]
        pct = stats["porcentaje_anomalias"]
        dana = detecta_dana(mascara)
        fila = {
            "algoritmo": "IF temporal",
            "configuracion": cfg["label"],
            "base": es_base,
            "n_anomalias": n_anom,
            "pct_anomalias": round(pct, 2),
            "detecta_dana": "SÍ" if dana else "NO",
            "fechas_anomalas": "; ".join(fechas_anomalas(mascara)[:10]),
        }
        filas.append(fila)
        print(f"  {cfg['label']:35s}  anomalias={n_anom:4d} ({pct:.2f}%)  DANA={dana}  {es_base}")
    return filas


def run_fod(data, n_t):
    filas = []
    print(f"\n{'='*60}")
    print("AR(1) / FOD — sensibilidad")
    print(f"{'='*60}")
    for cfg in FOD_CONFIGS:
        es_base = "base" if cfg["umbral_zscore_residuo"] == 2.5 else ""
        scores, mascara, stats = detectar_anomalias_diferencias_primer_orden_temporal(
            data,
            umbral_zscore_residuo=cfg["umbral_zscore_residuo"],
            min_puntos_ajuste=10,
        )
        n_anom = stats["num_anomalias"]
        pct = stats["porcentaje_anomalias"]
        dana = detecta_dana(mascara)
        fila = {
            "algoritmo": "AR(1) FOD",
            "configuracion": cfg["label"],
            "base": es_base,
            "n_anomalias": n_anom,
            "pct_anomalias": round(pct, 2),
            "detecta_dana": "SÍ" if dana else "NO",
            "fechas_anomalas": "; ".join(fechas_anomalas(mascara)[:10]),
        }
        filas.append(fila)
        print(f"  {cfg['label']:35s}  anomalias={n_anom:4d} ({pct:.2f}%)  DANA={dana}  {es_base}")
    return filas


def run_autoencoder(data, n_t):
    filas = []
    print(f"\n{'='*60}")
    print("AUTOENCODER — sensibilidad")
    print(f"{'='*60}")
    for cfg in AE_CONFIGS:
        es_base = "base" if cfg["label"] == "H=[16,6,16], contam=0.05" else ""
        scores, mascara, stats = detectar_anomalias_autoencoder_temporal(
            data,
            hidden_layer_sizes=tuple(cfg["hidden_layer_sizes"]),
            contamination=cfg["contamination"],
            random_state=42,
            max_iter=2000,
        )
        n_anom = stats["num_anomalias"]
        pct = stats["porcentaje_anomalias"]
        dana = detecta_dana(mascara)
        fila = {
            "algoritmo": "Autoencoder",
            "configuracion": cfg["label"],
            "base": es_base,
            "n_anomalias": n_anom,
            "pct_anomalias": round(pct, 2),
            "detecta_dana": "SÍ" if dana else "NO",
            "fechas_anomalas": "; ".join(fechas_anomalas(mascara)[:10]),
        }
        filas.append(fila)
        print(f"  {cfg['label']:35s}  anomalias={n_anom:4d} ({pct:.2f}%)  DANA={dana}  {es_base}")
    return filas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"ANÁLISIS DE SENSIBILIDAD — variable: {VARIABLE}")
    print("=" * 60)

    datos, n_t = cargar_datos()

    if VARIABLE not in datos:
        raise ValueError(f"Variable '{VARIABLE}' no disponible en el dataset")

    data = datos[VARIABLE]

    todas_filas = []
    todas_filas.extend(run_if_temporal(data, n_t))
    todas_filas.extend(run_fod(data, n_t))
    todas_filas.extend(run_autoencoder(data, n_t))

    # Guardar CSV
    out = Path("resultados_anomalias/sensibilidad")
    out.mkdir(parents=True, exist_ok=True)
    ruta_csv = out / f"sensibilidad_{VARIABLE}.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        campos = ["algoritmo", "configuracion", "base", "n_anomalias",
                  "pct_anomalias", "detecta_dana", "fechas_anomalas"]
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(todas_filas)

    # Guardar JSON
    ruta_json = out / f"sensibilidad_{VARIABLE}.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(todas_filas, f, indent=2, ensure_ascii=False)

    # Tabla resumen impresa
    print(f"\n{'='*60}")
    print("TABLA RESUMEN")
    print(f"{'='*60}")
    print(f"{'Algoritmo':<14} {'Configuración':<35} {'Anomalías':>10} {'%':>6} {'DANA':>6} {'Base':>5}")
    print("-" * 80)
    for fila in todas_filas:
        print(
            f"{fila['algoritmo']:<14} "
            f"{fila['configuracion']:<35} "
            f"{fila['n_anomalias']:>10} "
            f"{fila['pct_anomalias']:>6.2f} "
            f"{fila['detecta_dana']:>6} "
            f"{fila['base']:>5}"
        )

    print(f"\n[OK] CSV  : {ruta_csv}")
    print(f"[OK] JSON : {ruta_json}")
    print(f"\nFechas DANA buscadas: {sorted(FECHAS_DANA)}")


if __name__ == "__main__":
    main()

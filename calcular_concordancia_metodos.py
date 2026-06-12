"""
Calcula indices de concordancia entre metodos de deteccion de anomalias
temporales (Isolation Forest, FOD, Autoencoder) por variable.

Metricas calculadas por par de metodos y variable:
  - Jaccard  : |A ∩ B| / |A ∪ B|
  - kappa    : Cohen's kappa sobre los 182 timesteps completos

Salida: tabla en consola + JSON en comparativa_temporal/concordancia_metodos.json
"""

import csv
import json
from pathlib import Path

N_TIMESTEPS = 182
VARIABLES = ["t2m", "wind_speed", "sp", "tp"]
METODOS = ["if_temporal", "fod_temporal", "autoencoder_temporal"]
PARES = [
    ("if_temporal", "fod_temporal"),
    ("if_temporal", "autoencoder_temporal"),
    ("fod_temporal", "autoencoder_temporal"),
]
ETIQUETAS = {
    "if_temporal": "IF",
    "fod_temporal": "FOD",
    "autoencoder_temporal": "AE",
}

base = Path("resultados_anomalias/fase1/comparativa_temporal")


def calcular_metricas(sets: dict, n_union: int) -> dict:
    """Jaccard y kappa de Cohen para cada par de metodos."""
    resultados = {}
    n_normal = N_TIMESTEPS - n_union

    for m1, m2 in PARES:
        tp = len(sets[m1] & sets[m2])
        fp = len(sets[m2] - sets[m1])
        fn = len(sets[m1] - sets[m2])
        tn_csv = n_union - tp - fp - fn
        tn_total = tn_csv + n_normal

        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0

        po = (tp + tn_total) / N_TIMESTEPS
        pa = (tp + fn) / N_TIMESTEPS
        pb = (tp + fp) / N_TIMESTEPS
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0

        clave = f"{ETIQUETAS[m1]}_vs_{ETIQUETAS[m2]}"
        resultados[clave] = {
            "jaccard": round(jaccard, 4),
            "kappa": round(kappa, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn_total,
        }

    return resultados


resultados_globales = {}

for var in VARIABLES:
    csv_path = base / f"anomalias_comparativa_{var}.csv"
    if not csv_path.exists():
        print(f"[WARN] No encontrado: {csv_path}")
        continue

    sets = {m: set() for m in METODOS}

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for m in METODOS:
                if int(row[m]) == 1:
                    sets[m].add(row["fecha"])

    n_union = len(sets[METODOS[0]] | sets[METODOS[1]] | sets[METODOS[2]])
    resultados_globales[var] = calcular_metricas(sets, n_union)


# --- Tabla en consola ---
print()
print(f"{'Variable':<12} {'Par':<14} {'Jaccard':>8} {'Kappa':>8}  Interpretacion")
print("-" * 65)

for var, pares in resultados_globales.items():
    for par, m in pares.items():
        j, k = m["jaccard"], m["kappa"]
        if k < 0:
            interp = "sin acuerdo"
        elif k < 0.20:
            interp = "leve"
        elif k < 0.40:
            interp = "discreto"
        elif k < 0.60:
            interp = "moderado"
        elif k < 0.80:
            interp = "considerable"
        else:
            interp = "casi perfecto"
        print(f"{var:<12} {par:<14} {j:>8.4f} {k:>8.4f}  {interp}")

# --- Guardar JSON ---
output_path = base / "concordancia_metodos.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(resultados_globales, f, indent=2, ensure_ascii=False)

print(f"\nResultados guardados en: {output_path}")

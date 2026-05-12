"""Series temporales y diagnostico de scores para IF espacial."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def visualizar_serie_temporal_if(
    data: xr.DataArray,
    mascara: xr.DataArray,
    titulo: str,
    variable_nombre: str = "",
    output_path: Optional[Path] = None,
) -> str:
    """Serie temporal comparando puntos normales vs anomalos."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    tiempos = data.valid_time.values
    puntos_normales = data.where(~mascara)
    puntos_anomalos = data.where(mascara)

    serie_normal = puntos_normales.mean(dim=["latitude", "longitude"])
    serie_anomala = puntos_anomalos.mean(dim=["latitude", "longitude"])

    ax = axes[0]
    ax.plot(
        tiempos,
        serie_normal.values,
        "b-",
        linewidth=1.5,
        label=f"Puntos normales (n={int((~mascara).sum())})",
        alpha=0.8,
    )
    ax.plot(
        tiempos,
        serie_anomala.values,
        "r-",
        linewidth=1.5,
        label=f"Puntos anomalos (n={int(mascara.sum())})",
        alpha=0.8,
    )
    ax.set_ylabel(variable_nombre if variable_nombre else "Valor", fontsize=11)
    ax.set_title("Series Temporales: Puntos Normales vs Anomalos", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax = axes[1]
    diferencia = serie_anomala - serie_normal
    ax.plot(tiempos, diferencia.values, "purple", linewidth=1.5, alpha=0.8)
    ax.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax.fill_between(
        tiempos,
        0,
        diferencia.values,
        where=(diferencia.values > 0),
        color="red",
        alpha=0.3,
        label="Anomalos > Normales",
    )
    ax.fill_between(
        tiempos,
        0,
        diferencia.values,
        where=(diferencia.values <= 0),
        color="blue",
        alpha=0.3,
        label="Anomalos < Normales",
    )
    ax.set_ylabel(f"Diferencia ({variable_nombre})" if variable_nombre else "Diferencia", fontsize=11)
    ax.set_title("Diferencia: Anomalos - Normales", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax = axes[2]
    vals_normal = puntos_normales.values.flatten()
    vals_anomalo = puntos_anomalos.values.flatten()
    vals_normal = vals_normal[~np.isnan(vals_normal)]
    vals_anomalo = vals_anomalo[~np.isnan(vals_anomalo)]

    bins = np.linspace(
        min(np.percentile(vals_normal, 1), np.percentile(vals_anomalo, 1)),
        max(np.percentile(vals_normal, 99), np.percentile(vals_anomalo, 99)),
        50,
    )

    ax.hist(vals_normal, bins=bins, alpha=0.6, color="blue", label="Puntos normales", density=True)
    ax.hist(vals_anomalo, bins=bins, alpha=0.6, color="red", label="Puntos anomalos", density=True)

    ax.axvline(
        np.median(vals_normal),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mediana normal: {np.median(vals_normal):.2f}",
    )
    ax.axvline(
        np.median(vals_anomalo),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mediana anomala: {np.median(vals_anomalo):.2f}",
    )

    ax.set_xlabel(variable_nombre if variable_nombre else "Valor", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.set_title("Distribucion de Valores: Normales vs Anomalos", fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(titulo, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches="tight", dpi=150)
        plt.close()
        return str(output_path)
    plt.show()
    return ""


def visualizar_distribucion_scores_if(
    scores: xr.DataArray,
    titulo: str,
    variable_nombre: str = "",
    output_path: Optional[Path] = None,
) -> str:
    """Histograma y CDF de scores para diagnostico de contaminacion."""
    vals = scores.values.flatten()
    vals = vals[~np.isnan(vals)]
    n_total = len(vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.hist(vals, bins=80, color="steelblue", alpha=0.75, edgecolor="white")
    ax1.set_xlabel("Score Isolation Forest", fontsize=11)
    ax1.set_ylabel("Numero de puntos", fontsize=11)
    ax1.set_title("Distribucion de Scores\n(mas negativo = mas anomalo)", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    umbrales = {
        "1%": np.percentile(vals, 1),
        "2%": np.percentile(vals, 2),
        "5%": np.percentile(vals, 5),
        "10%": np.percentile(vals, 10),
    }
    colores = {"1%": "green", "2%": "orange", "5%": "red", "10%": "purple"}
    for label, thr in umbrales.items():
        n_anom = int((vals < thr).sum())
        ax1.axvline(thr, color=colores[label], linestyle="--", linewidth=1.5, label=f"contam={label} → {n_anom} puntos")

    ax1.axvline(-0.5, color="black", linestyle=":", linewidth=2, label=f"auto (-0.5) → {int((vals < -0.5).sum())} puntos")
    ax1.legend(fontsize=9, loc="upper left")

    vals_sorted = np.sort(vals)
    cdf = np.arange(1, n_total + 1) / n_total

    ax2.plot(vals_sorted, cdf * 100, "b-", linewidth=1.5)
    ax2.set_xlabel("Score Isolation Forest", fontsize=11)
    ax2.set_ylabel("% de puntos con score <= x", fontsize=11)
    ax2.set_title('CDF de Scores\n(busca el "codo" para elegir el umbral)', fontweight="bold")
    ax2.grid(True, alpha=0.3)

    for label, thr in umbrales.items():
        ax2.axvline(thr, color=colores[label], linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.axvline(-0.5, color="black", linestyle=":", linewidth=2, alpha=0.7)

    for pct in [1, 2, 5, 10]:
        ax2.axhline(pct, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.suptitle(f"{titulo} — {variable_nombre}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches="tight", dpi=150)
        plt.close()
        return str(output_path)
    plt.show()
    return ""

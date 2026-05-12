"""Visualizaciones para Isolation Forest temporal (uni y multivariante)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def visualizar_serie_temporal_if_temporal(
    data: xr.DataArray,
    scores: xr.DataArray,
    mascara: xr.DataArray,
    titulo: str = "IF Temporal - Deteccion de Eventos",
    variable_nombre: str = "Variable",
    output_path: Optional[Path] = None,
    umbral_decision: Optional[float] = None,
) -> str:
    """Visualiza resultados de IF temporal con desglose IF/media."""
    import pandas as pd

    media_temporal = data.mean(dim=["latitude", "longitude"])
    tiempos = pd.to_datetime(media_temporal.valid_time.values)
    valores = media_temporal.values
    mask_vals = mascara.values.astype(bool)

    has_componentes = ("if_mask" in mascara.coords) and ("media_mask" in mascara.coords)
    if has_componentes:
        mask_if = mascara.coords["if_mask"].values.astype(bool)
        mask_media = mascara.coords["media_mask"].values.astype(bool)
        mask_if_solo = np.logical_and(mask_if, np.logical_not(mask_media))
        mask_media_solo = np.logical_and(mask_media, np.logical_not(mask_if))
        mask_ambos = np.logical_and(mask_if, mask_media)
    else:
        mask_if_solo = np.zeros_like(mask_vals, dtype=bool)
        mask_media_solo = np.zeros_like(mask_vals, dtype=bool)
        mask_ambos = mask_vals

    score_vals = scores.values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    ax1.plot(tiempos, valores, color="steelblue", linewidth=1.0, alpha=0.8, label="Media espacial")

    for t in tiempos[mask_vals]:
        ax1.axvline(t, color="red", alpha=0.12, linewidth=1.2)

    if has_componentes:
        if mask_if_solo.any():
            ax1.scatter(
                tiempos[mask_if_solo],
                valores[mask_if_solo],
                color="tab:blue",
                s=35,
                zorder=5,
                alpha=0.85,
                marker="x",
                label=f"IF solo (n={mask_if_solo.sum()})",
            )
        if mask_media_solo.any():
            ax1.scatter(
                tiempos[mask_media_solo],
                valores[mask_media_solo],
                color="tab:orange",
                s=35,
                zorder=5,
                alpha=0.85,
                marker="+",
                label=f"z-score media solo (n={mask_media_solo.sum()})",
            )
        if mask_ambos.any():
            ax1.scatter(
                tiempos[mask_ambos],
                valores[mask_ambos],
                color="tab:red",
                s=45,
                zorder=5,
                alpha=0.85,
                marker="o",
                label=f"IF + media (n={mask_ambos.sum()})",
            )
    elif mask_vals.any():
        ax1.scatter(
            tiempos[mask_vals],
            valores[mask_vals],
            color="red",
            s=45,
            zorder=5,
            alpha=0.85,
            label=f"Timestep anomalo (n={mask_vals.sum()})",
        )

    ax1.set_ylabel(variable_nombre, fontsize=11)
    ax1.set_title("Media espacial - eventos anomalos en rojo", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    colores_barras = ["tab:red" if m else "steelblue" for m in mask_vals]

    ax2.bar(tiempos, score_vals, color=colores_barras, width=pd.Timedelta(hours=10), alpha=0.75)

    if umbral_decision is not None:
        umbral = umbral_decision
    elif mask_vals.any():
        umbral = float(score_vals[mask_vals].max())
    else:
        umbral = None
    if umbral is not None:
        ax2.axhline(umbral, color="red", linestyle="--", linewidth=1.5, label=f"Umbral de decision ({umbral:.3f})")
        ax2.legend(fontsize=9)

    ax2.set_xlabel("Fecha", fontsize=11)
    ax2.set_ylabel("Score IF\n(mas negativo = mas anomalo)", fontsize=11)
    ax2.set_title("Score de Anomalia por Timestep", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches="tight", dpi=150)
        plt.close()
        return str(output_path)
    plt.show()
    return ""


def visualizar_resumen_temporal_multivariante(
    variables: Mapping[str, xr.DataArray],
    scores: xr.DataArray,
    mascara: xr.DataArray,
    titulo: str = "IF Temporal Multivariante",
    output_path: Optional[Path] = None,
) -> str:
    """Visualiza IF temporal multivariante con desglose IF/media."""
    import pandas as pd

    times = pd.to_datetime(scores.valid_time.values)
    mask_vals = mascara.values.astype(bool)
    has_componentes = ("if_mask" in mascara.coords) and ("media_mask" in mascara.coords)
    if has_componentes:
        mask_if = mascara.coords["if_mask"].values.astype(bool)
        mask_media = mascara.coords["media_mask"].values.astype(bool)
        mask_if_solo = np.logical_and(mask_if, np.logical_not(mask_media))
        mask_media_solo = np.logical_and(mask_media, np.logical_not(mask_if))
        mask_ambos = np.logical_and(mask_if, mask_media)
    else:
        mask_if_solo = np.zeros_like(mask_vals, dtype=bool)
        mask_media_solo = np.zeros_like(mask_vals, dtype=bool)
        mask_ambos = mask_vals

    score_vals = scores.values
    n_anom = int(mask_vals.sum())

    colores_var = ["steelblue", "darkorange", "seagreen", "orchid"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    for i, (nombre, da) in enumerate(variables.items()):
        media = da.mean(dim=["latitude", "longitude"]).values
        z = (media - media.mean()) / (media.std() + 1e-9)
        color = colores_var[i % len(colores_var)]
        ax1.plot(times, z, color=color, linewidth=1.0, alpha=0.8, label=nombre)

    for t in times[mask_vals]:
        ax1.axvline(t, color="red", alpha=0.12, linewidth=1.2)

    if has_componentes:
        if mask_if_solo.any():
            ax1.scatter(
                times[mask_if_solo],
                np.zeros(int(mask_if_solo.sum())),
                color="tab:blue",
                s=50,
                zorder=5,
                marker="x",
                label=f"IF solo (n={int(mask_if_solo.sum())})",
            )
        if mask_media_solo.any():
            ax1.scatter(
                times[mask_media_solo],
                np.zeros(int(mask_media_solo.sum())),
                color="tab:orange",
                s=50,
                zorder=5,
                marker="+",
                label=f"z-score media solo (n={int(mask_media_solo.sum())})",
            )
        if mask_ambos.any():
            ax1.scatter(
                times[mask_ambos],
                np.zeros(int(mask_ambos.sum())),
                color="tab:red",
                s=60,
                zorder=5,
                marker="v",
                label=f"IF + media (n={int(mask_ambos.sum())})",
            )
    elif mask_vals.any():
        ax1.scatter(
            times[mask_vals],
            np.zeros(n_anom),
            color="red",
            s=60,
            zorder=5,
            marker="v",
            label=f"Timestep anomalo (n={n_anom})",
        )

    ax1.set_ylabel("Z-score (variables normalizadas)", fontsize=11)
    ax1.set_title("Medias espaciales normalizadas - eventos anomalos en rojo", fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3)

    if has_componentes:
        colores_barras = []
        for i in range(len(mask_vals)):
            if mask_ambos[i]:
                colores_barras.append("tab:red")
            elif mask_if_solo[i]:
                colores_barras.append("tab:blue")
            elif mask_media_solo[i]:
                colores_barras.append("tab:orange")
            else:
                colores_barras.append("steelblue")
    else:
        colores_barras = ["red" if m else "steelblue" for m in mask_vals]

    ax2.bar(times, score_vals, color=colores_barras, width=pd.Timedelta(hours=10), alpha=0.75)

    if mask_vals.any():
        umbral = float(score_vals[mask_vals].max())
        ax2.axhline(umbral, color="red", linestyle="--", linewidth=1.5, label=f"Umbral de decision ({umbral:.3f})")
        ax2.legend(fontsize=9)

    ax2.set_xlabel("Fecha", fontsize=11)
    ax2.set_ylabel("Score IF\n(mas negativo = mas anomalo)", fontsize=11)
    ax2.set_title("Score de Anomalia por Timestep (multivariante)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches="tight", dpi=150)
        plt.close()
        return str(output_path)
    plt.show()
    return ""

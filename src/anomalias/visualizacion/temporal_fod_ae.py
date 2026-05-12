"""Visualizaciones para detectores temporales FOD y Autoencoder."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def visualizar_serie_temporal_fod(
    data: xr.DataArray,
    scores: xr.DataArray,
    mascara: xr.DataArray,
    titulo: str = "Ecuacion en diferencias de primer orden - eventos anomalos",
    variable_nombre: str = "Variable",
    output_path: Optional[Path] = None,
) -> str:
    """Visualiza resultados del detector x_t = a*x_(t-1) + b."""
    import pandas as pd

    tiempos = pd.to_datetime(data.valid_time.values)
    serie_media = data.mean(dim=["latitude", "longitude"]).values
    serie_pred = (
        scores.coords["serie_pred"].values
        if "serie_pred" in scores.coords
        else np.full_like(serie_media, np.nan, dtype=float)
    )
    z_resid = (
        scores.coords["zscore_residuo"].values
        if "zscore_residuo" in scores.coords
        else np.full_like(serie_media, np.nan, dtype=float)
    )
    mask_vals = mascara.values.astype(bool)
    umbral = float(mascara.attrs.get("umbral_zscore_residuo", scores.attrs.get("umbral_zscore_residuo", 2.5)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    ax1.plot(tiempos, serie_media, color="steelblue", linewidth=1.2, alpha=0.85, label="Media espacial observada")
    ax1.plot(
        tiempos,
        serie_pred,
        color="tab:green",
        linewidth=1.0,
        alpha=0.9,
        linestyle="--",
        label="Prediccion 1-paso (a*x_(t-1)+b)",
    )

    if mask_vals.any():
        ax1.scatter(
            tiempos[mask_vals],
            serie_media[mask_vals],
            color="tab:red",
            s=45,
            alpha=0.9,
            zorder=5,
            label=f"Anomalos (n={int(mask_vals.sum())})",
        )
        for t in tiempos[mask_vals]:
            ax1.axvline(t, color="tab:red", alpha=0.12, linewidth=1.0)

    ax1.set_ylabel(variable_nombre, fontsize=11)
    ax1.set_title("Serie observada vs prediccion de ecuacion en diferencias", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    colores = ["tab:red" if m else "steelblue" for m in mask_vals]
    ax2.bar(tiempos, z_resid, color=colores, width=pd.Timedelta(hours=10), alpha=0.75)
    ax2.axhline(umbral, color="tab:red", linestyle="--", linewidth=1.2, label=f"+umbral z ({umbral:.2f})")
    ax2.axhline(-umbral, color="tab:red", linestyle="--", linewidth=1.2, label=f"-umbral z ({umbral:.2f})")
    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)

    ax2.set_xlabel("Fecha", fontsize=11)
    ax2.set_ylabel("z-score residuo", fontsize=11)
    ax2.set_title("Anomalia temporal por residuo del modelo", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
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


def visualizar_serie_temporal_autoencoder(
    data: xr.DataArray,
    scores: xr.DataArray,
    mascara: xr.DataArray,
    titulo: str = "Autoencoder temporal - eventos anomalos",
    variable_nombre: str = "Variable",
    output_path: Optional[Path] = None,
) -> str:
    """Visualiza la deteccion temporal por error de reconstruccion."""
    import pandas as pd

    tiempos = pd.to_datetime(data.valid_time.values)
    serie_media = data.mean(dim=["latitude", "longitude"]).values
    mask_vals = mascara.values.astype(bool)

    if "error_reconstruccion" in scores.coords:
        err = scores.coords["error_reconstruccion"].values
    else:
        err = np.full_like(serie_media, np.nan, dtype=float)

    threshold = float(mascara.attrs.get("threshold_error", scores.attrs.get("threshold_error", np.nan)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(titulo, fontsize=13, fontweight="bold")

    ax1.plot(tiempos, serie_media, color="steelblue", linewidth=1.2, alpha=0.85, label="Media espacial")
    if mask_vals.any():
        ax1.scatter(
            tiempos[mask_vals],
            serie_media[mask_vals],
            color="tab:red",
            s=45,
            zorder=5,
            alpha=0.9,
            label=f"Anomalos AE (n={int(mask_vals.sum())})",
        )
        for t in tiempos[mask_vals]:
            ax1.axvline(t, color="tab:red", alpha=0.12, linewidth=1.0)

    ax1.set_ylabel(variable_nombre, fontsize=11)
    ax1.set_title("Serie temporal y timesteps detectados", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    colores = ["tab:red" if m else "steelblue" for m in mask_vals]
    ax2.bar(tiempos, err, color=colores, width=pd.Timedelta(hours=10), alpha=0.75)
    if np.isfinite(threshold):
        ax2.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.3, label=f"Umbral error ({threshold:.4f})")
        ax2.legend(fontsize=9)

    ax2.set_xlabel("Fecha", fontsize=11)
    ax2.set_ylabel("Error reconstruccion", fontsize=11)
    ax2.set_title("Error de reconstruccion del autoencoder", fontweight="bold")
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

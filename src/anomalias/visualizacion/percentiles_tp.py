"""Visualizaciones para anomalias de precipitacion por percentiles."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def visualizar_anomalias_tp_percentiles(
    data_tp: xr.DataArray,
    mascara: xr.DataArray,
    umbrales: xr.DataArray,
    titulo: str,
    output_path: Optional[Path] = None,
) -> str:
    """Visualizacion completa de anomalias de precipitacion por P99."""
    fig = plt.figure(figsize=(16, 12))
    ax_ul = fig.add_subplot(2, 2, 1)
    ax_ur = fig.add_subplot(2, 2, 2)
    ax_bot = fig.add_subplot(2, 1, 2)

    umbrales.plot(
        ax=ax_ul,
        cmap="Blues",
        add_colorbar=True,
        cbar_kwargs={"label": "Umbral P99 (mm)"},
    )
    ax_ul.set_title("Umbral P99 Local de Precipitacion", fontweight="bold")
    ax_ul.set_xlabel("Longitud")
    ax_ul.set_ylabel("Latitud")

    dias_anom = mascara.sum(dim="valid_time").astype(int)
    dias_anom.plot(
        ax=ax_ur,
        cmap="YlOrRd",
        add_colorbar=True,
        cbar_kwargs={"label": "Num. dias anomalos"},
    )
    ax_ur.set_title("Dias con Precipitacion Extrema por Punto", fontweight="bold")
    ax_ur.set_xlabel("Longitud")
    ax_ur.set_ylabel("Latitud")

    puntos_por_dia = mascara.sum(dim=["latitude", "longitude"]).values
    tiempos = mascara.valid_time.values

    ax_bot.bar(np.arange(len(tiempos)), puntos_por_dia, color="steelblue", alpha=0.75, width=0.8)
    ax_bot.set_ylabel("N. Puntos con tp > P99", fontsize=11)
    ax_bot.set_xlabel("Fecha", fontsize=11)
    ax_bot.set_title("Puntos con Precipitacion Extrema por Dia", fontweight="bold")
    ax_bot.grid(True, alpha=0.3, axis="y")

    step = max(1, len(tiempos) // 15)
    ax_bot.set_xticks(np.arange(0, len(tiempos), step))
    try:
        import pandas as pd

        fechas_str = [str(pd.Timestamp(t).date()) for t in tiempos[::step]]
        ax_bot.set_xticklabels(fechas_str, rotation=45, ha="right", fontsize=9)
    except Exception:
        pass

    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches="tight", dpi=150)
        plt.close()
        return str(output_path)
    plt.show()
    return ""

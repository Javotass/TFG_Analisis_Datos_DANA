"""Visualizaciones espaciales para Isolation Forest."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm


def visualizar_mapa_scores_if(
    scores: xr.DataArray,
    mascara: xr.DataArray,
    titulo: str,
    variable_nombre: str = "",
    output_path: Optional[Path] = None,
) -> str:
    """Mapa de scores IF y mascara de anomalias."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    vmin, vmax = float(scores.min()), float(scores.max())
    vcenter = float(scores.median()) if vmin < 0 < vmax else (vmin + vmax) / 2
    try:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        scores.plot(
            ax=axes[0],
            cmap="RdYlBu",
            norm=norm,
            add_colorbar=True,
            cbar_kwargs={"label": "Score IF (mas negativo = mas anomalo)"},
        )
    except Exception:
        scores.plot(ax=axes[0], cmap="RdYlBu", add_colorbar=True)

    axes[0].set_title(f"Score de Anomalia\n{variable_nombre}", fontweight="bold")
    axes[0].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")

    has_componentes = ("if_mask" in mascara.coords) and ("media_mask" in mascara.coords)
    if has_componentes:
        if_mask = mascara.coords["if_mask"].values.astype(bool)
        media_mask = mascara.coords["media_mask"].values.astype(bool)
        categorias = np.zeros_like(mascara.values, dtype=int)
        categorias[np.logical_and(if_mask, np.logical_not(media_mask))] = 1
        categorias[np.logical_and(media_mask, np.logical_not(if_mask))] = 2
        categorias[np.logical_and(if_mask, media_mask)] = 3

        cat_da = xr.DataArray(categorias, dims=mascara.dims, coords=mascara.coords)
        cmap = ListedColormap(["#f0f0f0", "#1f77b4", "#ff7f0e", "#d62728"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        cat_da.plot(
            ax=axes[1],
            cmap=cmap,
            norm=norm,
            add_colorbar=True,
            cbar_kwargs={"label": "Tipo de anomalia", "ticks": [0, 1, 2, 3]},
        )
        cbar = axes[1].collections[-1].colorbar
        cbar.set_ticklabels(["Normal", "IF", "z-score media", "IF + media"])
    else:
        mascara.astype(int).plot(
            ax=axes[1],
            cmap="Reds",
            vmin=0,
            vmax=1,
            add_colorbar=True,
            cbar_kwargs={"label": "Anomalia (1=si, 0=no)"},
        )

    n_anom = int(mascara.sum())
    n_total = mascara.size
    pct = 100 * n_anom / n_total
    axes[1].set_title(f"Puntos Anomalos: {n_anom}/{n_total} ({pct:.1f}%)", fontweight="bold")
    axes[1].set_xlabel("Longitud")
    axes[1].set_ylabel("Latitud")

    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches="tight", dpi=150)
        plt.close()
        return str(output_path)
    plt.show()
    return ""


def visualizar_mapa_anomalias_if(
    data: xr.DataArray,
    mascara: xr.DataArray,
    titulo: str,
    variable_nombre: str = "",
    output_path: Optional[Path] = None,
) -> str:
    """Valor medio temporal y ubicacion de puntos anomalos."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    media = data.mean(dim="valid_time")
    media.plot(ax=axes[0], cmap="RdYlBu_r", add_colorbar=True)
    axes[0].set_title(f"Valor Medio Temporal\n{variable_nombre}", fontweight="bold")
    axes[0].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")

    media.plot(ax=axes[1], cmap="RdYlBu_r", add_colorbar=True, alpha=0.7)

    has_componentes = ("if_mask" in mascara.coords) and ("media_mask" in mascara.coords)
    if has_componentes:
        if_mask = mascara.coords["if_mask"].values.astype(bool)
        media_mask = mascara.coords["media_mask"].values.astype(bool)
        mask_if_solo = np.logical_and(if_mask, np.logical_not(media_mask))
        mask_media_solo = np.logical_and(media_mask, np.logical_not(if_mask))
        mask_ambos = np.logical_and(if_mask, media_mask)

        for mask_plot, color, marker, label in [
            (mask_if_solo, "tab:blue", "x", "IF solo"),
            (mask_media_solo, "tab:orange", "+", "z-score media solo"),
            (mask_ambos, "tab:red", "o", "IF + media"),
        ]:
            idx = np.where(mask_plot)
            if len(idx[0]) > 0:
                axes[1].scatter(
                    mascara.longitude.values[idx[1]],
                    mascara.latitude.values[idx[0]],
                    c=color,
                    s=22,
                    marker=marker,
                    label=f"{label} ({len(idx[0])})",
                    zorder=5,
                    alpha=0.85,
                )
        axes[1].legend(fontsize=9)
    else:
        lats_anom = mascara.latitude.values[np.where(mascara.values)[0]]
        lons_anom = mascara.longitude.values[np.where(mascara.values)[1]]
        if len(lats_anom) > 0:
            axes[1].scatter(
                lons_anom,
                lats_anom,
                c="black",
                s=18,
                marker="x",
                label=f"Anomalos ({len(lats_anom)})",
                zorder=5,
                alpha=0.8,
            )
            axes[1].legend(fontsize=10)

    axes[1].set_title("Puntos Anomalos (x)", fontweight="bold")
    axes[1].set_xlabel("Longitud")
    axes[1].set_ylabel("Latitud")

    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches="tight", dpi=150)
        plt.close()
        return str(output_path)
    plt.show()
    return ""


def visualizar_resumen_multivariante(
    scores: xr.DataArray,
    mascara: xr.DataArray,
    titulo: str,
    output_path: Optional[Path] = None,
) -> str:
    """Mapa resumen del IF multivariante: scores y mascara."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    try:
        vmin, vmax = float(scores.min()), float(scores.max())
        vcenter = float(scores.median()) if vmin < 0 < vmax else (vmin + vmax) / 2
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        scores.plot(
            ax=axes[0],
            cmap="RdYlBu",
            norm=norm,
            add_colorbar=True,
            cbar_kwargs={"label": "Score IF multivariante"},
        )
    except Exception:
        scores.plot(ax=axes[0], cmap="RdYlBu", add_colorbar=True)

    variables = scores.attrs.get("variables", [])
    axes[0].set_title(f"Score Multivariante\n{', '.join(variables)}", fontweight="bold")
    axes[0].set_xlabel("Longitud")
    axes[0].set_ylabel("Latitud")

    mascara.astype(int).plot(
        ax=axes[1],
        cmap="Reds",
        vmin=0,
        vmax=1,
        add_colorbar=True,
        cbar_kwargs={"label": "Anomalia multivariante (1=si)"},
    )
    n_anom = int(mascara.sum())
    n_total = mascara.size
    pct = 100 * n_anom / n_total
    axes[1].set_title(f"Anomalias Multivariantes: {n_anom}/{n_total} ({pct:.1f}%)", fontweight="bold")
    axes[1].set_xlabel("Longitud")
    axes[1].set_ylabel("Latitud")

    fig.suptitle(titulo, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(str(output_path), bbox_inches="tight", dpi=150)
        plt.close()
        return str(output_path)
    plt.show()
    return ""

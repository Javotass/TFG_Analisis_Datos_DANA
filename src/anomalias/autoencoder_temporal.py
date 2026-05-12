"""
Deteccion temporal de anomalias con autoencoder.

El autoencoder se implementa con un MLP que aprende a reconstruir un vector
de features temporales por timestep. Los timesteps con alto error de
reconstruccion se marcan como anomalos.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.neural_network import MLPRegressor


def _fit_fod_params(y: np.ndarray) -> tuple[float, float]:
    """Ajusta x_t = a*x_(t-1) + b por minimos cuadrados."""
    x_prev = y[:-1]
    x_curr = y[1:]
    mask = np.isfinite(x_prev) & np.isfinite(x_curr)
    if int(mask.sum()) < 5:
        return 0.0, float(np.nanmean(x_curr)) if x_curr.size else 0.0
    A = np.column_stack((x_prev[mask], np.ones(int(mask.sum()))))
    target = x_curr[mask]
    coef, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    return float(coef[0]), float(coef[1])


def detectar_anomalias_autoencoder_temporal(
    data: xr.DataArray,
    hidden_layer_sizes: tuple[int, ...] = (16, 6, 16),
    contamination: float = 0.05,
    random_state: int = 42,
    max_iter: int = 2000,
) -> tuple[xr.DataArray, xr.DataArray, dict]:
    """
    Detecta anomalias temporales por error de reconstruccion del autoencoder.

    Se construye una serie media espacial x_t y, para cada t>=1, un vector
    de features dinamicas:
      [x_t, x_(t-1), delta_t, x_pred_t, resid_t, z_resid_t]

    El autoencoder aprende el patron normal de estos vectores. Timesteps con
    error de reconstruccion elevado se marcan como anomalos.
    """
    required = {"valid_time", "latitude", "longitude"}
    missing = required - set(data.dims)
    if missing:
        raise ValueError(f"Faltan dimensiones en data: {missing}")

    times = data.valid_time.values
    y = data.mean(dim=["latitude", "longitude"]).values.astype(float)
    n_time = y.size
    if n_time < 8:
        raise ValueError("Se requieren al menos 8 timesteps para autoencoder temporal")

    a, b = _fit_fod_params(y)

    y_pred = np.full_like(y, np.nan, dtype=float)
    y_pred[1:] = a * y[:-1] + b
    resid = y - y_pred
    resid[0] = np.nan

    resid_mu = float(np.nanmean(resid))
    resid_std = float(np.nanstd(resid))
    if resid_std > 0:
        z_resid = (resid - resid_mu) / resid_std
    else:
        z_resid = np.zeros_like(resid)
        z_resid[0] = np.nan

    x_t = y[1:]
    x_lag = y[:-1]
    delta = x_t - x_lag
    pred_t = y_pred[1:]
    resid_t = resid[1:]
    zres_t = z_resid[1:]

    X = np.column_stack((x_t, x_lag, delta, pred_t, resid_t, zres_t))

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    X_std = (X - mu[np.newaxis, :]) / sd[np.newaxis, :]
    X_std = np.where(np.isnan(X_std), 0.0, X_std)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=25,
        validation_fraction=0.15,
    )
    model.fit(X_std, X_std)

    X_rec = model.predict(X_std)
    err = np.mean((X_std - X_rec) ** 2, axis=1)

    contamination = float(contamination)
    if not (0.0 < contamination < 0.5):
        raise ValueError("'contamination' debe estar en (0, 0.5)")
    threshold = float(np.quantile(err, 1.0 - contamination))
    mask_eval = err >= threshold

    err_mu = float(np.mean(err))
    err_sd = float(np.std(err))
    if err_sd > 0:
        z_err = (err - err_mu) / err_sd
    else:
        z_err = np.zeros_like(err)

    scores = np.zeros(n_time, dtype=float)
    scores[1:] = -z_err

    mask_full = np.zeros(n_time, dtype=bool)
    mask_full[1:] = mask_eval

    err_full = np.full(n_time, np.nan, dtype=float)
    err_full[1:] = err
    zerr_full = np.full(n_time, np.nan, dtype=float)
    zerr_full[1:] = z_err

    scores_da = xr.DataArray(
        scores,
        dims=["valid_time"],
        coords={"valid_time": times},
        name="ae_temporal_score",
        attrs={
            "long_name": "Autoencoder temporal anomaly score",
            "variable": str(data.name),
            "modelo": "MLP autoencoder (reconstruccion)",
            "contamination": contamination,
            "threshold_error": threshold,
            "a_fod": a,
            "b_fod": b,
        },
    )
    scores_da = scores_da.assign_coords(
        serie_media=("valid_time", y),
        serie_pred_fod=("valid_time", y_pred),
        residuo_fod=("valid_time", resid),
        zscore_residuo_fod=("valid_time", z_resid),
        error_reconstruccion=("valid_time", err_full),
        zscore_error_reconstruccion=("valid_time", zerr_full),
    )

    mascara_da = xr.DataArray(
        mask_full,
        dims=["valid_time"],
        coords={"valid_time": times},
        name="ae_temporal_anomalia",
        attrs={
            "long_name": "Autoencoder temporal anomaly mask",
            "variable": str(data.name),
            "contamination": contamination,
            "threshold_error": threshold,
        },
    )

    n_anom = int(mask_eval.sum())
    n_eval = int(err.size)
    stats = {
        "n_total_timesteps": int(n_time),
        "n_timesteps_evaluados": n_eval,
        "num_anomalias": n_anom,
        "porcentaje_anomalias": round(100.0 * n_anom / n_eval, 3),
        "contamination": contamination,
        "threshold_error": threshold,
        "error_media": err_mu,
        "error_std": err_sd,
        "score_min": float(np.min(scores[1:])),
        "score_max": float(np.max(scores[1:])),
        "score_media": float(np.mean(scores[1:])),
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "max_iter": int(max_iter),
        "random_state": int(random_state),
        "n_features": int(X_std.shape[1]),
        "a_fod": a,
        "b_fod": b,
    }

    return scores_da, mascara_da, stats

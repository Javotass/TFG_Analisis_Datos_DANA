"""Detectores de anomalias basados en Isolation Forest."""

from .univariante_espacial import detectar_anomalias_isolation_forest
from .univariante_temporal import detectar_anomalias_isolation_forest_temporal
from .multivariante_espacial import detectar_anomalias_isolation_forest_multivariante
from .multivariante_temporal import detectar_anomalias_isolation_forest_temporal_multivariante

__all__ = [
    "detectar_anomalias_isolation_forest",
    "detectar_anomalias_isolation_forest_temporal",
    "detectar_anomalias_isolation_forest_multivariante",
    "detectar_anomalias_isolation_forest_temporal_multivariante",
]

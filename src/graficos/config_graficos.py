"""Configuración de estilo para los gráficos."""

import matplotlib.pyplot as plt
import seaborn as sns


def configurar_estilo():
    """Configura el estilo de los gráficos."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 10

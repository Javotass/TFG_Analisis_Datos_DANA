"""
Ejemplo rápido de uso de los métodos de detección de anomalías - Fase 1.

Este script muestra cómo usar individualmente cada método baseline.
"""

import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# Importar métodos de detección
from src.anomalias import (
    detectar_anomalias_persistencia,
    detectar_anomalias_zscore,
    detectar_anomalias_iqr,
    detectar_anomalias_stl
)


def crear_serie_ejemplo():
    """Crea una serie temporal de ejemplo con anomalías artificiales."""
    # Crear índice temporal (30 días, datos horarios)
    inicio = datetime(2024, 1, 1)
    tiempos = [inicio + timedelta(hours=i) for i in range(720)]  # 30 días
    
    # Generar datos sintéticos
    t = np.arange(len(tiempos))
    
    # Componentes
    tendencia = 0.01 * t  # Tendencia leve
    estacional = 5 * np.sin(2 * np.pi * t / 24)  # Ciclo diario
    ruido = np.random.normal(0, 0.5, len(t))
    
    # Serie base
    valores = 20 + tendencia + estacional + ruido
    
    # Insertar anomalías artificiales
    # Anomalía 1: Pico abrupto
    valores[100] = valores[100] + 15
    # Anomalía 2: Caída abrupta
    valores[200] = valores[200] - 12
    # Anomalía 3: Valor extremo alto
    valores[400] = valores[400] + 20
    # Anomalía 4: Secuencia anómala
    valores[500:505] = valores[500:505] + 10
    
    # Crear DataArray de xarray
    data = xr.DataArray(
        valores,
        coords={'valid_time': tiempos},
        dims=['valid_time'],
        name='temperatura'
    )
    
    return data


def main():
    """Ejemplo de deteccion de anomalias."""
    
    print("=" * 70)
    print("EJEMPLO: Deteccion de Anomalias - Fase 1")
    print("=" * 70)
    
    # Crear datos de ejemplo
    print("\n1. Generando serie temporal de ejemplo...")
    data = crear_serie_ejemplo()
    print(f"   * Serie creada: {len(data)} puntos temporales")
    print(f"   * Anomalias artificiales insertadas: 4")
    
    # Metodo 1: Persistencia
    print("\n2. Metodo: Persistencia")
    print("   " + "-" * 50)
    anomalias_pers, stats_pers = detectar_anomalias_persistencia(
        data,
        umbral_cambio=3.0,
        ventana_comparacion=1
    )
    print(f"   Metodo: {stats_pers['metodo']}")
    print(f"   Anomalias detectadas: {stats_pers['num_anomalias']}")
    print(f"   Porcentaje: {stats_pers['porcentaje_anomalias']:.2f}%")
    print(f"   Umbral superior: {stats_pers['umbral_superior']:.2f}")
    print(f"   Umbral inferior: {stats_pers['umbral_inferior']:.2f}")
    
    # Metodo 2: Z-score
    print("\n3. Metodo: Z-score")
    print("   " + "-" * 50)
    anomalias_zscore, stats_zscore = detectar_anomalias_zscore(
        data,
        umbral=3.0,
        usar_mediana=False
    )
    print(f"   Metodo: {stats_zscore['metodo']}")
    print(f"   Anomalias detectadas: {stats_zscore['num_anomalias']}")
    print(f"   Porcentaje: {stats_zscore['porcentaje_anomalias']:.2f}%")
    print(f"   Media: {stats_zscore['centro']:.2f}")
    print(f"   Desv. estandar: {stats_zscore['escala']:.2f}")
    
    # Metodo 3: IQR
    print("\n4. Metodo: IQR (Rango Intercuartilico)")
    print("   " + "-" * 50)
    anomalias_iqr, stats_iqr = detectar_anomalias_iqr(
        data,
        factor=1.5
    )
    print(f"   Metodo: {stats_iqr['metodo']}")
    print(f"   Anomalias detectadas: {stats_iqr['num_anomalias']}")
    print(f"   Porcentaje: {stats_iqr['porcentaje_anomalias']:.2f}%")
    print(f"   Anomalias bajas: {stats_iqr['num_anomalias_bajas']}")
    print(f"   Anomalias altas: {stats_iqr['num_anomalias_altas']}")
    print(f"   Q1: {stats_iqr['q1']:.2f}, Q3: {stats_iqr['q3']:.2f}")
    
    # Metodo 4: STL
    print("\n5. Metodo: STL + Analisis de Residuos")
    print("   " + "-" * 50)
    try:
        anomalias_stl, stats_stl = detectar_anomalias_stl(
            data,
            periodo_estacional=24,  # Ciclo diario
            umbral_residuos=3.0
        )
        print(f"   Metodo: {stats_stl['metodo']}")
        print(f"   Anomalias detectadas: {stats_stl['num_anomalias']}")
        print(f"   Porcentaje: {stats_stl['porcentaje_anomalias']:.2f}%")
        print(f"   Varianza explicada: {stats_stl.get('varianza_explicada_pct', 0):.1f}%")
        print(f"   Std residuos: {stats_stl.get('std_residuos', 0):.2f}")
    except Exception as e:
        print(f"   ! Error: {e}")
        print(f"   (Instalar: pip install statsmodels)")
    
    # Resumen comparativo
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"{'Metodo':<25} {'Anomalias':>12} {'Porcentaje':>12}")
    print("-" * 70)
    print(f"{'Persistencia':<25} {stats_pers['num_anomalias']:>12} {stats_pers['porcentaje_anomalias']:>11.2f}%")
    print(f"{'Z-score':<25} {stats_zscore['num_anomalias']:>12} {stats_zscore['porcentaje_anomalias']:>11.2f}%")
    print(f"{'IQR':<25} {stats_iqr['num_anomalias']:>12} {stats_iqr['porcentaje_anomalias']:>11.2f}%")
    try:
        print(f"{'STL + Residuos':<25} {stats_stl['num_anomalias']:>12} {stats_stl['porcentaje_anomalias']:>11.2f}%")
    except:
        pass
    print("=" * 70)
    
    # Consenso
    print("\n6. Analisis de Consenso")
    print("   " + "-" * 50)
    print("   Puntos detectados por multiples metodos:")
    
    # Encontrar anomalias detectadas por al menos 2 metodos
    consenso_2 = (anomalias_pers.astype(int) + 
                  anomalias_zscore.astype(int) + 
                  anomalias_iqr.astype(int)) >= 2
    
    num_consenso = int(consenso_2.sum().values)
    print(f"   > Detectadas por >=2 metodos: {num_consenso} anomalias")
    
    if num_consenso > 0:
        print(f"   > Estas tienen mayor probabilidad de ser anomalias reales")
    
    print("\n" + "=" * 70)
    print("* Ejemplo completado")
    print("=" * 70)
    print("\nPara analisis completo con datos ERA5-Land:")
    print("  > Ejecutar: python deteccion_anomalias_fase1.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

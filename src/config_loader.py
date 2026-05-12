"""Módulo para cargar y validar la configuración desde JSON."""

import os
import json
from datetime import datetime
from pathlib import Path


def cargar_configuracion(ruta_config: str) -> dict:
    """Carga la configuración desde un archivo JSON."""
    if not os.path.exists(ruta_config):
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta_config}")

    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    campos_requeridos = ["fecha_inicio", "fecha_fin", "variables", "area", "horas", "output_folder"]
    for campo in campos_requeridos:
        if campo not in config:
            raise ValueError(f"Falta el campo obligatorio '{campo}' en {ruta_config}")

    return config


def obtener_rutas(config: dict) -> dict:
    """Genera las rutas de archivos basándose en la configuración."""
    fecha_inicio = config["fecha_inicio"]
    fecha_fin = config["fecha_fin"]
    fecha_inicio_dt = datetime.strptime(fecha_inicio, "%d/%m/%Y")
    fecha_fin_dt = datetime.strptime(fecha_fin, "%d/%m/%Y")
    fecha_inicio_str = fecha_inicio_dt.strftime("%Y%m%d")
    fecha_fin_str = fecha_fin_dt.strftime("%Y%m%d")

    data_folder = config.get("output_folder", "datos")
    graficos_folder = config.get("graficos_folder", "graficos")

    Path(data_folder).mkdir(exist_ok=True)
    Path(graficos_folder).mkdir(exist_ok=True)
    
    return {
        "data_folder": data_folder,
        "graficos_folder": graficos_folder,
        "data_file": f"{data_folder}/era5_land_{fecha_inicio_str}_{fecha_fin_str}.nc",
        "log_file": f"{data_folder}/exploracion_{fecha_inicio_str}_{fecha_fin_str}_resultado.json",
        "csv_file": f"{data_folder}/variables_meteorologicas_{fecha_inicio_str}_{fecha_fin_str}.csv",
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
        "fecha_inicio_str": fecha_inicio_str,
        "fecha_fin_str": fecha_fin_str
    }


def inicializar_resultados(config: dict, rutas: dict) -> dict:
    """Inicializa el diccionario de resultados para el análisis."""
    return {
        "configuracion_origen": "config.json",
        "archivo_datos": rutas["data_file"],
        "fecha_analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "periodo": {
            "fecha_inicio": rutas["fecha_inicio"],
            "fecha_fin": rutas["fecha_fin"]
        },
        "area": config.get("area", {}),
        "dimensiones": {},
        "variables": {},
        "coordenadas": {},
        "estadisticas": {},
        "valores_faltantes": {},
        "graficos_generados": [],
        "archivos_exportados": []
    }

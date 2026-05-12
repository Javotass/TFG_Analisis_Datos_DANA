import cdsapi
import os
import json
from datetime import datetime, timedelta

CONFIG_FILE = "config.json"

def cargar_configuracion(ruta_config):
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

user_config = cargar_configuracion(CONFIG_FILE)
os.makedirs(user_config["output_folder"], exist_ok=True)
c = cdsapi.Client()

fecha_inicio = user_config["fecha_inicio"]
fecha_fin = user_config["fecha_fin"]
fecha_inicio_dt = datetime.strptime(fecha_inicio, "%d/%m/%Y")
fecha_fin_dt = datetime.strptime(fecha_fin, "%d/%m/%Y")

def generar_rango_fechas(inicio, fin):
    """Genera años, meses y días únicos del rango de fechas."""
    fechas = []
    fecha_actual = inicio
    while fecha_actual <= fin:
        fechas.append(fecha_actual)
        fecha_actual += timedelta(days=1)
    
    years = sorted(list(set(f.strftime("%Y") for f in fechas)))
    months = sorted(list(set(f.strftime("%m") for f in fechas)))
    days = sorted(list(set(f.strftime("%d") for f in fechas)))
    
    return years, months, days, fechas

years, months, days, todas_las_fechas = generar_rango_fechas(fecha_inicio_dt, fecha_fin_dt)

api_config = {
    "product_type": ["reanalysis"],
    "variable": user_config["variables"],
    "year": years,
    "month": months,
    "day": days,
    "time": user_config["horas"],
    "area": [
        user_config["area"]["norte"],
        user_config["area"]["oeste"],
        user_config["area"]["sur"],
        user_config["area"]["este"]
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

fecha_inicio_str = fecha_inicio_dt.strftime("%Y%m%d")
fecha_fin_str = fecha_fin_dt.strftime("%Y%m%d")
output_file = f"{user_config['output_folder']}/era5_land_{fecha_inicio_str}_{fecha_fin_str}.nc"
log_file = f"{user_config['output_folder']}/era5_land_{fecha_inicio_str}_{fecha_fin_str}_log.json"

info_descarga = {
    "configuracion_origen": CONFIG_FILE,
    "fecha_inicio": fecha_inicio,
    "fecha_fin": fecha_fin,
    "total_dias": len(todas_las_fechas),
    "variables": api_config["variable"],
    "area": user_config["area"],
    "horas": user_config["horas"],
    "archivo_salida": output_file,
    "estado": "iniciando",
    "timestamp_inicio": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(log_file, 'w', encoding='utf-8') as f:
    json.dump(info_descarga, f, indent=4, ensure_ascii=False)

print("=" * 50)
print("DESCARGA ERA5-Land")
print("=" * 50)
print(f"Configuración cargada desde: {CONFIG_FILE}")
print(f"Log guardado en: {log_file}")
print("=" * 50)

c.retrieve(
    "reanalysis-era5-land",
    api_config,
    output_file
)

info_descarga["estado"] = "completado"
info_descarga["timestamp_fin"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
info_descarga["tamano_archivo_mb"] = round(os.path.getsize(output_file) / (1024**2), 2)

with open(log_file, 'w', encoding='utf-8') as f:
    json.dump(info_descarga, f, indent=4, ensure_ascii=False)

print(f"\n[OK] Descarga completada: {output_file}")
print(f"[OK] Log guardado: {log_file}")
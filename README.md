# TFG — Análisis de Datos Meteorológicos ERA5-Land

Trabajo de Fin de Grado de Javier Revilla. Análisis exploratorio y detección de anomalías sobre datos meteorológicos de reanálisis ERA5-Land para la Comunidad Valenciana.

---

## Descripción

El proyecto descarga datos climáticos reales de la API de Copernicus Climate Data Store (CDS) en formato NetCDF y aplica sobre ellos:

1. **Análisis exploratorio**: estadísticas descriptivas, visualizaciones estáticas (SVG) y mapas animados interactivos (HTML).
2. **Detección de anomalías**: cinco algoritmos de distinta naturaleza, con perspectiva espacial (¿dónde?) y temporal (¿cuándo?).

Todo se ejecuta en local mediante scripts de línea de comandos. No hay base de datos ni backend web.

---

## Fuente de datos

**ERA5-Land** — reanálisis climático de la ECMWF distribuido por Copernicus.

| Parámetro | Valor |
|---|---|
| Resolución espacial | ~9 km (0.1° × 0.1°) |
| Resolución temporal | Horaria (00:00 y 12:00 UTC) |
| Formato | NetCDF (`.nc`) |
| Período configurado | 01/06/2023 – 15/08/2023 |
| Dominio geográfico | Comunidad Valenciana (N 40.90°, S 37.80°, O −1.90°, E 0.90°) |

### Variables

| Variable ERA5 | Nombre | Unidad interna | Unidad en análisis |
|---|---|---|---|
| `2m_temperature` | Temperatura a 2 m | K | °C |
| `10m_u_component_of_wind` | Componente U del viento | m/s | m/s |
| `10m_v_component_of_wind` | Componente V del viento | m/s | m/s |
| `total_precipitation` | Precipitación total | m | mm |
| `surface_pressure` | Presión superficial | Pa | hPa |

La velocidad escalar del viento (`wind_speed = √(u10² + v10²)`) se calcula en código.

---

## Requisitos

- Python 3.10+
- Credenciales de Copernicus CDS en `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <tu-uid>:<tu-api-key>
```

Crear cuenta y obtener la clave en: https://cds.climate.copernicus.eu/how-to-api

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

Dependencias principales: `cdsapi`, `xarray`, `netCDF4`, `matplotlib`, `seaborn`, `pandas`, `numpy`, `scipy`, `plotly`, `geopandas`, `shapely`, `scikit-learn`, `statsmodels`, `cartopy`.

---

## Uso

### 1. Configurar el análisis

Editar `config.json` en la raíz del proyecto:

```json
{
    "fecha_inicio": "01/06/2023",
    "fecha_fin": "15/08/2023",
    "variables": ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
                  "total_precipitation", "surface_pressure"],
    "area": { "nombre": "Comunidad Valenciana", "norte": 40.90, "oeste": -1.90, "sur": 37.80, "este": 0.90 },
    "horas": ["00:00", "12:00"],
    "output_folder": "datos",
    "graficos_folder": "graficos",
    "archivo_entrada": "datos/era5_land_20230601_20230815.nc"
}
```

Para cambiar el período o región basta con modificar este archivo; no hay fechas ni rutas hardcodeadas en el código.

### 2. Descargar datos ERA5-Land

```bash
python descarga_era5.py
```

Descarga el NetCDF en `datos/` y genera un log JSON con metadatos. Solo necesario la primera vez o al cambiar el período.

### 3. Análisis exploratorio

```bash
python exploracion_era5.py
```

Genera estadísticas descriptivas, todas las visualizaciones y exporta los resultados a CSV y JSON.

### 4. Detección de anomalías

```bash
python deteccion_anomalias_fase1.py
```

Ejecuta los cinco algoritmos de detección y guarda los resultados en `resultados_anomalias/fase1/`.

### 5. Demo con datos sintéticos (opcional)

```bash
python ejemplo_fase1.py
```

Demuestra el uso de todos los métodos de detección sin necesidad de datos descargados.

---

## Estructura del proyecto

```
TFG_Analisis_Datos-main/
│
├── config.json                        # Configuración global (fechas, área, variables)
├── requirements.txt                   # Dependencias Python
│
├── descarga_era5.py                   # Descarga via API Copernicus CDS
├── exploracion_era5.py                # Análisis exploratorio completo
├── deteccion_anomalias_fase1.py       # Ejecución de los cinco detectores
├── ejemplo_fase1.py                   # Demo con datos sintéticos
│
├── datos/                             # NetCDF descargados + CSV/JSON de resultados
├── graficos/                          # Visualizaciones generadas
│   ├── series_temporales/
│   ├── mapas/
│   ├── distribuciones/
│   ├── correlaciones/
│   └── ciclo_mensual/
│
├── resultados_anomalias/              # Outputs de la detección de anomalías
│   └── fase1/
│       ├── graficos/
│       └── comparativa_temporal/
│
└── src/                               # Paquete Python con toda la lógica
    ├── config_loader.py               # Carga y validación del config.json
    ├── estadisticas.py                # Estadísticas descriptivas
    ├── exportador.py                  # Exportación a CSV y JSON
    ├── visualizaciones.py             # Orquestador de gráficos
    │
    ├── graficos/                      # Submódulos de visualización
    │   ├── series_temporales.py
    │   ├── mapas.py                   # Mapas estáticos (SVG) y animados (HTML/Plotly)
    │   ├── distribuciones.py
    │   ├── correlaciones.py
    │   └── ciclo_mensual.py
    │
    └── anomalias/                     # Submódulos de detección de anomalías
        ├── config_anomalias.py        # Hiperparámetros centralizados
        ├── percentiles.py             # Percentil P99 (precipitación)
        ├── diferencias_primer_orden.py # Modelo AR(1)
        ├── autoencoder_temporal.py    # Autoencoder MLP
        ├── isolation_forest/
        │   ├── univariante_espacial.py
        │   ├── univariante_temporal.py
        │   ├── multivariante_espacial.py
        │   └── multivariante_temporal.py
        └── visualizacion/
```

---

## Algoritmos de detección de anomalías

Se implementan cinco algoritmos con dos perspectivas: **espacial** (¿qué zonas son atípicas?) y **temporal** (¿qué instantes de tiempo son atípicos?).

| Algoritmo | Perspectiva | Variables | Descripción |
|---|---|---|---|
| Isolation Forest univariante | Espacial + Temporal | Todas | Cada punto (lat, lon) o timestep como muestra; los N timesteps/puntos como features |
| Isolation Forest multivariante | Espacial + Temporal | Todas combinadas | Concatena todas las variables en el vector de features |
| Percentil P99 | Espacial+temporal | Solo precipitación | Umbral local por celda; evita el problema zero-inflated del IF |
| AR(1) — diferencias de 1er orden | Temporal | Todas | Residuos del modelo `x_t = a·x_{t-1} + b` normalizados por z-score |
| Autoencoder MLP temporal | Temporal | Todas | Features dinámicas AR(1) comprimidas por un MLP `[16, 6, 16]`; alto error de reconstrucción = anomalía |

---

## Salidas generadas

### `datos/`
- `era5_land_YYYYMMDD_YYYYMMDD.nc` — datos meteorológicos descargados
- `variables_meteorologicas_YYYYMMDD_YYYYMMDD.csv` — series temporales exportadas
- `exploracion_YYYYMMDD_YYYYMMDD_resultado.json` — estadísticas completas

### `graficos/`
- Series temporales por variable (SVG)
- Mapas medios por variable (SVG)
- Mapas animados frame a frame con slider temporal (HTML interactivo)
- Histogramas de distribución (SVG)
- Matriz de correlación (SVG)
- Ciclo mensual (SVG)

### `resultados_anomalias/fase1/`
- Mapas y series de anomalías por algoritmo y variable (SVG)
- `estadisticas.json` — resumen numérico de anomalías detectadas
- `comparativa_temporal/anomalias_comparativa_resumen.json` — comparación entre algoritmos

---

## Notas

- Los archivos NetCDF pueden pesar varios GB para períodos largos. El período configurado (jun–ago 2023, Comunidad Valenciana) es manejable.
- La descarga puede tardar minutos dependiendo de la carga de los servidores de Copernicus.
- Los mapas animados HTML son autocontenidos y se abren en cualquier navegador sin servidor.
- El autoencoder puede tardar varios minutos con `max_iter=2000`; con pocas semanas de datos es rápido.
- `random_state=42` está fijado en todos los algoritmos para garantizar reproducibilidad.
- Todos los gráficos estáticos se guardan en formato SVG (escalable), excepto los mapas animados que son HTML.

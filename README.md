# Dashboard Macrosad - Análisis Sociodemográfico

Este proyecto proporciona un dashboard interactivo para el análisis del perfil sociodemográfico de los usuarios de Macrosad, con visualizaciones avanzadas y pronósticos temporales basados en datos históricos.

## 📋 Descripción

El dashboard ofrece un análisis completo que incluye:
- Métricas clave de usuarios y servicios
- Visualizaciones interactivas de datos demográficos
- Análisis geográfico con mapas interactivos
- Pronósticos temporales usando modelos de suavizado exponencial
- Análisis de cambios porcentuales entre períodos

## 🚀 Características Principales

### Análisis de Datos
- **Métricas Clave**: Número de usuarios, horas PIA promedio, edad promedio, distribución por género
- **Filtros Interactivos**: Por grado de dependencia, género, edad y ubicación
- **Visualizaciones**: Gráficos de barras, diagramas de caja, dispersión y líneas de tendencia

### Análisis Geográfico
- Mapas interactivos con datos de usuarios
- Filtros por centro, sexo, edad y horas PIA
- Visualización con escala de colores basada en horas de servicio

### Pronósticos Temporales
- Pronósticos de horas de servicio a 5 años usando Exponential Smoothing
- Análisis por centro individual
- Pronósticos de grado de dependencia
- Intervalos de confianza del 95%

### Análisis de Cambios
- Comparación de cambios porcentuales entre 2020-2024
- Métricas por centro
- Exportación automática de datos procesados

## 📁 Estructura del Proyecto

```
Dashboard-Macrosad/
├── src/
│   ├── app.py                        # Aplicación principal de Streamlit
│   ├── preprocessing.py              # Funciones de preprocesamiento de datos
│   ├── expedientes_processed.xlsx    # Datos procesados de expedientes
│   ├── db_usuarios_preprocessed.xlsx # Datos preprocessados de usuarios
│   ├── mapdata.xlsx                  # Datos geográficos
│   └── exports/                      # Carpeta de exportaciones automáticas
├── data/                             # Carpeta de datos adicionales
├── exports/                          # Exportaciones del proyecto
├── ContratosContraprestacion.csv     # Datos de contratos (históricos)
├── Expedientes.csv                   # Datos originales de expedientes
└── README.md
```

## 🛠️ Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Dependencias
Instala las dependencias necesarias:

```powershell
pip install streamlit plotly pandas numpy statsmodels openpyxl pydeck
```

### Configuración del Entorno
1. Clona o descarga el proyecto
2. Navega al directorio del proyecto:
   ```powershell
   cd "c:\Users\friva\Desktop\practicas\Dashboard Macrosad"
   ```

### Ejecución
Ejecuta la aplicación desde la carpeta raíz del proyecto:

```powershell
streamlit run src/app.py
```

La aplicación se abrirá automáticamente en tu navegador web predeterminado, típicamente en `http://localhost:8501`.

## 📊 Fuentes de Datos

El dashboard utiliza los siguientes archivos de datos:

- **`ContratosContraprestacion.csv`**: Datos históricos de contratos con información temporal
- **`Expedientes.csv`**: Datos originales de expedientes de usuarios
- **`src/expedientes_processed.xlsx`**: Datos procesados de expedientes
- **`src/db_usuarios_preprocessed.xlsx`**: Datos de usuarios preprocessados (datos NAIHA)
- **`src/mapdata.xlsx`**: Datos geográficos para visualización en mapas

## 🔧 Funcionalidades Técnicas

### Preprocesamiento de Datos
El archivo `src/preprocessing.py` incluye:
- Conversión de campos numéricos
- Limpieza de categorías de grado de dependencia
- Binning de horas PIA en intervalos
- Manejo de valores faltantes

### Modelos de Pronóstico
- **Modelo**: Exponential Smoothing (Holt-Winters)
- **Componentes**: Tendencia aditiva y estacionalidad aditiva
- **Período estacional**: 12 meses
- **Horizonte de pronóstico**: 36-60 meses

### Funciones Principales de `app.py`

#### Carga de Datos
- `load_data()`: Carga y limpia datos principales de expedientes
- `load_contratos_data()`: Procesa datos históricos de contratos
- `load_contratos_data_by_centro()`: Agrupa datos por centro
- `load_contratos_data_grado()`: Procesa datos de grado de dependencia

#### Modelos Predictivos
- `create_time_series_forecast()`: Genera pronósticos temporales generales
- `create_forecast_by_centro()`: Pronósticos específicos por centro
- `create_time_series_forecast_grado()`: Pronósticos de grado de dependencia
- `create_forecast_by_centro_grado()`: Pronósticos de grado por centro

#### Análisis de Cambios
- `calculate_yearly_changes()`: Calcula cambios porcentuales 2020-2024

## 📈 Uso del Dashboard

### Filtros Disponibles
- **Grado de Dependencia**: Filtro múltiple por categorías
- **Género**: Selección de H/M
- **Rango de Edad**: Slider interactivo
- **Centro**: Búsqueda por texto

### Secciones del Dashboard

#### 1. Métricas Clave
Cards con estadísticas principales:
- Número total de usuarios
- Promedio de horas PIA
- Edad promedio
- Distribución por género

#### 2. Visualizaciones Interactivas
- **Gráfico de Barras**: Horas PIA por grado de dependencia y sexo
- **Diagrama de Caja**: Distribución de horas PIA
- **Dispersión**: Relación edad vs. horas PIA con línea de tendencia
- **Línea de Tendencia**: Evolución de horas PIA por edad
- **Barras Apiladas**: Comparación de cuidado doméstico vs. personal (datos NAIHA)

#### 3. Análisis Geográfico
- Mapa interactivo con puntos coloreados por horas PIA
- Filtros específicos del mapa
- Información contextual en hover

#### 4. Pronósticos Temporales
- **Pronóstico General**: Horas de servicio proyectadas a 5 años
- **Pronósticos por Centro**: Análisis individual por ubicación
- **Pronósticos de Grado**: Evolución del grado de dependencia
- Intervalos de confianza del 95%

#### 5. Cambios Porcentuales
- Comparación 2020-2024 por centro
- Métricas de cambio en horas y grado de dependencia

## 📤 Exportación de Datos

El sistema genera automáticamente archivos Excel en la carpeta `src/exports/` con:
- Promedios anuales por centro
- Datos de cambios porcentuales
- Listas de centros analizados
- Timestamps para control de versiones

Formato de archivos: `analysis_export_YYYYMMDD_HHMMSS.xlsx`

## 🎨 Personalización

### Estilos CSS
El archivo `app.py` incluye estilos personalizados:
- **Cards de métricas**: Fondo azul con texto blanco
- **Separadores de sección**: Diferentes estilos decorativos
- **Elementos interactivos**: Gradientes y sombras

### Paleta de Colores
- **Azul principal**: `#34a3d3`
- **Azul claro**: `#5cb3d9`
- **Verde para pronósticos**: `#4CAF50`
- **Azul oscuro para datos históricos**: `#1976D2`

### Configuración de Gráficos
- Uso de Plotly para visualizaciones interactivas
- Escalas de colores personalizadas
- Configuración responsiva

## 🔍 Solución de Problemas

### Errores Comunes

1. **Error de archivo no encontrado**
   - Verifica que todos los archivos CSV y Excel estén en las ubicaciones correctas
   - Ejecuta desde la carpeta raíz del proyecto

2. **Error de dependencias**
   ```powershell
   pip install --upgrade streamlit plotly pandas numpy statsmodels openpyxl pydeck
   ```

3. **Error de memoria en pronósticos**
   - Reduce el número de centros seleccionados
   - Verifica que los datos tengan al menos 24 meses de historia

### Logs y Debugging
- Los errores se muestran directamente en la interfaz de Streamlit
- Revisa la consola de PowerShell para errores detallados

## 🚀 Mejoras Futuras

- Integración con bases de datos en tiempo real
- Modelos de machine learning más avanzados
- Dashboard móvil responsivo
- Alertas automáticas por email
- API REST para integración externa

## 🤝 Contribución

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📞 Soporte

Si encuentras algún problema:
1. Revisa la sección de solución de problemas
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de ejecutar desde la ubicación correcta
4. Reporta issues detallados con logs de error

## 📄 Licencia

Este proyecto está desarrollado para el análisis sociodemográfico de usuarios Macrosad.

---

**Versión**: 1.0  
**Última actualización**: Junio 2025  
**Desarrollado con**: Python, Streamlit, Plotly, Pandas, NumPy, Statsmodels

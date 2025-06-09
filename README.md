# Dashboard Macrosad - Análisis Sociodemográfico


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dashboard-macrosad.streamlit.app/)


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


---

**Versión**: 1.0  
**Última actualización**: Junio 2025  
**Desarrollado con**: Python, Streamlit, Plotly, Pandas, NumPy, Statsmodels

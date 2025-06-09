import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from preprocessing import load_and_clean_data
import os
import pandas as pd
import pydeck as pdk
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.markdown(
    """
    <style>
    p, div[class^="css"] {
        font-size: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

card_style_general = """
    <div style="padding: 10px; border-radius: 10px; background-color: #34a3d3; text-align: center; color: white;">
        <h4 style="margin: 0;">{title}</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">{value}</p>
    </div>
"""

# Section separator styles
section_divider = """
    <div style="margin: 30px 0; border-bottom: 3px solid #34a3d3; width: 100%;"></div>
"""

section_divider_with_icon = """
    <div style="text-align: center; margin: 30px 0;">
        <div style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #34a3d3, #5cb3d9); 
                    border-radius: 20px; color: white; font-weight: bold;">
            📊 {section_name}
        </div>
        <div style="margin-top: 10px; border-bottom: 2px solid #e0e0e0; width: 100%;"></div>
    </div>
"""

section_divider_elegant = """
    <div style="margin: 40px 0 20px 0; text-align: center;">
        <div style="display: inline-block; position: relative;">
            <div style="height: 2px; width: 150px; background: linear-gradient(to right, transparent, #34a3d3, transparent); 
                        display: inline-block; vertical-align: middle;"></div>
            <span style="margin: 0 20px; color: #34a3d3; font-size: 18px; font-weight: bold; vertical-align: middle;">✨</span>
            <div style="height: 2px; width: 150px; background: linear-gradient(to right, transparent, #34a3d3, transparent); 
                        display: inline-block; vertical-align: middle;"></div>
        </div>
    </div>
"""

section_divider_modern = """
    <div style="margin: 30px 0;">
        <div style="height: 4px; background: linear-gradient(90deg, #34a3d3 0%, #5cb3d9 50%, #34a3d3 100%); 
                    border-radius: 2px; box-shadow: 0 2px 4px rgba(52, 163, 211, 0.3);"></div>
    </div>
"""


# File path to the Excel file
data_file = os.path.join(os.path.dirname(__file__), 'expedientes_processed.xlsx')

# Load and preprocess the data
@st.cache_data
def load_data():
    """Load data from expedientes_processed.xlsx"""
    # Load the data
    df = pd.read_excel(data_file)
    
    # Convert edad to numeric, coercing errors to NaN
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    
    # Convert other important numeric columns
    numeric_columns = ['horas_pia', 'horas_dom', 'horas_per', 'Horas_Mes_Cp']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle categorical columns
    if 'grado_dep' in df.columns:
        df['grado_dep'] = df['grado_dep'].astype(str)
    if 'gradocat1' in df.columns:
        df['gradocat1'] = df['gradocat1'].astype(str)
    
    
    return df

@st.cache_data
def load_contratos_data():
    """Load and process ContratosContraprestacion data for time series analysis."""
    contratos_file = os.path.join(os.path.dirname(__file__), '..', 'ContratosContraprestacion.csv')
    df = pd.read_csv(contratos_file, sep=';')
    
    # Select required columns and create a copy
    data = df[['Fecha_Alta', 'Total_Horas_Mes']].copy()
    
    # Convert to datetime and clean data
    data['Fecha_Alta'] = pd.to_datetime(data['Fecha_Alta'], format='mixed')
    data = data.dropna(subset=['Fecha_Alta', 'Total_Horas_Mes'])
    data['Total_Horas_Mes'] = pd.to_numeric(data['Total_Horas_Mes'], errors='coerce')
    data = data.dropna(subset=['Total_Horas_Mes'])
    data = data[data['Fecha_Alta'].dt.year >= 2021]
    
    # Group by month
    data['Month'] = data['Fecha_Alta'].dt.to_period('M')
    data_grouped = data.groupby('Month').agg({'Total_Horas_Mes': 'mean'}).reset_index()
    data_grouped.columns = ['Month', 'Average_Hours']
    data_grouped['Month'] = data_grouped['Month'].dt.to_timestamp()
    
    return data_grouped

@st.cache_data
def load_contratos_data_by_centro():
    """Load and process ContratosContraprestacion data grouped by centro for time series analysis."""
    contratos_file = os.path.join(os.path.dirname(__file__), '..', 'ContratosContraprestacion.csv')
    df = pd.read_csv(contratos_file, sep=';')
    
    # Select required columns and create a copy
    data = df[['Fecha_Alta', 'Total_Horas_Mes', 'Centro']].copy()
    
    # Convert to datetime and clean data
    data['Fecha_Alta'] = pd.to_datetime(data['Fecha_Alta'], format='mixed')
    data = data.dropna(subset=['Fecha_Alta', 'Total_Horas_Mes', 'Centro'])
    data['Total_Horas_Mes'] = pd.to_numeric(data['Total_Horas_Mes'], errors='coerce')
    data = data.dropna(subset=['Total_Horas_Mes'])
    data = data[data['Fecha_Alta'].dt.year >= 2021]
    
    # Group by month and centro
    data['Month'] = data['Fecha_Alta'].dt.to_period('M')
    data_grouped = data.groupby(['Month', 'Centro'])['Total_Horas_Mes'].mean().reset_index()
    data_grouped['Month'] = data_grouped['Month'].dt.to_timestamp()
    
    return data_grouped

@st.cache_data
def create_time_series_forecast(data_grouped):
    """Create time series forecast using historical data."""
    # Convert to time series with proper frequency
    ts = data_grouped.set_index('Month')['Average_Hours']

    # Create model and fit
    model = ExponentialSmoothing(
        ts,
        seasonal_periods=12,
        trend='add',
        seasonal='add',
        initialization_method="estimated"
    ).fit()    # Generate forecast for 60 months (5 years)
    steps = 60
    
    # Get the last date from historical data
    last_date = ts.index[-1]
    
    # Create future dates starting from the last historical date
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='M')[1:]
    
    # Generate the forecast
    forecast = pd.Series(
        model.forecast(steps).values,
        index=future_dates
    )
    
    # Calculate confidence intervals
    residuals = model.resid
    std_dev = residuals.std()
    ci_lower = pd.Series(forecast.values - 1.96 * std_dev, index=future_dates)
    ci_upper = pd.Series(forecast.values + 1.96 * std_dev, index=future_dates)
    
    return ts, forecast, ci_lower, ci_upper

@st.cache_data
def create_forecast_by_centro(data_grouped):
    """Create time series forecast for each centro."""
    forecasts = {}
    
    # Get unique centers
    centers = data_grouped['Centro'].unique()
    
    for centro in centers:
        # Filter data for this centro
        centro_data = data_grouped[data_grouped['Centro'] == centro].copy()
        centro_data = centro_data.set_index('Month')['Total_Horas_Mes']
        
        # Check if we have enough data points for forecasting
        if len(centro_data) < 24:  # Need at least 2 years of data
            continue
            
        try:
            # Create and fit model
            model = ExponentialSmoothing(
                centro_data,
                seasonal_periods=12,
                trend='add',
                seasonal='add',
                initialization_method="estimated"
            ).fit()
            
            # Generate forecast
            steps = 36
            last_date = centro_data.index[-1]
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='M')[1:]
            
            # Store historical and forecast data
            forecasts[centro] = {
                'historical': centro_data,
                'forecast': pd.Series(
                    model.forecast(steps).values,
                    index=future_dates
                )
            }
        except Exception:
            # Silently skip centers where forecast cannot be generated
            continue
    
    return forecasts

@st.cache_data
def load_contratos_data_grado():
    """Load and process ContratosContraprestacion data for grado dependencia time series analysis."""
    contratos_file = os.path.join(os.path.dirname(__file__), '..', 'ContratosContraprestacion.csv')
    df = pd.read_csv(contratos_file, sep=';')
    
    # Select required columns and create a copy
    data = df[['Fecha_Alta', 'gradodependencia']].copy()
    
    # Convert to datetime and clean data
    data['Fecha_Alta'] = pd.to_datetime(data['Fecha_Alta'], format='mixed')
    data = data.dropna(subset=['Fecha_Alta', 'gradodependencia'])
    data['gradodependencia'] = pd.to_numeric(data['gradodependencia'], errors='coerce')
    data = data.dropna(subset=['gradodependencia'])
    data = data[data['Fecha_Alta'].dt.year >= 2021]
    
    # Group by month
    data['Month'] = data['Fecha_Alta'].dt.to_period('M')
    data_grouped = data.groupby('Month').agg({'gradodependencia': 'mean'}).reset_index()
    data_grouped.columns = ['Month', 'Average_Grado']
    data_grouped['Month'] = data_grouped['Month'].dt.to_timestamp()
    
    return data_grouped

@st.cache_data
def load_contratos_data_by_centro_grado():
    """Load and process ContratosContraprestacion data grouped by centro for grado dependencia time series analysis."""
    contratos_file = os.path.join(os.path.dirname(__file__), '..', 'ContratosContraprestacion.csv')
    df = pd.read_csv(contratos_file, sep=';')
    
    # Select required columns and create a copy
    data = df[['Fecha_Alta', 'gradodependencia', 'Centro']].copy()
    
    # Convert to datetime and clean data
    data['Fecha_Alta'] = pd.to_datetime(data['Fecha_Alta'], format='mixed')
    data = data.dropna(subset=['Fecha_Alta', 'gradodependencia', 'Centro'])
    data['gradodependencia'] = pd.to_numeric(data['gradodependencia'], errors='coerce')
    data = data.dropna(subset=['gradodependencia'])
    data = data[data['Fecha_Alta'].dt.year >= 2021]
      # Group by month and centro
    data['Month'] = data['Fecha_Alta'].dt.to_period('M')
    data_grouped = data.groupby(['Month', 'Centro'])['gradodependencia'].mean().reset_index()
    data_grouped['Month'] = data_grouped['Month'].dt.to_timestamp()
    
    return data_grouped

@st.cache_data
def create_time_series_forecast_grado(data_grouped):
    """Create time series forecast for grado dependencia using historical data."""
    # Convert to time series with proper frequency
    ts = data_grouped.set_index('Month')['Average_Grado']

    # Create model and fit
    model = ExponentialSmoothing(
        ts,
        seasonal_periods=12,
        trend='add',
        seasonal='add',
        initialization_method="estimated"
    ).fit()
    # Generate forecast for 60 months (5 years)
    steps = 60
    
    # Get the last date from historical data
    last_date = ts.index[-1]
    
    # Create future dates starting from the last historical date
    future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='M')[1:]
    
    # Generate the forecast
    forecast = pd.Series(
        model.forecast(steps).values,
        index=future_dates
    )
    
    # Calculate confidence intervals
    residuals = model.resid
    std_dev = residuals.std()
    ci_lower = pd.Series(forecast.values - 1.96 * std_dev, index=future_dates)
    ci_upper = pd.Series(forecast.values + 1.96 * std_dev, index=future_dates)
    
    return ts, forecast, ci_lower, ci_upper

@st.cache_data
def create_forecast_by_centro_grado(data_grouped):
    """Create time series forecast for grado dependencia for each centro."""
    forecasts = {}
    
    # Get unique centers
    centers = data_grouped['Centro'].unique()
    
    for centro in centers:        # Filter data for this centro
        centro_data = data_grouped[data_grouped['Centro'] == centro].copy()
        centro_data = centro_data.set_index('Month')['gradodependencia']
        
        # Check if we have enough data points for forecasting
        if len(centro_data) < 24:  # Need at least 2 years of data
            continue
            
        try:
            # Create and fit model
            model = ExponentialSmoothing(
                centro_data,
                seasonal_periods=12,
                trend='add',
                seasonal='add',
                initialization_method="estimated"
            ).fit()
            
            # Generate forecast
            steps = 36
            last_date = centro_data.index[-1]
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq='M')[1:]
            
            # Store historical and forecast data
            forecasts[centro] = {
                'historical': centro_data,
                'forecast': pd.Series(
                    model.forecast(steps).values,
                    index=future_dates
                )
            }
        except Exception:
            # Silently skip centers where forecast cannot be generated
            continue
    
    return forecasts

@st.cache_data
def calculate_yearly_changes():
    """Calculate yearly changes in horas pia and grado dependencia by center."""
    contratos_file = os.path.join(os.path.dirname(__file__), '..', 'ContratosContraprestacion.csv')
    df = pd.read_csv(contratos_file, sep=';')
    
    # Convert to datetime and clean data
    df['Fecha_Alta'] = pd.to_datetime(df['Fecha_Alta'], format='mixed')
    df['year'] = df['Fecha_Alta'].dt.year
    

    
    # Ensure numeric values and remove invalid entries
    df['Total_Horas_Mes'] = pd.to_numeric(df['Total_Horas_Mes'], errors='coerce')
    df['gradodependencia'] = pd.to_numeric(df['gradodependencia'], errors='coerce')
    
    # Remove rows with missing values
    df = df.dropna(subset=['Centro', 'Total_Horas_Mes', 'gradodependencia'])
    
    # Convert to numeric before aggregation and ensure float type
    df['Total_Horas_Mes'] = pd.to_numeric(df['Total_Horas_Mes'], errors='coerce').astype('float64')
    df['gradodependencia'] = pd.to_numeric(df['gradodependencia'], errors='coerce').astype('float64')
    
    # Group by and calculate means, ensuring numeric values
    yearly_avg = df.groupby(['Centro', 'year'], as_index=False).agg({
        'Total_Horas_Mes': lambda x: x.astype('float64').mean(),
        'gradodependencia': lambda x: x.astype('float64').mean()
    })
    
    
    # Get earliest and latest years with data
    start_year = 2020
    end_year = 2024
    
    # Get data for start and end years
    data_start = yearly_avg[yearly_avg['year'] == start_year].set_index('Centro')
    data_end = yearly_avg[yearly_avg['year'] == end_year].set_index('Centro')
    
    
    # Get all unique centers
    all_centers = pd.Index(yearly_avg['Centro'].unique())
    
    # Initialize changes DataFrame with all centers
    changes = pd.DataFrame(index=all_centers)
    changes['cambio_horas'] = np.nan
    changes['cambio_grado'] = np.nan        # Find centers present in both periods
    common_centers = data_start.index.intersection(data_end.index)
    # Ensure numeric data types
    data_end.loc[common_centers, 'Total_Horas_Mes'] = pd.to_numeric(data_end.loc[common_centers, 'Total_Horas_Mes'], errors='coerce')
    data_start.loc[common_centers, 'Total_Horas_Mes'] = pd.to_numeric(data_start.loc[common_centers, 'Total_Horas_Mes'], errors='coerce')
    data_end.loc[common_centers, 'gradodependencia'] = pd.to_numeric(data_end.loc[common_centers, 'gradodependencia'], errors='coerce')
    data_start.loc[common_centers, 'gradodependencia'] = pd.to_numeric(data_start.loc[common_centers, 'gradodependencia'], errors='coerce')
      # Calculate changes with numeric data
    # Handle Total_Horas_Mes changes
    end_hours = data_end.loc[common_centers, 'Total_Horas_Mes'].astype('float64')
    start_hours = data_start.loc[common_centers, 'Total_Horas_Mes'].astype('float64')
    changes.loc[common_centers, 'cambio_horas'] = (
        (end_hours - start_hours) / start_hours.replace(0, np.nan) * 100
    )

    # Handle gradodependencia changes
    end_grado = data_end.loc[common_centers, 'gradodependencia'].astype('float64')
    start_grado = data_start.loc[common_centers, 'gradodependencia'].astype('float64')
    changes.loc[common_centers, 'cambio_grado'] = (
        (end_grado - start_grado) / start_grado.replace(0, np.nan) * 100
    )
    
    # Add period information
    changes['periodo'] = f"{start_year}-{end_year}"
    
    # Reset index to make Centro a column
    changes = changes.reset_index()
    changes.rename(columns={'index': 'Centro'}, inplace=True)
    
    # Clean any potential infinite values (division by zero)
    changes = changes.replace([np.inf, -np.inf], np.nan)

    # Fill any remaining NaN values with 0
    changes = changes.fillna(0)

    return changes

def main():
    st.title("Análisis del Perfil Sociodemográfico Usuarios Macrosad")
    st.markdown("""
    Este panel proporciona un análisis del perfil sociodemográfico de los usuarios de Macrosad, centrándose en métricas clave como horas PIA y grado de dependencia, y visualizaciones interactivas.
    """)

    # Load data
    try:
        data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return


    # Ensure 'Centro' is a string before applying .str.contains
    data['Centro'] = data['Centro'].astype(str)    # Ensure gradocat1 is treated as a string
    data['gradocat1'] = data['gradocat1'].astype(str)
      # Round edad to the nearest first decimal
    data['edad'] = data['edad'].round(1)
    
    # Filtros en la barra lateral
    st.sidebar.header("Filtros")
    
    # Filter out NaN values from gradocat1 before creating the multiselect
    valid_dependency_options = data['gradocat1'].dropna().unique()
    valid_dependency_options2 = valid_dependency_options[ valid_dependency_options != "nan" ]
    dependency_filter = st.sidebar.multiselect("Grado de Dependencia", options=data['gradocat1'].unique(), default=data['gradocat1'].unique()) 
      # Add Centro (Proyectos) filter
    active_user_filter = st.sidebar.multiselect("Estado de Usuario", options=data['Estadebaja'].unique(), default=data['Estadebaja'].unique())
    centro_filter = st.sidebar.multiselect("Proyectos", options=data['Centro'].unique(), default=data['Centro'].unique())

    
    gender_filter = st.sidebar.multiselect("Género", options=data['Sexo'].unique(), default=data['Sexo'].unique())
    age_filter = st.sidebar.slider("Rango de Edad", int(data['edad'].min()), int(data['edad'].max()), (int(data['edad'].min()), int(data['edad'].max())))
    horas_filter = st.sidebar.slider("Rango de Horas PIA", 0, int(data['Horas_Mes_Cp'].max()), (0, int(data['Horas_Mes_Cp'].max())))
    
    # Apply Filters
    if 'Estadebaja' in data.columns:
        filtered_data = data[
            (data['gradocat1'].isin(dependency_filter)) &
            (data['Estadebaja'].isin(active_user_filter)) &
            (data['Centro'].isin(centro_filter)) &
            (data['Sexo'].isin(gender_filter)) &
            (data['edad'].between(*age_filter)) &
            (data['Horas_Mes_Cp'].between(*horas_filter))
        ]
    else:
        # If Estadebaja column doesn't exist, skip that filter
        filtered_data = data[
            (data['gradocat1'].isin(dependency_filter)) &
            (data['Centro'].isin(centro_filter)) &
            (data['Sexo'].isin(gender_filter)) &
            (data['edad'].between(*age_filter)) &
            (data['Horas_Mes_Cp'].between(*horas_filter))
        ]
    
    filtered_data2 = data[
            (data['gradocat1'].isin(dependency_filter)) &
            (data['Estadebaja'].isin(active_user_filter)) &
            (data['Centro'].isin(centro_filter)) &
            (data['Sexo'].isin(gender_filter)) 
        ]
    
    # Métricas clave (now based on filtered data)
    st.subheader("Métricas Clave")
    col1, col2, col3, col4 = st.columns(4)
    
    # Custom styled metric cards using filtered data
    with col1:
        st.markdown(
            card_style_general.format(
                title="Número de Usuarios",
                value=filtered_data2['ID'].count()
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            card_style_general.format(
                title="Horas PIA Promedio",
                value=f"{filtered_data['Horas_Mes_Cp'].mean():.2f}" if len(filtered_data) > 0 else "0.00"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            card_style_general.format(
                title="Edad Promedio",
                value=f"{filtered_data['edad'].mean():.2f}" if len(filtered_data) > 0 else "0.00"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        if len(filtered_data) > 0:
            gender_counts = filtered_data['Sexo'].value_counts(normalize=True) * 100
            gender_display = f"{gender_counts.get('Hombre', 0):.1f} / {gender_counts.get('Mujer', 0):.1f}"
        else:
            gender_display = "0.0 / 0.0"
        st.markdown(
            card_style_general.format(
                title="% Hombres/Mujeres",
                value=gender_display
            ),
            unsafe_allow_html=True
        )
    # Check if filtered data is empty
    if len(filtered_data) == 0:
        st.warning("⚠️ No hay datos que coincidan con los filtros seleccionados. Por favor, ajusta los filtros para ver los resultados.")
        st.stop()
    
    # Gráficos Interactivos
    st.markdown(section_divider_with_icon.format(section_name="VISUALIZACIONES INTERACTIVAS"), unsafe_allow_html=True)
    st.subheader("Gráficos Interactivos")    # Gráfico de barras de Horas PIA promedio por grado de dependencia y sexo
    st.markdown("### Promedio de Horas PIA por Grado de Dependencia y Sexo")
    # Calculate average Horas_Mes_Cp by gradocat1 and Sexo
    avg_data = filtered_data.groupby(['gradocat1', 'Sexo'])['Horas_Mes_Cp'].mean().reset_index()
    fig1 = px.bar(avg_data, x='gradocat1', y='Horas_Mes_Cp', color='Sexo', barmode='group', 
                  title="Promedio de Horas PIA por Grado de Dependencia y Sexo",
                  labels={'gradocat1': 'Grado de Dependencia', 'Horas_Mes_Cp': 'Promedio de Horas PIA', 'Sexo': 'Sexo'})
    fig1.update_xaxes(title_text='Grado de Dependencia')
    fig1.update_yaxes(title_text='Promedio de Horas PIA')
    st.plotly_chart(fig1)# Diagrama de caja de Horas_Mes_Cp vs. gradocat1
    st.markdown("### Diagrama de Caja de Horas PIA vs. Grado de Dependencia")
    # Sort the categories and create category order
    category_order = sorted(filtered_data['gradocat1'].unique())
    fig2 = px.box(filtered_data, x='gradocat1', y='Horas_Mes_Cp', color='gradocat1', 
                  title="Horas PIA por Grado de Dependencia",
                  labels={'gradocat1': 'Grado de Dependencia', 'Horas_Mes_Cp': 'Horas PIA'},
                  category_orders={'gradocat1': category_order})    # Set y-axis range from 0 to 100
    fig2.update_layout(yaxis_range=[0, 100])
    fig2.update_xaxes(title_text='Grado de Dependencia')
    fig2.update_yaxes(title_text='Horas PIA')
    st.plotly_chart(fig2)
    
    # Diagrama de dispersión de edad vs. Horas_Mes_Cp con línea de tendencia    
    st.markdown("### Diagrama de Dispersión de Edad vs. Horas PIA con Línea de Tendencia")
    fig3 = px.scatter(filtered_data, x='edad', y='Horas_Mes_Cp', trendline='ols', color='gradocat1', 
                      title="Edad vs. Horas PIA", opacity=0.3,
                      labels={'edad': 'Edad', 'Horas_Mes_Cp': 'Horas PIA', 'gradocat1': 'Grado de Dependencia'})
    fig3.update_layout(yaxis_range=[0, 120])
    fig3.update_xaxes(title_text='Edad')
    fig3.update_yaxes(title_text='Horas PIA')
    st.plotly_chart(fig3)# Línea de tendencia de Horas_Mes_Cp promedio por edad
    st.markdown("### Línea de Tendencia de Horas PIA Promedio por Edad")
    avg_pia_by_age = filtered_data.groupby('edad')['Horas_Mes_Cp'].mean().reset_index()
    fig4 = px.line(avg_pia_by_age, x='edad', y='Horas_Mes_Cp', title="Horas PIA Promedio por Edad")
    st.plotly_chart(fig4)

    # Load additional data for stacked bar chart
    db_processed_file = os.path.join(os.path.dirname(__file__), 'db_usuarios_preprocessed.xlsx')
    db_data = pd.read_excel(db_processed_file, engine='openpyxl')
    
    # Gráficos de barras apiladas comparando horas_dom vs. horas_per por grado_dep
    st.markdown("### Gráfico de Barras Apiladas: Horas de Cuidado Doméstico vs. Personal por Grado de Dependencia (datos de NAIHA)")
    # Group data by grado_dep and sum the hours
    stacked_data = db_data.groupby(['grado_dep'])[['horas_dom', 'horas_per']].sum().reset_index()
    fig5 = go.Figure(data=[
        go.Bar(name='Horas de Cuidado Doméstico', x=stacked_data['grado_dep'], y=stacked_data['horas_dom']),
        go.Bar(name='Horas de Cuidado Personal', x=stacked_data['grado_dep'], y=stacked_data['horas_per'])
    ])
    fig5.update_layout(barmode='stack', title="Cuidado Doméstico vs. Personal por Grado de Dependencia")
    st.plotly_chart(fig5)
      # Map section separator
    st.markdown(section_divider_with_icon.format(section_name="ANÁLISIS GEOGRÁFICO"), unsafe_allow_html=True)
    
    # Use filtered data for map (already filtered by sidebar controls)
    df_map_src = filtered_data.copy()

    # Rename columns for compatibility
    df_map_src.rename(columns={"Gis_Longitud": "lon", "Gis_Latitud": "lat"}, inplace=True)

    # Ensure required columns are numeric
    df_map_src['edad'] = pd.to_numeric(df_map_src['edad'], errors='coerce')
    df_map_src['Horas_Mes_Cp'] = pd.to_numeric(df_map_src['Horas_Mes_Cp'], errors='coerce')
    
    # Round edad to the nearest first decimal
    df_map_src['edad'] = df_map_src['edad'].round(1)

    # Drop rows with missing values in critical columns
    critical_columns = ['lon', 'lat', 'Centro', 'Horas_Mes_Cp', 'edad', 'Sexo']
    df_map_src.dropna(subset=critical_columns, inplace=True)

    # Fill missing values in non-critical columns
    df_map_src.fillna({'Centro': 'Unknown', 'Tipo_Usuario': 'Unknown', 'Sexo': 'Unknown'}, inplace=True)

    # Display map with color differentiation using Plotly
    st.subheader("Mapa de Usuarios por Horas PIA")
    
    # Use the already filtered data for map display
    plotly_map_data = df_map_src# Create a scatter map with Plotly and color by Horas_Mes_Cp
    fig_map = px.scatter_mapbox(
        plotly_map_data,
        lat="lat",
        lon="lon",
        color="Horas_Mes_Cp",
        hover_name="Centro",
        hover_data={"lat": False, "lon": False, "edad": True, "Sexo": True, "Horas_Mes_Cp": True},
        zoom=7,
        height=600,
        color_continuous_scale="YlOrRd",  # Red scale: light red (low hours) to dark red (high hours)
        title="Usuarios por Horas PIA"
    )

    fig_map.update_layout(
        mapbox_style="carto-positron",
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig_map)

    # Time Series Forecast Section
    st.markdown(section_divider_with_icon.format(section_name="PRONÓSTICOS TEMPORALES"), unsafe_allow_html=True)
    st.subheader("Pronóstico de Horas de Servicio (5 años)")
    st.markdown("""
    Esta sección muestra el análisis de la evolución temporal de las horas de servicio y su proyección
    para los próximos 5 años, basado en datos históricos reales.
    """)

    # Load and process time series data
    try:
        ts_data = load_contratos_data()
        
        # Generate forecast
        ts, forecast, ci_lower, ci_upper = create_time_series_forecast(ts_data)
        
        # Create Plotly figure for time series
        fig_ts = go.Figure()
        
        # Add historical data
        fig_ts.add_trace(go.Scatter(
            x=ts.index,
            y=ts.values,
            name='Datos Históricos',
            line=dict(color='#1976D2', width=1),
            mode='lines+markers',
            marker=dict(size=3)
        ))
        
        # Add forecast
        fig_ts.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name='Pronóstico',
            line=dict(color='#4CAF50', width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=3)
        ))
        
        # Add confidence intervals
        fig_ts.add_trace(go.Scatter(
            x=forecast.index,
            y=ci_upper,
            name='Intervalo Superior',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=forecast.index,
            y=ci_lower,
            name='Intervalo de Confianza 95%',
            fill='tonexty',
            fillcolor='rgba(76, 175, 80, 0.2)',
            line=dict(width=0)
        ))
        
        # Update layout
        fig_ts.update_layout(
            title='Pronóstico de Horas de Servicio (5 años)',
            xaxis_title='Fecha',
            yaxis_title='Horas Promedio por Mes',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Add vertical line at May 2025
        fig_ts.add_vline(
            x="2025-05-01",
            line_width=1,
            line_color="white",
            line_dash="solid",
            showlegend=False
        )
          # Add vertical line at May 2025
        fig_ts.add_vline(
            x="2025-05-01",
            line_width=1,
            line_color="white",
            line_dash="solid",
            showlegend=False
        )

        # Display the plot
        st.plotly_chart(fig_ts)
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Promedio Histórico",
            f"{ts.mean():.1f} horas"
        )
        col2.metric(
            "Tendencia",
            "↗️ Creciente" if forecast.values[-5] > forecast.values[0] else "↘️ Decreciente"
        )
        col3.metric(
            "Pronóstico Final",
            f"{forecast.values[-1]:.1f} horas"        )        # Add forecast by centro section
        st.markdown("### Pronóstico por Proyectos")
        st.markdown("""
        Esta sección muestra la evolución y pronóstico de horas de servicio para cada proyecto.
        Las líneas sólidas representan datos históricos y las líneas punteadas son pronósticos.
        """)

        # Load and process time series data by centro
        ts_data_by_centro = load_contratos_data_by_centro()
        forecasts_by_centro = create_forecast_by_centro(ts_data_by_centro)
        
        # Add center filter
        all_centers = list(forecasts_by_centro.keys())
        all_centers2 = [center for center in all_centers if center != "CAMAS "]
        
        selected_centers = st.multiselect(
            "Seleccionar Proyectos",
            options=all_centers,
            default=all_centers2,
            help="Selecciona los proyectos que deseas visualizar en el gráfico"
        )
        
        # Create Plotly figure for multi-series forecast
        fig_ts_centro = go.Figure()
        
        # Color palette for different centers
        colors = px.colors.qualitative.Set3
          # Add traces for each selected centro
        for i, centro in enumerate(selected_centers):
            if centro in forecasts_by_centro:
                data = forecasts_by_centro[centro]
                color = colors[i % len(colors)]  # Cycle through colors if more centers than colors
              # Add historical data
            fig_ts_centro.add_trace(go.Scatter(
                x=data['historical'].index,
                y=data['historical'].values,
                name=centro,  # Only show center name
                line=dict(color=color, width=2),
                mode='lines+markers',
                marker=dict(size=3),
                showlegend=True
            ))
            
            # Add forecast
            fig_ts_centro.add_trace(go.Scatter(
                x=data['forecast'].index,
                y=data['forecast'].values,
                name=centro,  # Same name as historical for shared legend entry
                line=dict(color=color, width=2, dash='dash'),
                mode='lines+markers',
                marker=dict(size=3),
                showlegend=False  # Hide from legend
            ))        # Update layout
        fig_ts_centro.update_layout(
            title='Pronóstico de Horas de Servicio por Proyectos (3 años)',
            xaxis_title='Fecha',
            yaxis_title='Horas Promedio por Mes',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            height=600  # Make the graph taller to accommodate the legend
        )

        # Add vertical line at May 2025
        fig_ts_centro.add_vline(
            x="2025-05-01",
            line_width=1,
            line_color="white",
            line_dash="solid",
            showlegend=False
        )
        
        # Display the plot
        st.plotly_chart(fig_ts_centro, use_container_width=True)

    except Exception as e:
        st.error(f"Error al generar el pronóstico: {str(e)}")

    # Grado Dependencia Time Series Forecast Section
    st.subheader("Pronóstico de Grado de Dependencia (5 años)")
    st.markdown("""
    Esta sección muestra el análisis de la evolución temporal del grado de dependencia y su proyección
    para los próximos 5 años, basado en datos históricos reales.
    """)

    # Load and process time series data for grado dependencia
    try:
        ts_data_grado = load_contratos_data_grado()
        
        # Generate forecast
        ts_grado, forecast_grado, ci_lower_grado, ci_upper_grado = create_time_series_forecast_grado(ts_data_grado)
        
        # Create Plotly figure for time series
        fig_ts_grado = go.Figure()
        
        # Add historical data
        fig_ts_grado.add_trace(go.Scatter(
            x=ts_grado.index,
            y=ts_grado.values,
            name='Datos Históricos',
            line=dict(color='#1976D2', width=1),
            mode='lines+markers',
            marker=dict(size=3)
        ))
        
        # Add forecast
        fig_ts_grado.add_trace(go.Scatter(
            x=forecast_grado.index,
            y=forecast_grado.values,
            name='Pronóstico',
            line=dict(color='#4CAF50', width=2, dash='dash'),
            mode='lines+markers',
            marker=dict(size=3)
        ))
        
        # Add confidence intervals
        fig_ts_grado.add_trace(go.Scatter(
            x=forecast_grado.index,
            y=ci_upper_grado,
            name='Intervalo Superior',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig_ts_grado.add_trace(go.Scatter(
            x=forecast_grado.index,
            y=ci_lower_grado,
            name='Intervalo de Confianza 95%',
            fill='tonexty',
            fillcolor='rgba(76, 175, 80, 0.2)',
            line=dict(width=0)
        ))
        
        # Update layout
        fig_ts_grado.update_layout(
            title='Pronóstico de Grado de Dependencia (5 años)',
            xaxis_title='Fecha',
            yaxis_title='Grado de Dependencia Promedio',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Add vertical line at May 2025
        fig_ts_grado.add_vline(
            x="2025-05-01",
            line_width=1,
            line_color="white",
            line_dash="solid",
            showlegend=False
        )

        # Display the plot
        st.plotly_chart(fig_ts_grado)
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Promedio Histórico",
            f"{ts_grado.mean():.1f}"
        )
        col2.metric(
            "Tendencia",
            "↗️ Creciente" if forecast_grado.values[-5] > forecast_grado.values[0] else "↘️ Decreciente"
        )
        col3.metric(
            "Pronóstico Final",
            f"{forecast_grado.values[-1]:.1f}"
        )        # Add forecast by centro section for grado dependencia
        st.markdown("### Pronóstico de Grado de Dependencia por Proyectos")
        st.markdown("""
        Esta sección muestra la evolución y pronóstico del grado de dependencia para cada proyecto.
        Las líneas sólidas representan datos históricos y las líneas punteadas son pronósticos.
        """)

        # Load and process time series data by centro for grado dependencia
        ts_data_by_centro_grado = load_contratos_data_by_centro_grado()
        forecasts_by_centro_grado = create_forecast_by_centro_grado(ts_data_by_centro_grado)
          # Add center filter
        all_centers_grado = list(forecasts_by_centro_grado.keys())
        
        selected_centers_grado = st.multiselect(
            "Seleccionar Proyectos (Grado de Dependencia)",
            options=all_centers_grado,
            default=all_centers_grado,
            help="Selecciona los proyectos que deseas visualizar en el gráfico de grado de dependencia",
            key="centers_grado"  # Unique key to avoid conflict with previous multiselect
        )
        
        # Create Plotly figure for multi-series forecast
        fig_ts_centro_grado = go.Figure()
        
        # Color palette for different centers
        colors = px.colors.qualitative.Set3
        
        # Add traces for each selected centro
        for i, centro in enumerate(selected_centers_grado):
            if centro in forecasts_by_centro_grado:
                data = forecasts_by_centro_grado[centro]
                color = colors[i % len(colors)]
                
                # Add historical data
                fig_ts_centro_grado.add_trace(go.Scatter(
                    x=data['historical'].index,
                    y=data['historical'].values,
                    name=centro,
                    line=dict(color=color, width=2),
                    mode='lines+markers',
                    marker=dict(size=3),
                    showlegend=True
                ))
                
                # Add forecast
                fig_ts_centro_grado.add_trace(go.Scatter(
                    x=data['forecast'].index,
                    y=data['forecast'].values,
                    name=centro,
                    line=dict(color=color, width=2, dash='dash'),
                    mode='lines+markers',
                    marker=dict(size=3),
                    showlegend=False
                ))
          # Update layout
        fig_ts_centro_grado.update_layout(
            title='Pronóstico de Grado de Dependencia por Proyectos (3 años)',
            xaxis_title='Fecha',
            yaxis_title='Grado de Dependencia Promedio',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            height=600
        )

        # Add vertical line at May 2025
        fig_ts_centro_grado.add_vline(
            x="2025-05-01",
            line_width=1,
            line_color="white",
            line_dash="solid",
            showlegend=False
        )
        
        # Display the plot
        st.plotly_chart(fig_ts_centro_grado, use_container_width=True)

    except Exception as e:
        st.error(f"Error al generar el pronóstico de grado de dependencia: {str(e)}")    # Percentage Change Comparison Section
    st.subheader("Cambio Porcentual 2020-2024 por Proyectos")    
    st.markdown("""
    Esta sección muestra el cambio porcentual en las horas PIA y el grado de dependencia 
    para cada proyecto entre 2020 y 2024, basado en promedios anuales.
    """)
    
    try:
        # 1) Obtener datos
        changes_data = calculate_yearly_changes()
        all_centers_changes = list(changes_data['Centro'].unique())
        default_centers= ['CAMAS ', 'DIPUTACION DE HUELVA', 'DIPUTACION_JAEN', 'LINARES', 'SAD DIPUTACIÓN CADIZ', 'UTE_MARTOS', 'ALCALA LA REAL']

        selected_centers_changes = st.multiselect(
            "Seleccionar Proyectos (Cambios Porcentuales)",
            options=all_centers_changes,
            default=default_centers,
            help="Selecciona los proyectos que deseas visualizar en el gráfico de cambios porcentuales",
            key="centers_changes"
        )
          # Filter out centers with NaN values and those selected by user
        filtered_changes = changes_data[
            changes_data['Centro'].isin(selected_centers_changes) & 
            changes_data['cambio_horas'].notna() & 
            changes_data['cambio_grado'].notna()
        ].copy()

        # Export filtered data to Excel
        try:
            output_dir = os.path.join(os.path.dirname(__file__), 'exports')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            
            with pd.ExcelWriter(os.path.join(output_dir, f'filtered_changes_export_{timestamp}.xlsx')) as writer:
                filtered_changes.to_excel(writer, sheet_name='filtered_changes', index=False)
                changes_data.to_excel(writer, sheet_name='changes_data', index=False)
                pd.DataFrame({'selected_centers': selected_centers_changes}).to_excel(
                    writer, sheet_name='selected_centers', index=False
                )
                pd.DataFrame({'all_centers': all_centers_changes}).to_excel(
                    writer, sheet_name='all_centers', index=False
                )
            
        except Exception as e:
            st.error(f"Error exporting filtered data: {str(e)}")

        # 3) Create visualization
        # Prepare period label
        if not filtered_changes.empty:
            periodo_label = filtered_changes["periodo"].iloc[0]
        else:
            periodo_label = "2020-2024"

        # Crear figura (todo con la indentación correcta)
        fig_changes = go.Figure()
        fig_changes.add_trace(go.Bar(
            name='Cambio en Horas PIA',
            x=filtered_changes['Centro'],
            y=filtered_changes['cambio_horas'],
            marker_color='#1976D2',
            text=filtered_changes['cambio_horas'].round(1).astype(str) + '%',
            textposition='auto',
        ))
        fig_changes.add_trace(go.Bar(
            name='Cambio en Grado de Dependencia',
            x=filtered_changes['Centro'],
            y=filtered_changes['cambio_grado'],
            marker_color='#4CAF50',
            text=filtered_changes['cambio_grado'].round(1).astype(str) + '%',            textposition='auto',
        ))
        
        fig_changes.update_layout(
            title=f'Cambio Porcentual por Proyectos ({periodo_label})',
            xaxis_title='Proyectos',
            yaxis_title='Cambio Porcentual (%)',
            barmode='group',
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            bargap=0.3,
            bargroupgap=0.1,
            xaxis=dict(tickangle=45)
        )

        # Mostrar el chart
        st.plotly_chart(fig_changes, use_container_width=True)

        # Métricas de promedio
        col1, col2 = st.columns(2)
        col1.metric(
            "Cambio Promedio en Horas PIA",
            f"{filtered_changes['cambio_horas'].mean():.1f}%"
        )
        col2.metric(
            "Cambio Promedio en Grado de Dependencia",
            f"{filtered_changes['cambio_grado'].mean():.1f}%"
        )

    except Exception as e:
        st.error(f"Error al generar el gráfico de cambios porcentuales: {str(e)}")

if __name__ == "__main__":
    main()
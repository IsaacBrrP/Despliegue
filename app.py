
#####################################################
# Importamos librerías
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

######################################################
# Definimos la instancia
@st.cache_resource
######################################################
# Creamos la función de carga de datos
def load_data():
    df = pd.read_csv("Madrid_AirBnb_010.csv")
    df = df.fillna(method="bfill")
    df = df.fillna(method="ffill")  
    Lista = [
        'host_location', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
        'host_is_superhost', 'neighbourhood_cleansed', 'property_type', 'room_type',
        'has_availability', 'availability_30', 'availability_60', 'availability_90',
        'availability_365', 'estimated_occupancy_l365d', 'host_listings_count',
        'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'price', 'maximum_nights_avg_ntm', 'number_of_reviews', 'number_of_reviews_ltm',
        'num_resenas_30d', 'number_of_reviews_ly', 'estimated_revenue_l365d',
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'reviews_per_month', 'response_time_bin', 'response_rate_bin',
        'acceptance_rate_bin', 'superhost_bin', 'neighbourhood_bin', 'accommodates_bin',
        'review_scores_rating_bin', 'accuracy_bin', 'cleanliness_bin', 'location_bin']
    return df, Lista

###############################################################################
# Cargo los datos obtenidos de la función "load_data"
df, Lista = load_data()

###############################################################################
# DICCIONARIO DE GLOSARIO PARA VARIABLES BINARIAS
glosario_binarias = {
    'tiempo_respuesta_anfitrion': {
        'variable': 'response_time_bin',
        'etiqueta_1': 'Rápida (within an hour, within a few hours)',
        'etiqueta_0': 'Lenta'
    },
    'tasa_respuesta_anfitrion': {
        'variable': 'response_rate_bin',
        'etiqueta_1': 'Alta (≥70)',
        'etiqueta_0': 'Media-Baja (<70)'
    },
    'tasa_aceptacion_anfitrion': {
        'variable': 'acceptance_rate_bin',
        'etiqueta_1': 'Alta (≥70)',
        'etiqueta_0': 'Media-Baja (<70)'
    },
    'es_superanfitrion': {
        'variable': 'superhost_bin',
        'etiqueta_1': 'Sí',
        'etiqueta_0': 'No'
    },
    'distrito': {
        'variable': 'neighbourhood_bin',
        'etiqueta_1': 'Centro',
        'etiqueta_0': 'Limítrofe'
    },
    'capacidad': {
        'variable': 'accommodates_bin',
        'etiqueta_1': '4 o más',
        'etiqueta_0': '3 o menos'
    },
    'review_scores_rating': {
        'variable': 'review_scores_rating_bin',
        'etiqueta_1': 'Buena (≥4)',
        'etiqueta_0': 'Regular-Mala (<4)'
    },
    'review_scores_accuracy': {
        'variable': 'accuracy_bin',
        'etiqueta_1': 'Buena (≥4)',
        'etiqueta_0': 'Regular-Mala (<4)'
    },
    'review_scores_cleanliness': {
        'variable': 'cleanliness_bin',
        'etiqueta_1': 'Buena (≥4)',
        'etiqueta_0': 'Regular-Mala (<4)'
    },
    'review_scores_location': {
        'variable': 'location_bin',
        'etiqueta_1': 'Buena (≥4)',
        'etiqueta_0': 'Regular-Mala (<4)'
    }
}

###############################################################################
# CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="Airbnb Madrid Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

###############################################################################
# ESTILOS CSS MEJORADOS
st.markdown("""
    <style>
    /* Importar fuente Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Estilos generales de la aplicación */
    .stApp {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Títulos principales */
    h1 {
        font-size: 42px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: left !important;
        margin-bottom: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Subtítulos */
    h2 {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #1F2937 !important;
        text-align: left !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
        font-family: 'Inter', sans-serif !important;
    }

    h3 {
        font-size: 18px !important;
        font-weight: 500 !important;
        color: #475569 !important;
        text-align: left !important;
        margin-bottom: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Estilo para el texto de descripción */
    .hero-sub {
        font-size: 18px;
        color: #64748B;
        margin-bottom: 30px;
        font-weight: 400;
    }

    /* Tarjetas para gráficos */
    .chart-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 24px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .chart-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #3B82F6;
        margin-bottom: 16px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #1E3A8A;
        margin: 8px 0;
    }

    .metric-label {
        font-size: 14px;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Separador visual */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, #3B82F6 0%, transparent 100%);
        margin: 30px 0;
    }

    /* Tabla de glosario estilizada */
    .glosario-table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 13px;
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }

    .glosario-table thead tr {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        text-align: left;
        font-weight: 600;
    }

    .glosario-table th {
        padding: 12px 15px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .glosario-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #E5E7EB;
    }

    .glosario-table tbody tr:hover {
        background-color: #F8FAFC;
    }

    .glosario-table tbody tr:last-child td {
        border-bottom: none;
    }

    .etiqueta-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }

    .badge-1 {
        background-color: #DBEAFE;
        color: #1E40AF;
    }

    .badge-0 {
        background-color: #FEF3C7;
        color: #92400E;
    }

    /* Logo en sidebar */
    .sidebar-logo {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px 0;
        margin-bottom: 20px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }

    .sidebar-logo img {
        max-width: 150px;
        height: auto;
        filter: brightness(0) invert(1);
    }

    /* Mejoras en los selectbox y widgets */
    .stSelectbox label {
        font-weight: 600 !important;
        color: #1F2937 !important;
        font-size: 14px !important;
    }

    .stMultiSelect label {
        font-weight: 600 !important;
        color: #1F2937 !important;
        font-size: 14px !important;
    }

    /* Botones mejorados */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #1E3A8A 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
    }

    /* Estilo de las métricas de Streamlit */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #1E3A8A;
    }

    [data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: #64748B;
    }

    /* Mejorar el aspecto de los dataframes */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }

    /* Espaciado entre columnas */
    .row-widget.stHorizontal {
        gap: 20px;
    }

    /* Estilo del expander */
    .streamlit-expanderHeader {
        background-color: #F8FAFC;
        border-radius: 8px;
        font-weight: 600;
        color: #1F2937;
    }
    </style>
""", unsafe_allow_html=True)

############################################################################################
# CUSTOMIZACIÓN DEL SIDEBAR CON GRADIENTE MODERNO
st.sidebar.markdown("""
    <style>
    /* Sidebar con gradiente elegante */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
    }

    /* Título del sidebar */
    [data-testid="stSidebar"] h1 {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        text-align: center !important;
        padding: 15px 0 !important;
        margin-bottom: 20px !important;
        background: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    /* Labels del sidebar */
    [data-testid="stSidebar"] label {
        color: #E5E7EB !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        margin-bottom: 8px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Selectbox en sidebar */
    [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 24px;
    }

    /* Multiselect en sidebar */
    [data-testid="stSidebar"] .stMultiSelect {
        margin-bottom: 24px;
    }

    /* Texto en sidebar */
    [data-testid="stSidebar"] p {
        color: #D1D5DB;
        font-size: 14px;
    }

    /* Línea separadora en sidebar */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 24px 0;
    }
    </style>
""", unsafe_allow_html=True)

################################################################################
# SIDEBAR CON LOGO DE AIRBNB
# Logo de Airbnb
st.sidebar.markdown(
    """
    <div class="sidebar-logo">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/2560px-Airbnb_Logo_B%C3%A9lo.svg.png" alt="Airbnb Logo">
    </div>
    """,
    unsafe_allow_html=True
)

# GENERAMOS LOS ENCABEZADOS PARA LA BARRA LATERAL
st.sidebar.title("MADRID ANALYTICS")
st.sidebar.markdown("---")

# Widget 1: Selectbox con icono
View = st.sidebar.selectbox(
    label="📊 Tipo de Análisis", 
    options=["Extracción de Características", "Regresión Lineal", 
             "Regresión No Lineal", "Regresión Logística"]
)

# GLOSARIO DE VARIABLES BINARIAS EN SIDEBAR
st.sidebar.markdown("---")
with st.sidebar.expander("📚 Glosario de Variables Binarias"):
    st.markdown("""
    <table class="glosario-table">
        <thead>
            <tr>
                <th>Variable</th>
                <th>Etiqueta 1 (🔵)</th>
                <th>Etiqueta 0 (🟡)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>response_time_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">Rápida</span><br><small>(within an hour, within a few hours)</small></td>
                <td><span class="etiqueta-badge badge-0">Lenta</span></td>
            </tr>
            <tr>
                <td><strong>response_rate_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">Alta (≥70)</span></td>
                <td><span class="etiqueta-badge badge-0">Media-Baja (&lt;70)</span></td>
            </tr>
            <tr>
                <td><strong>acceptance_rate_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">Alta (≥70)</span></td>
                <td><span class="etiqueta-badge badge-0">Media-Baja (&lt;70)</span></td>
            </tr>
            <tr>
                <td><strong>superhost_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">Sí</span></td>
                <td><span class="etiqueta-badge badge-0">No</span></td>
            </tr>
            <tr>
                <td><strong>neighbourhood_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">Centro</span></td>
                <td><span class="etiqueta-badge badge-0">Limítrofe</span></td>
            </tr>
            <tr>
                <td><strong>accommodates_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">4 o más</span></td>
                <td><span class="etiqueta-badge badge-0">3 o menos</span></td>
            </tr>
            <tr>
                <td><strong>review_scores_rating_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">Buena (≥4)</span></td>
                <td><span class="etiqueta-badge badge-0">Regular-Mala (&lt;4)</span></td>
            </tr>
            <tr>
                <td><strong>accuracy_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">Buena (≥4)</span></td>
                <td><span class="etiqueta-badge badge-0">Regular-Mala (&lt;4)</span></td>
            </tr>
            <tr>
                <td><strong>cleanliness_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">Buena (≥4)</span></td>
                <td><span class="etiqueta-badge badge-0">Regular-Mala (&lt;4)</span></td>
            </tr>
            <tr>
                <td><strong>location_bin</strong></td>
                <td><span class="etiqueta-badge badge-1">Buena (≥4)</span></td>
                <td><span class="etiqueta-badge badge-0">Regular-Mala (&lt;4)</span></td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

################################################################################
# CONTENIDO DE LA VISTA 1: EXTRACCIÓN DE CARACTERÍSTICAS
if View == "Extracción de Características":
    # Select box
    Variable_Cat = st.sidebar.selectbox(label="🔍 Variables", options=Lista)

    # Obtenemos las frecuencias
    Tabla_frecuencias = df[Variable_Cat].value_counts().reset_index()
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']

    # Header principal con subtítulo
    st.markdown("<h1>📈 Extracción de Características</h1>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Análisis visual interactivo de las variables de Airbnb Madrid</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Fila 1: Gráficos de barras y pastel
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Gráfico de Barras")
        figure1 = px.bar(
            data_frame=Tabla_frecuencias, 
            x='categorias', 
            y='frecuencia', 
            title='Distribución de Frecuencias',
            color='frecuencia',
            color_continuous_scale='Blues'
        )
        figure1.update_layout(
            height=400,
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#1F2937'),
            title_font_size=18,
            title_font_color='#1E3A8A',
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='#E5E7EB')
        )
        st.plotly_chart(figure1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("### 🥧 Gráfico Circular")
        figure2 = px.pie(
            data_frame=Tabla_frecuencias, 
            names='categorias', 
            values='frecuencia', 
            title='Proporción por Categoría',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        figure2.update_layout(
            height=400,
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#1F2937'),
            title_font_size=18,
            title_font_color='#1E3A8A',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(figure2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Fila 2: Gráfico de dona y área
    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("### 🍩 Gráfico de Anillo")
        figure3 = px.pie(
            data_frame=Tabla_frecuencias, 
            names='categorias', 
            values='frecuencia', 
            hole=0.45, 
            title='Vista de Anillo',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        figure3.update_layout(
            height=400,
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#1F2937'),
            title_font_size=18,
            title_font_color='#1E3A8A',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(figure3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("### 📉 Gráfico de Área")
        figure4 = px.area(
            data_frame=Tabla_frecuencias, 
            x='categorias', 
            y='frecuencia', 
            title='Tendencia de Frecuencias',
            color_discrete_sequence=['#3B82F6']
        )
        figure4.update_layout(
            height=400,
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#1F2937'),
            title_font_size=18,
            title_font_color='#1E3A8A',
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor='#E5E7EB')
        )
        st.plotly_chart(figure4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

################################################################################
# CONTENIDO DE LA VISTA 2: REGRESIÓN LINEAL
if View == "Regresión Lineal":
    # Generamos la lista de variables numéricas
    numeric_df = df.select_dtypes(['float','int'])
    Lista_num = numeric_df.columns

    # Select box
    Variable_y = st.sidebar.selectbox(label="🎯 Variable Objetivo (Y)", options=Lista_num)
    Variable_x = st.sidebar.selectbox(label="📍 Variable Independiente Simple (X)", options=Lista_num)

    # Header principal
    st.markdown("<h1>📐 Regresión Lineal</h1>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Análisis de correlación y modelos de predicción lineales</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Fila 1
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Correlación Lineal Simple")

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X=df[[Variable_x]], y=df[Variable_y])
        y_pred = model.predict(X=df[[Variable_x]])
        coef_Deter_simple = model.score(X=df[[Variable_x]], y=df[Variable_y])
        coef_Correl_simple = np.sqrt(coef_Deter_simple)

        # Mostrar métrica en tarjeta
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Coeficiente de Correlación</div>
                <div class='metric-value'>{coef_Correl_simple:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

        # Gráfico
        figure5 = px.scatter(
            data_frame=numeric_df, 
            x=Variable_x, 
            y=Variable_y, 
            title=f'Modelo Lineal: {Variable_y} vs {Variable_x}',
            opacity=0.6
        )
        order = np.argsort(df[Variable_x].values)
        x_sorted = df[Variable_x].values[order]
        y_line = y_pred[order]
        figure5.add_trace(
            go.Scatter(
                x=x_sorted, y=y_line, mode='lines',
                name='Línea de Ajuste', 
                line=dict(width=3, color='#F59E0B')
            )
        )
        figure5.update_layout(
            height=450,
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#1F2937'),
            title_font_size=16,
            title_font_color='#1E3A8A',
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            xaxis=dict(gridcolor='#E5E7EB'),
            yaxis=dict(gridcolor='#E5E7EB')
        )
        st.plotly_chart(figure5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Correlación Lineal Múltiple")

        Variables_x = st.sidebar.multiselect(
            label="📊 Variables Independientes Múltiples (X)", 
            options=Lista_num
        )

        if len(Variables_x) > 0:
            model_M = LinearRegression()
            model_M.fit(X=df[Variables_x], y=df[Variable_y])
            y_pred_M = model_M.predict(X=df[Variables_x])
            coef_Deter_multiple = model_M.score(X=df[Variables_x], y=df[Variable_y])
            coef_Correl_multiple = np.sqrt(coef_Deter_multiple)

            # Métrica
            st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Coef. Correlación Múltiple</div>
                    <div class='metric-value'>{coef_Correl_multiple:.4f}</div>
                </div>
            """, unsafe_allow_html=True)

            # Gráfico
            df_long = df[Variables_x + [Variable_y]].melt(
                id_vars=[Variable_y], 
                var_name='variable', 
                value_name='value'
            )
            figure6 = px.scatter(
                df_long, x='value', y=Variable_y, color='variable',
                title='Modelo Múltiple - Todas las Variables', 
                opacity=0.5
            )

            for var in Variables_x:
                x = df[var].values
                y = df[Variable_y].values
                if np.isfinite(x).sum() > 1 and np.isfinite(y).sum() > 1:
                    m, b = np.polyfit(x, y, 1)
                    xs = np.linspace(np.nanmin(x), np.nanmax(x), 120)
                    ys = m*xs + b
                    figure6.add_trace(
                        go.Scatter(
                            x=xs, y=ys, mode='lines', 
                            name=f"Ajuste {var}", 
                            line=dict(width=2)
                        )
                    )

            figure6.update_layout(
                height=450,
                template='plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', size=11, color='#1F2937'),
                title_font_size=16,
                title_font_color='#1E3A8A',
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                xaxis=dict(gridcolor='#E5E7EB'),
                yaxis=dict(gridcolor='#E5E7EB')
            )
            st.plotly_chart(figure6, use_container_width=True)
        else:
            st.info("👆 Selecciona al menos una variable independiente en el menú lateral")

        st.markdown("</div>", unsafe_allow_html=True)

################################################################################
# CONTENIDO DE LA VISTA 3: REGRESIÓN NO LINEAL
if View == "Regresión No Lineal":
    from scipy.optimize import curve_fit
    from sklearn.metrics import r2_score

    numeric_df = df.select_dtypes(['float','int'])
    Lista_num = numeric_df.columns

    df_no = df[df['superhost_bin'] == 0].copy()
    df_yes = df[df['superhost_bin'] == 1].copy()

    Variable_y = st.sidebar.selectbox(
        label="🎯 Variable Objetivo (Y)", 
        options=Lista_num, 
        index=max(0, list(Lista_num).index('host_response_rate') if 'host_response_rate' in Lista_num else 0)
    )
    Variable_x = st.sidebar.selectbox(
        label="📍 Variable Independiente (X)", 
        options=Lista_num, 
        index=max(0, list(Lista_num).index('price') if 'price' in Lista_num else 0)
    )
    Lista_mod = ["Función cuadrática", "Función exponencial"]
    Modelo = st.sidebar.selectbox(label="⚙️ Tipo de Modelo", options=Lista_mod)

    st.markdown("<h1>🔄 Regresión No Lineal</h1>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Modelos de regresión cuadrática y exponencial por tipo de Superhost</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    def func_quad(x, a, b, c):
        return a*x**2 + b*x + c

    def func_exp(x, a, b, c):
        return a*np.exp(-b*x) + c

    def fit_and_plot(df_sub, xcol, ycol, titulo, badge_color):
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown(f"### {titulo}")

        fig = px.scatter(df_sub, x=xcol, y=ycol, title=titulo, opacity=0.5)
        x = df_sub[xcol].astype(float).to_numpy()
        y = df_sub[ycol].astype(float).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        r2 = np.nan
        try:
            if Modelo == "Función cuadrática":
                popt, _ = curve_fit(func_quad, x, y, maxfev=20000)
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                ys = func_quad(xs, *popt)
                y_pred = func_quad(x, *popt)
                name = "Ajuste Cuadrático"
            else:
                a0 = (np.nanmax(y) - np.nanmin(y)) if np.isfinite(y).any() else 1.0
                b0 = 1e-3
                c0 = np.nanmin(y) if np.isfinite(y).any() else 0.0
                popt, _ = curve_fit(func_exp, x, y, p0=[a0, b0, c0], maxfev=20000)
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                ys = func_exp(xs, *popt)
                y_pred = func_exp(x, *popt)
                name = "Ajuste Exponencial"

            r2 = r2_score(y, y_pred)
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode='lines', 
                    name=f"{name} (R²={r2:.3f})", 
                    line=dict(width=3, color=badge_color)
                )
            )
        except Exception:
            pass

        fig.update_layout(
            height=430,
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#1F2937'),
            title_font_size=16,
            title_font_color='#1E3A8A',
            margin=dict(l=20, r=20, t=60, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            xaxis=dict(gridcolor='#E5E7EB'),
            yaxis=dict(gridcolor='#E5E7EB')
        )
        st.plotly_chart(fig, use_container_width=True)

        if np.isfinite(r2):
            st.markdown(f"""
                <div class='metric-card' style='border-left-color: {badge_color};'>
                    <div class='metric-label'>Coeficiente de Correlación</div>
                    <div class='metric-value'>{np.sqrt(max(r2,0)):.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⚠️ No fue posible ajustar el modelo con los datos seleccionados.")

        st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        fit_and_plot(df_no, Variable_x, Variable_y, "🏘️ Superhost: No (0)", "#EF4444")

    with col2:
        fit_and_plot(df_yes, Variable_x, Variable_y, "⭐ Superhost: Sí (1)", "#10B981")

################################################################################
# CONTENIDO DE LA VISTA 4: REGRESIÓN LOGÍSTICA
if View == "Regresión Logística":
    numeric_df = df.select_dtypes(['float','int'])
    Lista_num = numeric_df.columns

    Lista_dicot = [
        'response_time_bin','response_rate_bin','acceptance_rate_bin','superhost_bin',
        'neighbourhood_bin','accommodates_bin','review_scores_rating_bin',
        'accuracy_bin','cleanliness_bin','location_bin'
    ]

    default_idx = Lista_dicot.index('superhost_bin') if 'superhost_bin' in Lista_dicot else 0
    Variable_y = st.sidebar.selectbox("🎯 Variable Dependiente (Y)", options=Lista_dicot, index=default_idx)
    Variables_x = st.sidebar.multiselect(
        "📊 Variables Independientes (X)",
        options=Lista_num,
        default=['estimated_occupancy_l365d'] if 'estimated_occupancy_l365d' in Lista_num else []
    )

    st.markdown("<h1>🎲 Regresión Logística</h1>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Clasificación binaria y evaluación del modelo predictivo</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Matriz de Confusión")

        X = df[Variables_x]
        y = df[Variable_y]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

        escalar = StandardScaler()
        X_train = escalar.fit_transform(X_train)
        X_test = escalar.transform(X_test)

        from sklearn.linear_model import LogisticRegression
        algoritmo = LogisticRegression(max_iter=1000)
        algoritmo.fit(X_train, y_train)
        y_pred = algoritmo.predict(X_test)

        matriz = confusion_matrix(y_test, y_pred)
        clases = np.unique(df[Variable_y])
        labels = [clases[0], clases[1]]

        figure9 = go.Figure(data=go.Heatmap(
            z=matriz,
            x=labels,
            y=labels,
            hoverinfo="z",
            colorscale="Blues",
            showscale=True,
            zmin=0
        ))

        annotations = []
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                valor = matriz[i, j]
                if i == 0 and j == 0:
                    texto = f'TP: {valor}'
                    emoji = '✅'
                elif i == 0 and j == 1:
                    texto = f'FP: {valor}'
                    emoji = '⚠️'
                elif i == 1 and j == 0:
                    texto = f'FN: {valor}'
                    emoji = '❌'
                elif i == 1 and j == 1:
                    texto = f'TN: {valor}'
                    emoji = '✔️'

                annotations.append(
                    dict(
                        x=labels[j],
                        y=labels[i],
                        text=f"{emoji}<br>{texto}",
                        showarrow=False,
                        font=dict(
                            color="white" if valor > matriz.max()/2 else "#1F2937",
                            size=14,
                            family='Inter'
                        )
                    )
                )

        figure9.update_layout(
            title='Matriz de Confusión del Modelo',
            xaxis_title="Predicción",
            yaxis_title="Valor Real",
            annotations=annotations,
            width=550,
            height=500,
            template='plotly_white',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', size=12, color='#1F2937'),
            title_font_size=16,
            title_font_color='#1E3A8A'
        )

        st.plotly_chart(figure9, use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Métricas del Modelo")

        from sklearn.metrics import accuracy_score, precision_score
        exactitud = accuracy_score(y_test, y_pred)
        precision_0 = precision_score(y_test, y_pred, average="binary", pos_label=clases[0])
        precision_1 = precision_score(y_test, y_pred, average="binary", pos_label=clases[1])

        # Métricas en tarjetas estilizadas
        st.markdown(f"""
            <div class='metric-card' style='border-left-color: #3B82F6;'>
                <div class='metric-label'>🎯 Exactitud del Modelo</div>
                <div class='metric-value'>{exactitud:.3f}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='metric-card' style='border-left-color: #10B981;'>
                <div class='metric-label'>📌 Precisión Etiqueta {clases[0]}</div>
                <div class='metric-value'>{precision_0:.3f}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class='metric-card' style='border-left-color: #F59E0B;'>
                <div class='metric-label'>📌 Precisión Etiqueta {clases[1]}</div>
                <div class='metric-value'>{precision_1:.3f}</div>
            </div>
        """, unsafe_allow_html=True)

        # Información adicional
        st.markdown("---")
        st.markdown("#### 💡 Interpretación")
        st.markdown(f"""
        <div style='font-size: 13px; color: #64748B; line-height: 1.6;'>
        <p><strong>Exactitud:</strong> {exactitud*100:.1f}% de las predicciones son correctas</p>
        <p><strong>TP (True Positive):</strong> Predicciones correctas positivas ✅</p>
        <p><strong>TN (True Negative):</strong> Predicciones correctas negativas ✔️</p>
        <p><strong>FP (False Positive):</strong> Falsos positivos ⚠️</p>
        <p><strong>FN (False Negative):</strong> Falsos negativos ❌</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #64748B; padding: 20px; font-size: 14px;'>
        <p>📊 Dashboard desarrollado con Streamlit | Datos de Airbnb Madrid</p>
        <p style='font-size: 12px;'>Análisis de datos de propiedades en la ciudad de Madrid</p>
    </div>
""", unsafe_allow_html=True)




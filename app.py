#Creamos el archivo de la APP en el interprete principal (Phyton)
#####################################################
#Importamos librerias
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
#Definimos la instancia
@st.cache_resource
######################################################
#Creamos la función de carga de datos
def load_data():
   #Lectura del archivo csv
   df=pd.read_csv("titanic_despliegue.csv")
    #Rellenamos nulos
   df =df.fillna(method="bfill")
   df =df.fillna(method="ffill")  
   Lista=['Survived', 'Pclass', 'Sex', 'Age']
   return df, Lista
###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df, Lista = load_data()
###############################################################################
#CREACIÓN DEL DASHBOARD
#Generamos las páginas que utilizaremos en el diseño
##############################################################################
#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("TITANIC")
#Widget 1: Selectbox
#Menu desplegable de opciones de laa páginas seleccionadas
View= st.sidebar.selectbox(label= "Tipo de Análisis", options= ["Extracción de Características", 
                                              "Regresión Lineal", "Regresión No Lineal",
                                              "Regresión Logística"])

# CONTENIDO DE LA VISTA 1
if View == "Extracción de Características":
    #EXTRACCIÓN DE CARACTERÍSTICAS
    #Select box
    Variable_Cat= st.sidebar.selectbox(label= "Variables", options= Lista)
    #Obtenemos las frecuencias de las categorías de la variable seleccionada
    Tabla_frecuencias = df[Variable_Cat].value_counts().reset_index()
    #Ajustamos los nombre de las cabeceras de las columnas
    Tabla_frecuencias.columns = ['categorias', 'frecuencia']

    #Generamos los encabezados para el dashboard
    st.title("Extracción de Características")

    #Generamos el diseño del Layout deseado
    # Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A: 
        st.write("Grafico de Barras")
        #GRAPH 1: BARPLOT
        #Despliegue de un bar plot, definiendo las variables "X categorias" y "Y numéricas" 
        figure1 = px.bar(data_frame=Tabla_frecuencias, x='categorias', 
                  y= 'frecuencia', title= str('Frecuencia por categoría'))
        figure1.update_xaxes(automargin=True)
        figure1.update_yaxes(automargin=True)
        figure1.update_layout(height=300)
        st.plotly_chart(figure1, use_container_width=True)

    with Contenedor_B:
        st.write("Grafico de Pastel")
        #GRAPH 2: PIEPLOT
        #Despliegue de un pie plot, definiendo las variables "X categorias" y "Y numéricas" 
        figure2 = px.pie(data_frame=Tabla_frecuencias, names='categorias', 
                  values= 'frecuencia', title= str('Frecuencia por categoría'))
        figure2.update_layout(height=300)
        st.plotly_chart(figure2, use_container_width=True)

    # Fila 2
    Contenedor_C, Contenedor_D = st.columns(2)
    with Contenedor_C:
        st.write("Grafico de anillo o dona")
        #GRAPH 3: DONUT PLOT
        #Despliegue de un line plot, definiendo las variables "X categorias" y "Y numéricas" 
        figure3 = px.pie(data_frame=Tabla_frecuencias, names='categorias', 
                  values= 'frecuencia', hole=0.4, title= str('Frecuencia por categoría'))
        figure3.update_layout(height=300)
        st.plotly_chart(figure3, use_container_width=True)

    with Contenedor_D:  
        st.write("Grafico de area")
        #GRAPH 4: AREA PLOT
        #Despliegue de un area plot, definiendo las variables "X categorias" y "Y numéricas" 
        figure4 = px.area(data_frame=Tabla_frecuencias, x='categorias', 
                  y= 'frecuencia', title= str('Frecuencia por categoría'))
        figure4.update_layout(height=300)
        st.plotly_chart(figure4, use_container_width=True)
 ############################################################################

 ###################################################################################

 # CONTENIDO DE LA VISTA 2
if View == "Regresión Lineal":
    #REGRESIÓN LINEAL SIMPLE 
    #Generamos la lista de variables numéricas
    numeric_df = df.select_dtypes(['float','int'])  #Devuelve Columnas
    Lista_num= numeric_df.columns                   #Devuelve lista de Columnas numéricas
    #Select box
    Variable_y= st.sidebar.selectbox(label= "Variable objetivo (Y)", options= Lista_num)
    Variable_x= st.sidebar.selectbox(label= "Variable independiente del modelo simple (X)", options= Lista_num)    


    #Generamos los encabezados para el dashboard
    st.title("Regresión Lineal")  

    #Generamos el diseño del Layout deseado
    # Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A: 
        st.write("Correlación Lineal Simple")

        #Se define model como la función de regresión lineal
        from sklearn.linear_model import LinearRegression
        model= LinearRegression()
        #Ajustamos el modelo con las variables antes declaradas
        model.fit(X=df[[Variable_x]], y=df[Variable_y])
        #Predecimos los valores de la variable objetivo
        y_pred= model.predict(X=df[[Variable_x]])
        #Corroboramos cual es el coeficiente de Determinación de nuestro modelo
        coef_Deter_simple=model.score(X=df[[Variable_x]], y=df[Variable_y])
        #Corroboramos cual es el coeficiente de Correlación de nuestro modelo
        coef_Correl_simple=np.sqrt(coef_Deter_simple)
        #Mostramos el dataset
        st.write(coef_Correl_simple)

        #GRAPH 5: SCATTERPLOT
        figure5 = px.scatter(data_frame=numeric_df, x=Variable_x, y=Variable_y, 
                     title= 'Modelo Lineal Simple')
        st.plotly_chart(figure5)

    with Contenedor_B:
        st.write("Correlación Lineal Múltiple")
        #Widget 3: Multiselect box
        #Generamos un cuadro de multi-selección (X) para seleccionar variables independientes
        Variables_x= st.sidebar.multiselect(label="Variables independientes del modelo múltiple (X)", options= Lista_num)
        #Se define model como la función de regresión lineal
        from sklearn.linear_model import LinearRegression
        model_M= LinearRegression()
        #Ajustamos el modelo con las variables antes declaradas
        model_M.fit(X=df[Variables_x], y=df[Variable_y])
        #Predecimos los valores de la variable objetivo
        y_pred_M= model_M.predict(X=df[Variables_x])
        #Corroboramos cual es el coeficiente de Determinación de nuestro modelo
        coef_Deter_multiple=model_M.score(X=df[Variables_x], y=df[Variable_y])
        #Corroboramos cual es el coeficiente de Correlación de nuestro modelo
        coef_Correl_multiple=np.sqrt(coef_Deter_multiple)
        #Mostramos el coeficiente de correlación múltiple
        st.write(coef_Correl_multiple)

        #GRAPH 6: SCATTERPLOT
        figure6 = px.scatter(data_frame=numeric_df, x=Variables_x, y=Variable_y,
                     title= 'Modelo Lineal Múltiple')
        st.plotly_chart(figure6)

 ####################################################################################



"""Script con la aplicación principal"""

# Librerías internas de python
from datetime import datetime
from io import BytesIO
import pytz
import time
from typing import Tuple
# Librerías de terceros
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
# Librerías propias del proyecto
from models.convnet_model import load_model
from streamlit_func import show_sidebar, config_page
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import cv2


# Constantes #
DAY_HOUR_FORMAT = """%d/%m/%y\n%H:%M"""
DAY_FORMAT = "%d/%m/%y"
COLOR_BLUE = '#213f99'
CATEGORIES = ['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic', 'trash']
THRESHOLD = 0.97
TARGET_SIZE = (254, 254)  # Tamaño objetivo

def display_image_with_label(img_array: np.ndarray, pred_index: int, conf: float):
    
    THRESHOLD = 0.97
    
    if img_array is not None and img_array.size > 0:
        # Convertimos el array a una imagen PIL
        img = Image.fromarray(img_array.astype(np.uint8))  # Asegúrate de que el array sea uint8

        # Dibujamos el texto en la imagen
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()  # Cargar una fuente por defecto
        pred_category = CATEGORIES[pred_index]
        
        if conf < THRESHOLD:
            label_text = f"No se puede clasificar la imagen ({conf:.2%})"
        else:
            pred_category = CATEGORIES[pred_index]
            label_text = f"{pred_category} ({conf:.2%})"
        
        text_position = (10, 10)  # Posición del texto en la imagen
        text_color = "white"
        background_color = "black"  # Color de fondo para el texto
        
        # Calcular el tamaño del rectángulo de fondo en función del texto
        #text_size = draw.textsize(label_text, font=font)
        text_bbox = draw.textbbox(text_position, label_text, font=font)  # Obtiene la caja de texto
        background_position = (
            text_bbox[0] - 5,
            text_bbox[1] - 5,
            text_bbox[2] + 5,
            text_bbox[3] + 5
        )
        
        # Dibujar el fondo antes de escribir el texto
        draw.rectangle(background_position, fill=background_color)
        
        draw.text(text_position, label_text, fill=text_color, font=font)
    
        # Mostrar la imagen en Streamlit
        st.image(img, caption="Predicción del modelo con categoría", use_column_width=True)
    else:
        st.error("Cargue una imagen.")
    
# Funciones auxiliares #
def process_image(img_array:np.ndarray) -> np.ndarray:
    """Preprocesa la imagen cargada para adecuarla a los requerimientos
    del modelo

    Parameters
    ----------
    img_array : np.ndarray
        Imagen en formato array sin procesar

    Returns
    -------
    np.ndarray
        Imagen reescalada con el formato adecuado para alimentar el modelo
    """
    #img_process = img_array.reshape(1,28,28,1).astype(np.float32) / 255
    img_process = img_array.reshape(1, 224, 224, 3).astype(np.float32) / 255.0
    return img_process

@st.cache_data()
def predict(img_processed:np.ndarray) -> Tuple[int, float]:
   
    """Lanza el modelo sobre la imagen procesada
    y devuelve una tupla con la predicción del modelo
    y la confianza

    Parameters
    ----------
    img_processed : np.ndarray
        Imagen en formato array de numpy con el formato
        adecuado

    Returns
    -------
    Tuple[int, float]
        Imagen predicho y confianza
    """
    # Cargamos el modelo
    model = load_model(from_weights=False)
    # Obtener las probabilidades de cada clase (7 clases en total)
    probs: np.ndarray = model.predict(img_processed)
    # Obtener la clase predicha (índice de la clase con la probabilidad más alta)
    pred = int(np.argmax(probs))
    # Sacamos la predicción de la imagen
    #pred = np.argmax(probs)
    # Obtener la confianza de la predicción
    conf = float(probs[0, pred])  # Proporción de confianza en la clase predicha
    
    # Sacamos la confianza de dicha predicción
    #conf = probs.max()
    return pred, conf

def reset_predictions() -> None:
    """Resetea algunas variables de sesión cuando se cambia de imagen cargada
    o se elimina la imagen cargada"""
    if 'ultima_prediccion' in st.session_state:
        del st.session_state['ultima_prediccion']

def get_timestamp(formato:str) -> str:
    """Recibe un formato de timestamp en string y devuelve
    en ese formato el momento del día actual

    Parameters
    ----------
    formato : str
        Formato de tipo datetime para mostrar
        Por ejemplo: %d/%m/%y

    Returns
    -------
    str
        Devuelve momento actual del día
        en el formato especificado
    """
    return datetime.strftime(datetime.now(tz=pytz.timezone('America/Argentina/Buenos_Aires')),format=formato)

# Validaciones #
def pred_already_saved(filename:str) -> bool:
    """Comprueba si el nombre de archivo está guardado ya en el historial.
    Devuelve True si ya ha sido guardado y False en caso contrario

    Parameters
    ----------
    filename : str
        _description_

    Returns
    -------
    bool
        True si el nombre del archivo está en historial
        False en caso contrario
    """
    return any(filename in pred.values() for pred in st.session_state['historial'])

def is_valid_image(img_array:np.ndarray) -> Tuple[bool, str]:
    """Realiza validaciones al array de la imagen.
    Devuelve una tupla con un bool y un mensaje de error

    Parameters
    ----------
    img_array : np.ndarray
        La imagen en formato array de numpy

    Returns
    -------
    Tuple[bool, str]
        True, "" si la imagen es válida
        False, "mensaje de error" si la imagen no es válida
    """
    # Verificar si la imagen es un array numpy
    if not isinstance(img_array, np.ndarray):
        return False ,"La imagen es un array numpy"
    
    # Verificar dimensiones
    if img_array.shape != (224, 224, 3):
        return False, "No cumple con las dimensiones (224,224,3)"

    # Verificar tipo de dato y rango de valores
    if img_array.dtype != np.uint8 or img_array.min() < 0 or img_array.max() > 255:
        return False, "Tipo de dato y rango de valores"
    
    if img_array is not None and img_array.size > 0:
        return True, ""
    else:
        return False, "Error con el formato de imagen"
    
   

# Función principal #
def main() -> None:
    """Entry point de la app"""

    # Configuración de la app
    config_page()
    # Mostramos la Sidebar que hemos configurado en streamlit_func
    show_sidebar()
    # Título y descripción de la app
    st.write(''' Grupo 6 -  Botta Lucarella Magistocchi Cornelio''')
    st.title('Clasificación de Residuos')
    st.subheader('Una App para mostrar y probar un Modelo de Aprendizaje')
    st.write('''Con esta app podras evaluar un modelo convolucional entrenado 
                con el dataset [waste-classifier](https://huggingface.co/datasets/rootstrap-org/waste-classifier).<br>
                Sube una imagen con residuo a clasificar.<br>
                Pulsa sobre la solapa "**predecir**" y comprueba si el modelo ha sido
                capaz de clasificar correctamente la imagen.
                Si el accuracy supera el 97% muestra la clasifición.''', unsafe_allow_html=True)
    
    # Inicializamos variables de sesión para llevar un registro de las predicciones
    if st.session_state.get('historial') is None:
        st.session_state['historial'] = []
    # Flag para saber si tenemos una imagen cargada y validada
    st.session_state['imagen_cargada_y_validada'] = False
    # Definimos las 5 tabs que tendrá nuestra app
    tab_cargar_imagen, tab_ver_imagen, tab_predecir, \
        tab_evaluar, tab_estadisticas = st.tabs(['Cargar imagen', 'Ver imagen', 
                                        'Predecir', 'Evaluar', 'Ver estadísticas'])
    ################
    ## TAB Cargar ##
    ################
    
    with tab_cargar_imagen:
        st.write('''Carga tu imagen con la basura a clasificar.''', unsafe_allow_html=True)

        imagen_bruta = st.file_uploader('Sube tu imagen', type=["png", "tif", "jpg", "bmp", "jpeg"], on_change=reset_predictions)

        if imagen_bruta is not None:
            # Leer y convertir la imagen
            img = Image.open(BytesIO(imagen_bruta.read()))

            # Convertir la imagen a RGB si no tiene 3 canales
            if img.mode != 'RGB':
                img = img.convert('RGB')
        
            # Redimensionar la imagen a (224, 224)
            img = img.resize((224, 224))

            # Convertir la imagen a array numpy
            img_array = np.array(img)

            # Validar si la imagen tiene el formato esperado
            if img_array.shape != (224, 224, 3):
                st.error("No se pudo redimensionar la imagen a (224, 224, 3)")
            else:
                st.session_state['imagen_cargada_y_validada'] = imagen_bruta.name
                st.success("Imagen cargada y redimensionada correctamente a (224, 224, 3)")


    ################
    ## TAB Imagen ##
    ################        
    with tab_ver_imagen:
        # Verificamos que tengamos una imagen cargada y validada en sesión
        if nombre_archivo:=st.session_state.get('imagen_cargada_y_validada'):
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.imshow(img_array, cmap="gray")
            ax.axis('off')
            ax.set_title(nombre_archivo, fontsize=5)
            st.pyplot(fig)
        else:
            st.info('Carga una imagen para visualizar.')

    ##################
    ## TAB Predecir ##
    ##################
    with tab_predecir:
        # Comprobamos si se ha lanzado una predicción y por tanto está almacenada en sesión.
        # Si tenemos una predicción la mostramos
        if (last_pred:=st.session_state.get('ultima_prediccion')) is not None:
            pred = last_pred['pred']
            pred_index = pred
            conf = last_pred['conf']
            #st.metric('Predicción del modelo', value=pred, delta=f"{'-' if conf < 0.7 else ''}{conf:.2%}")
            try:
                if img_array is not None and img_array.size > 0:
                    display_image_with_label(img_array, pred_index, conf)
            except UnboundLocalError:
                pass
                        
        else:
            # Si no se ha lanzado una predicción, mostramos mecanismo para lanzarla
            # Verificamos que tengamos una imagen cargada y validada en sesión
            if nombre_imagen:=st.session_state.get('imagen_cargada_y_validada'):
                predecir = st.button(f'Predecir "{nombre_imagen}"')
                if predecir:
                    # Procesamos la imagen
                    img_processed = process_image(img_array)
                    # Lanzamos las predicciones
                    with st.spinner(text='Prediciendo imagen...'):
                        try:
                            pred, conf = predict(img_processed)
                            time.sleep(1)
                        except Exception as exc:
                            st.error(f'Se ha producido un error al predecir: {exc}')
                            st.stop()
                    # Si la confianza es menor del 70% ponemos un signo menos para que streamlit lo muestre
                    # en color rojo
                    #st.metric('Predicción del modelo', value=pred, delta=f"{'-' if conf < 0.7 else ''}{conf:.2%}")
                    # Suponiendo que `pred` es el índice de la categoría predicha y `conf` es la confianza de la predicción
                    pred_index = pred  # índice de la predicción
                    ##pred_category = CATEGORIES[pred_index]  # Obtener el nombre de la categoría
                    ##confidence_display = f"{'-' if conf < 0.7 else ''}{conf:.2%}"

                    # Mostrar el nombre de la categoría y la confianza con Streamlit
                    ##st.metric('Predicción del modelo', value=pred_category, delta=confidence_display)
                    
                    try:
                        if img_array is not None and img_array.size > 0:
                            display_image_with_label(img_array, pred_index, conf)
                    except UnboundLocalError:
                        pass        
                    
                    # Guardamos en sesión
                    st.session_state['ultima_prediccion'] = {
                        'pred': int(pred),
                        'conf': conf,
                        'archivo': nombre_imagen,
                    }
                    
            else:
                st.info('Carga una imagen para predecir.')

    #################
    ## TAB Evaluar ##
    #################
    with tab_evaluar:
        # Verificamos si hay una predicción lanzada y guardada en sesión
        if (last_pred:=st.session_state.get('ultima_prediccion')) is not None:
            # Posibilidad de contrastar con la realidad para almacenar porcentaje de aciertos
            st.subheader('¿ Ha acertado el modelo ?')
            # Selección de categoría de residuos
            category = st.selectbox('Selecciona la categoría de residuos que deseas clasificar', CATEGORIES)
            #digit = st.number_input('Marca el dígito que habías dibujado', min_value=0, max_value=9)
            guardar_pred = st.button('Guardar evaluación', help='Añade la evaluación al historial')
            # Si se pulsa el botón
            if guardar_pred:
                # Comprobamos que no hayamos guardado ya en sesión para no falsear las estadísticas
                if not pred_already_saved(last_pred['archivo']):
                    # Añadimos a ultima_prediccion la evaluación del usuario
                    # last_pred['real'] = digit
                    last_pred['real'] = CATEGORIES.index(category)
                    # Añadimos la hora
                    last_pred['fecha'] = get_timestamp(DAY_HOUR_FORMAT)
                    # Añadimos los valores al historial
                    st.session_state['historial'].append(last_pred)
                    # Mostramos mensaje de éxito
                    st.success('Evaluación guardada correctamente.')
                else:
                    # Mostramos advertencia
                    st.info('La evaluación ya se ha guardado.')
        else:
            st.info('Lanza una predicción para evaluar.')

    ######################
    ## TAB Estadísticas ##
    ######################
    with tab_estadisticas:
        # Comprobamos que haya historial guardado en sesión
        if st.session_state.get('historial'):
            
            # Creación del DataFrame con el dataset de ejemplo
            df = pd.DataFrame(st.session_state.get('historial'))
            df['pred'] = df['pred'].apply(lambda x: CATEGORIES[x])
            df['real'] = df['real'].apply(lambda x: CATEGORIES[x])
            # Sacamos los aciertos comparando la variable pred y real
            df['acierto'] = df['pred'] == df['real']
            
            # Convertir columna 'fecha' a tipo datetime
            df['fecha'] = pd.to_datetime(df['fecha'], format="%d/%m/%y %H:%M")
            
            # Calcular estadísticas agrupadas
            estadisticas = df.groupby('pred').agg(
            porcentaje_aciertos=('acierto', 'mean'),
            numero_predicciones=('pred', 'count'),
            confianza_media=('conf', 'mean')
            )

            # Convertir porcentajes a % para mejor legibilidad
            estadisticas['porcentaje_aciertos'] *= 100
            estadisticas['confianza_media'] *= 100

            # Mostrar estadísticas en Streamlit
            st.write("### Estadísticas por Categoría")
            st.dataframe(estadisticas.style.format({
                'porcentaje_aciertos': '{:.2f}%',
                'confianza_media': '{:.2f}%'
            }))

            # Gráfico de porcentaje de aciertos y número de predicciones
            fig, ax1 = plt.subplots(figsize=(10, 6))
            bar_width = 0.35
            indices = np.arange(len(estadisticas))

            ax1.bar(indices - bar_width/2, estadisticas['porcentaje_aciertos'], bar_width, label='% de Aciertos', color='#213f99')
            ax1.set_xlabel('Categoría')
            ax1.set_ylabel('% de Aciertos', color='#213f99')
            ax1.set_xticks(indices)
            ax1.set_xticklabels(estadisticas.index)
            ax1.legend(loc='upper left')

            # Gráfico de número de predicciones en el mismo gráfico (eje derecho)
            ax2 = ax1.twinx()
            ax2.bar(indices + bar_width/2, estadisticas['numero_predicciones'], bar_width, label='Número de Predicciones', color='orange')
            ax2.set_ylabel('Número de Predicciones', color='orange')
            ax2.legend(loc='upper right')

            plt.title('Porcentaje de Aciertos y Número de Predicciones por Categoría')
            fig.tight_layout()
            st.pyplot(fig)

            # Gráfico de la confianza media por categoría
            fig2, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=estadisticas.index, y='confianza_media', data=estadisticas, palette="viridis", ax=ax3)
            ax3.set_title('Confianza Promedio por Categoría')
            ax3.set_ylabel('Confianza Promedio (%)')
            ax3.set_xlabel('Categoría')
            st.pyplot(fig2)

            # Mostramos el dataframe
            # Supongamos que `df` es tu DataFrame original
            
            st.dataframe(df, use_container_width=True, hide_index=True, column_order=['archivo', 'pred', 'conf', 'real', 'fecha'])

        # Si no hay historial en sesión (lista vacía) mostramos mensaje de información 
        else:
            st.info('No hay estadísticas disponibles.')
        
if __name__ == '__main__':
    main()
    
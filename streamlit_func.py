# Copyright 2024 Sergio Tejedor Moreno

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st

DEFAULT_COLOR = '#8398a0'

def imagen_con_enlace(url_imagen, url_enlace, 
                    alt_text="Imagen", 
                    max_width:int=100, 
                    centrar:bool=False, 
                    radio_borde:int=15,
                    ) -> None:
    """Muestra una imagen que es también un hipervínculo en Streamlit con bordes redondeados.

    Args:
    url_imagen (str): URL de la imagen a mostrar.
    url_enlace (str): URL a la que el enlace de la imagen debe dirigir.
    alt_text (str): Texto alternativo para la imagen.
    max_width (int): Ancho máximo de la imagen como porcentaje.
    centrar (bool): Si es verdadero, centra la imagen.
    radio_borde (int): Radio del borde redondeado en píxeles.
    """    
    html = f'<a href="{url_enlace}" target="_blank"><img src="{url_imagen}" alt="{alt_text}" style="max-width:{max_width}%; height:auto; border-radius:{radio_borde}px;"></a>'
    if centrar:
        html = f"""
                    <div style='text-align: center'>
                        {html}
                    </div>
                    """
    st.markdown(html, unsafe_allow_html=True)

def texto(texto:str, /, *, 
            font_size:int=10, 
            color:str=DEFAULT_COLOR, 
            font_family:str="Helvetica", 
            formato:str="", 
            centrar:bool=False) -> None:
    """ Función para personalizar el texto con HTML"""
    if formato:
        texto = f"<{formato}>{texto}</{formato}>"
    if centrar:
        texto = f"""
                    <div style='text-align: center'>
                        {texto}
                    </div>
                    """
    texto_formateado = f"""<div style='font-size: {font_size}px; color: {color}; font-family: {font_family}'>{texto}</div>"""
    st.markdown(texto_formateado, unsafe_allow_html=True)

def añadir_salto(num_saltos:int=1) -> None:
    """Añade <br> en forma de HTML para agregar espacio
    """
    saltos = f"{num_saltos * '<br>'}"
    st.markdown(saltos, unsafe_allow_html=True)

def show_sidebar() -> None:
   import streamlit as st
import os

def show_sidebar() -> None:
    """Función para personalizar la sidebar"""
    with st.sidebar:
        # Imagen de la app
        # Carga la imagen desde la ubicación especificada
        
        imagen_path = r'./img/No_hay_PLANetaB.jpeg'
        imagen_path2 = r'./img/logo-austral.png'
        
        if os.path.exists(imagen_path):  # Verifica si la imagen existe
            # st.image(imagen_path, use_column_width=True)
            st.image(imagen_path)
        else:
            st.error("Imagen no encontrada.")

        #with st.sidebar.expander("Modelo2"):
        #modelo_sub_menu = st.radio("Seleccione una opción:", ["Entrenamiento", "Evaluación", "Predicciones"], key="modelo2")

        # Año autor y copyright
        añadir_salto()
        texto('© TF - Diplomatura en IA - Cohorte 2024 ', centrar=True)
        añadir_salto()
        st.image(imagen_path2,use_column_width=True,channels="RGB")
        


def config_page() -> None:
    """Configura los parámetros de la página"""
    st.set_page_config(
        page_title=f"No hay PLANeta B",
        page_icon="🌎", 
        layout="centered",
        initial_sidebar_state="auto",
    )
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

"""Script que recoge el código relacionado con la visualización de las
características del modelo entrenado"""

import time

import streamlit as st

from models.convnet_model import get_model_summary
from streamlit_func import show_sidebar, config_page

# funciones auxiliares
def stream_model_info() -> None:
    """Streamea la información del modelo"""
    stream_container = st.empty()
    with stream_container:
        output = ""
        for letter in get_model_summary():
            output += letter
            st.code(output)
            time.sleep(0.01)

def print_model_info() -> None:
    """Printea toda la información del modelo"""
    st.code(get_model_summary())

def main_model() -> None:
    """Entry point de la app"""

    # Configuración de la app como función en streamlit_func
    config_page()
    # Configuramos la sidebar también importando de streamlit_func
    show_sidebar()

    st.title('Red Neuronal Convolucional')
    st.subheader('Dataset')
    st.image('img/MNIST Dataset example.JPG')
    st.subheader('Arquitectura')

    if st.session_state.get('session_flag') is None:
        # Es la primera vez que entramos en la página modelo
        # asi que streameamos la info del modelo
        stream_model_info()
        # Cambiamos la flag a True
        st.session_state['session_flag'] = True
    else:
        # No es la primera vez que entramos asi que
        # mostramos todo directamente
        print_model_info()

if __name__ == '__main__':
    main_model()
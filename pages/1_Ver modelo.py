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
import os
import webbrowser

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
    
    config_page()
    
    show_sidebar()
    #st.write(dir())
    #st.write(st.session_state)
    notebook_path = './pages/NoHayPLANetaB.html'
    with open(notebook_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    
    st.components.v1.html(html_content, height=800,width=800, scrolling=True)
        
    
    

if __name__ == '__main__':
    main_model()
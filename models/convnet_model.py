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

"""Script que recoge funciones relacionadas con el modelo convnet"""

from io import StringIO
from pathlib import Path

import streamlit as st
from tensorflow import keras

# Constantes
MODEL_PATH = Path('models')

# Funciones
@st.cache_resource()
def load_model(from_weights: bool = True) -> keras.Model:
    """Devuelve el modelo con los weights cargados.

    Returns
    -------
    keras.Model
        El modelo con los coeficientes integrados.
    """
    if from_weights:
        # Construimos el modelo
        model = build_model()
        # Ruta a los weights del modelo
        weights = MODEL_PATH / 'model_NoHayPlanetaB.weights.h5'
        # Cargamos los weights
        model.load_weights(weights, by_name=True, skip_mismatch=True)
    else:
        model = keras.models.load_model(MODEL_PATH /'model_NoHayPlanetaB_tf2.keras')
    
    return model

def build_model() -> keras.Model:
    """Reconstruye el modelo y lo devuelve para posteriormente ser cargado
    con los weights del modelo entrenado. Lo tenemos que hacer asi
    debido a incompatibilidades al guardado el modelo original:

    https://github.com/keras-team/keras-core/issues/855

    Returns
    -------
    keras.Model
        Devuelve el modelo con los weights sin entrenar
    """
    # Usamos la forma 'Functional API' de Keras    
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_model_summary() -> str:
    """Devuelve el detalle del modelo como lo saca keras con
    el método summary

    Returns
    -------
    str
        detalle del modelo
    """
    # Captura la salida de model.summary()
    stream = StringIO()
    model = load_model()
    print_fn = lambda x, **kwargs: stream.write(x + '\n')
    model.summary(print_fn=print_fn)
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

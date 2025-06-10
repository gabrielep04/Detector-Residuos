import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Carga del modelo entrenado
model = load_model('./model.h5')
# Si lo guardaste dinámicamente según train_generator.num_classes, usa ese nombre.

# Clases 
classes = np.load('./classes.npy', allow_pickle=True)

# Función de preprocesamiento
def preprocess_image(image):
    img_resized = cv2.resize(image, (224, 224))
    img_array  = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Interfaz Streamlit ---
st.title("🌿 Detector y Clasificador de Residuos")
st.write("Sube una foto de un residuo y el modelo te dirá de qué tipo es.")

uploaded = st.file_uploader("Elige una imagen", type=["jpg","jpeg","png"])
if uploaded:
    # Leer bytes y decodificar
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Mostrar imagen
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Imagen cargada", use_column_width=True)
    
    # Predecir
    input_array = preprocess_image(frame)
    preds       = model.predict(input_array)
    label       = classes[np.argmax(preds)]
    
    st.success(f"Residuos detectado: **{label}**")
    st.write(f"Confianza: {preds[0][np.argmax(preds)]:.2f}")

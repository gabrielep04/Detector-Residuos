import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Rutas
MODEL_PATH  = 'modelo_residuos_entrenado_5_clases.h5'
DATA_DIR    = 'dataset'
BATCH_SIZE  = 32
IMG_SIZE    = (224, 224)

# 1) Cargar modelo
model = load_model(MODEL_PATH)

# 2) Generador de validación
datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 3) Predecir y extraer etiquetas
y_true = val_gen.classes
labels = list(val_gen.class_indices.keys())
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# 4) Reporte y matriz
report = classification_report(y_true, y_pred, target_names=labels)
cm     = confusion_matrix(y_true, y_pred)

print("=== Classification Report ===\n", report)
print("=== Confusion Matrix ===\n", cm)

# 5) Guardar a disco
with open('reporte_clasificacion.txt', 'w') as f:
    f.write(report)

df_cm = pd.DataFrame(cm, index=labels, columns=labels)
df_cm.to_csv('matriz_confusion.csv')

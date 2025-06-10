import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Cargar imágenes
def load_data(path):
    images, labels = [], []
    classes = os.listdir(path)
    for class_name in classes:
        class_path = os.path.join(path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(class_name)
    return np.array(images), np.array(labels)

# Ruta a tus datos (ajusta esto)
images, labels = load_data("dataset")

# Preprocesamiento
images = images / 255.0  # Normalizar

# Convertir etiquetas a números
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2)

# Modelo (Transfer Learning con MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(len(le.classes_), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Guardar modelo
model.save("model.h5")
np.save("classes.npy", le.classes_)
# Detección y Clasificación de Residuos con IA

## 6. Evaluación del Sistema

### 6.1. Script de Evaluación

El archivo `evaluacion_residuos.py` realiza los siguientes pasos:

1. Carga del modelo entrenado: `modelo_residuos_entrenado_5_clases.h5`.
2. Preparación del generador de validación (`ImageDataGenerator` con `rescale=1./255`).
3. Predicción sobre el conjunto de validación.
4. Cálculo de métricas (precisión, recall, f1-score) mediante `classification_report`.
5. Generación de la matriz de confusión con `confusion_matrix`.
6. Guardado de los resultados en:

   * `reporte_clasificacion.txt`
   * `matriz_confusion.csv`

### 6.2. Cómo ejecutar la evaluación

```bash
source venv/bin/activate
python evaluacion_residuos.py
```

### 6.3. Resultados de la evaluación

* **Reporte de clasificación**: `reporte_clasificacion.txt` (precision, recall, f1-score por clase).
* **Matriz de confusión**: `matriz_confusion.csv`.

### 6.4. Limitaciones

* Sensibilidad a variaciones de iluminación y calidad de la imagen.
* Posible confusión entre clases con características visuales similares.
* Sesgo por distribuciones desiguales de imágenes entre clases.
* Dependencia de la calidad del dataset y la cámara.

## 7. Entrega del Proyecto

### 7.1. Estructura del repositorio

```
Residuos-IA/
├── clasificador_residuos.py
├── detector_streamlit.py
├── evaluacion_residuos.py
├── modelo_residuos_entrenado_6_clases.h5
├── dataset/
│   ├── Metal/
│   ├── Vidrio/
│   ├── Plástico/
│   ├── Cartón/
│   ├── Papel/
│   └── Orgánico/
├── matriz_confusion.csv
├── reporte_clasificacion.txt
├── requirements.txt
└── README.md
```

### 7.2. Requisitos y dependencias

En `requirements.txt`:

```txt
tensorflow-macos
tensorflow-metal
opencv-python
pillow
numpy
matplotlib
scikit-learn
pandas
streamlit
```

### 7.3. Instalación

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 7.4. Uso

1. **Entrenar el modelo**

   ```bash
   python clasificador_residuos.py
   ```
2. **Evaluar el desempeño**

   ```bash
   python evaluacion_residuos.py
   ```
3. **Interfaz de detección**

   ```bash
   streamlit run detector_streamlit.py
   ```

### 7.5. Informe del proyecto

El archivo `informe_proyecto.pdf` incluye:

* Introducción, metodología, resultados, discusión y conclusiones.

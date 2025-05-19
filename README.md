# Predicción de Resultados Deportivos

Este proyecto implementa diferentes modelos de aprendizaje automático para predecir resultados de partidos de fútbol (empate, victoria local o victoria visitante) utilizando datos históricos.

## 📋 Descripción

El sistema utiliza diferentes técnicas de Machine Learning para analizar datos históricos de partidos de fútbol y predecir resultados futuros. Se han implementado y comparado tres tipos de modelos:

- **Red Neuronal MLP** (Multi-Layer Perceptron)
- **SVM** (Support Vector Machine)
- **Red Neuronal Profunda** (Keras/TensorFlow)

## 🚀 Estructura del Proyecto

```
.
├── dataset/                         # Datos de partidos históricos
│   ├── results.csv                  # Resultados de partidos
│   ├── goalscorers.csv              # Datos de goleadores
│   ├── former_names.csv             # Normalización de nombres de equipos
│   └── shootouts.csv                # Información sobre tandas de penaltis
│
├── modelVersions/                   # Modelos entrenados y guardados
│   ├── graphs/                      # Gráficos del proceso de entrenamiento
│   ├── keras_model_*.h5             # Modelos Keras guardados
│   ├── keras_scaler_*.joblib        # Escaladores para modelos Keras
│   ├── mlp_model_*.joblib           # Modelos MLP guardados
│   └── svm_model_*.joblib           # Modelos SVM guardados
│
├── report_assets/                   # Imágenes y gráficos para informes
│
├── train_keras_single.py            # Script para entrenar modelo con Keras
├── MLPClassifier.py                 # Implementación de modelo MLP
├── SVMModel.py                      # Implementación de modelo SVM
├── test_all_models.py               # Script para evaluar todos los modelos
├── main.py                          # Punto de entrada principal
└── requirements.txt                 # Dependencias del proyecto
```

## 🔧 Instalación

1. Clona este repositorio
2. Crea un entorno virtual de Python:
   ```
   python -m venv .venv
   ```
3. Activa el entorno virtual:
   ```
   # En Windows
   .venv\Scripts\activate
   
   # En macOS/Linux
   source .venv/bin/activate
   ```
4. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## 💻 Uso

### Entrenar un nuevo modelo

Para entrenar un nuevo modelo con Keras:
```
python train_keras_single.py
```

### Evaluar todos los modelos

Para comparar el rendimiento de todos los modelos guardados:
```
python test_all_models.py
```

### Ejecutar la aplicación principal

Para ejecutar la aplicación principal:
```
python main.py
```

## 📊 Características utilizadas

El modelo analiza diversas características para hacer predicciones:

- Equipos (local y visitante)
- Campo neutral
- Goles marcados
- Definición por penaltis
- Estadísticas históricas de equipos:
  - Promedio de goles a favor y en contra
  - Tendencia de resultados

## 📈 Métricas de Evaluación

Los modelos son evaluados usando:

- Precisión global (accuracy)
- Matriz de confusión
- Reporte de clasificación (precision, recall, F1-score)

## 🧠 Arquitectura de los Modelos

### Modelo Keras (Red Neuronal Profunda)
- Red neuronal con varias capas densas (256→128→64→32→3)
- Regularización mediante BatchNormalization y Dropout
- Función de activación ReLU en capas ocultas y Softmax en la capa de salida

### Modelo MLP
- Perceptrón multicapa con arquitectura optimizada

### Modelo SVM
- Máquina de vectores de soporte con kernel RBF

## 📝 Notas

- Los modelos se guardan automáticamente con marcas de tiempo
- Se generan gráficos de entrenamiento para analizar el rendimiento
- El script `test_all_models.py` permite comparar objetivamente todos los modelos

## 📚 Requisitos

Ver archivo `requirements.txt` para la lista completa de dependencias.

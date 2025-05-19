import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Importamos las bibliotecas necesarias:
# - pandas y numpy: para manipulación y análisis de datos
# - sklearn: para preprocesamiento, división de datos y métricas de evaluación
# - tensorflow/keras: framework de deep learning para construir y entrenar redes neuronales
# - matplotlib: para visualización de resultados
# - joblib: para guardar modelos y preprocesadores
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import joblib
import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Crear directorio para guardar gráficos si no existe
os.makedirs('modelVersions/graphs', exist_ok=True)

# Callback personalizado para mostrar métricas por época
# Este callback se activa al final de cada época y muestra la precisión de entrenamiento y validación
class AccuracyDisplayCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Época {epoch+1} - accuracy: {logs.get('accuracy'):.4f}, val_accuracy: {logs.get('val_accuracy'):.4f}")

# Cargar datasets
print("Cargando datasets...")
# Cargamos los datasets que contienen la información histórica de partidos de fútbol,
# incluyendo resultados, goleadores, equipos y partidos decididos por penaltis
results = pd.read_csv('dataset/results.csv')
former_names = pd.read_csv('dataset/former_names.csv')
goalscorers = pd.read_csv('dataset/goalscorers.csv')
shootouts = pd.read_csv('dataset/shootouts.csv')

print("Preprocesando datos...")
# Normalizar nombres de equipos usando former_names.csv
name_map = dict(zip(former_names['former'], former_names['current']))
results['home_team'] = results['home_team'].replace(name_map)
results['away_team'] = results['away_team'].replace(name_map)

# Codificar variables categóricas
le_team = LabelEncoder()
all_teams = pd.concat([results['home_team'], results['away_team']]).unique()
le_team.fit(all_teams)
results['home_team_enc'] = le_team.transform(results['home_team'])
results['away_team_enc'] = le_team.transform(results['away_team'])

# Etiqueta: resultado (0=empate, 1=victoria local, 2=victoria visitante)
# Esta función asigna una etiqueta numérica al resultado de cada partido:
# 0 para empate, 1 para victoria del equipo local, 2 para victoria del visitante
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 1
    elif row['home_score'] < row['away_score']:
        return 2
    else:
        return 0
results['result'] = results.apply(get_result, axis=1)
y = results['result'].values

# Característica 1: Número de goles marcados por equipos local y visitante por partido
home_goals = goalscorers.groupby(['date', 'home_team', 'away_team']).apply(
    lambda df: (df['team'] == df['home_team'].iloc[0]).sum()
).reset_index(name='home_goals')
away_goals = goalscorers.groupby(['date', 'home_team', 'away_team']).apply(
    lambda df: (df['team'] == df['away_team'].iloc[0]).sum()
).reset_index(name='away_goals')
results = results.merge(home_goals, on=['date', 'home_team', 'away_team'], how='left')
results = results.merge(away_goals, on=['date', 'home_team', 'away_team'], how='left')
results['home_goals'] = results['home_goals'].fillna(0)
results['away_goals'] = results['away_goals'].fillna(0)

# Característica 2: ¿El partido se decidió por penaltis?
# Creamos una variable binaria que indica si el partido se decidió en tanda de penaltis
results['shootout'] = results.apply(
    lambda row: 1 if ((shootouts['date'] == row['date']) & 
                     (shootouts['home_team'] == row['home_team']) & 
                     (shootouts['away_team'] == row['away_team'])).any() else 0,
    axis=1
)

# Característica 3: Diferencia de goles histórica entre equipos
# Este proceso crea estadísticas acumulativas para cada equipo, calculando 
# el promedio histórico de goles a favor y en contra para reflejar su rendimiento
team_stats = {}
for index, row in results.iterrows():
    home_team = row['home_team']
    away_team = row['away_team']
    
    # Inicializar estadísticas del equipo si no existen
    if home_team not in team_stats:
        team_stats[home_team] = {'goals_for': 0, 'goals_against': 0, 'matches': 0}
    if away_team not in team_stats:
        team_stats[away_team] = {'goals_for': 0, 'goals_against': 0, 'matches': 0}
    
    # Actualizar estadísticas
    team_stats[home_team]['goals_for'] += row['home_score']
    team_stats[home_team]['goals_against'] += row['away_score']
    team_stats[home_team]['matches'] += 1
    
    team_stats[away_team]['goals_for'] += row['away_score']
    team_stats[away_team]['goals_against'] += row['home_score']
    team_stats[away_team]['matches'] += 1

# Calcular promedios de goles para equipos
for team in team_stats:
    matches = max(1, team_stats[team]['matches'])
    team_stats[team]['avg_goals_for'] = team_stats[team]['goals_for'] / matches
    team_stats[team]['avg_goals_against'] = team_stats[team]['goals_against'] / matches

# Añadir características basadas en estadísticas históricas
results['home_avg_goals_for'] = results['home_team'].map(lambda t: team_stats.get(t, {}).get('avg_goals_for', 0))
results['home_avg_goals_against'] = results['home_team'].map(lambda t: team_stats.get(t, {}).get('avg_goals_against', 0))
results['away_avg_goals_for'] = results['away_team'].map(lambda t: team_stats.get(t, {}).get('avg_goals_for', 0))
results['away_avg_goals_against'] = results['away_team'].map(lambda t: team_stats.get(t, {}).get('avg_goals_against', 0))

# Selección de características
# Estas son las variables que usaremos como entrada para nuestro modelo
# incluyen: equipos codificados, cancha neutral, goles, penaltis y estadísticas históricas
features = [
    'home_team_enc', 'away_team_enc', 'neutral', 
    'home_goals', 'away_goals', 'shootout',
    'home_avg_goals_for', 'home_avg_goals_against',
    'away_avg_goals_for', 'away_avg_goals_against'
]

X = results[features].values

# División de entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir etiquetas a one-hot encoding para modelo Keras
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

print(f"Dimensiones de datos - X_train: {X_train.shape}, y_train: {y_train_cat.shape}")

# Normalización de datos con StandardScaler
# La normalización es un paso crítico en redes neuronales por varias razones:
# 1. Las redes neuronales convergen más rápido con datos normalizados
# 2. Previene que características con valores grandes dominen el aprendizaje
# 3. Reduce la sensibilidad a la inicialización de pesos
# 4. Mejora la eficacia del descenso de gradiente
#
# StandardScaler transforma cada característica para que tenga:
# - Media = 0
# - Desviación estándar = 1
# Esto se conoce como normalización Z-score
scaler = StandardScaler()
# Primero ajustamos el scaler solo a los datos de entrenamiento
# Es crucial no incluir datos de prueba en este cálculo para evitar data leakage
scaler.fit(X_train)
# Luego transformamos tanto datos de entrenamiento como de prueba
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurar e inicializar el modelo Keras
print("Configurando modelo Keras Deep Learning...")
input_dim = X_train.shape[1]  # Número de características de entrada
# Tasas de dropout para cada capa (técnica de regularización)
# El dropout desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento
# lo que ayuda a prevenir que la red se sobreajuste a los datos de entrenamiento
dropout_rates = [0.3, 0.4, 0.5, 0.3]
# Tasa de aprendizaje: controla cuánto se ajustan los pesos en cada iteración
# Un valor demasiado alto puede causar divergencia, uno muy bajo hará lento el aprendizaje
learning_rate = 0.001

# Construir modelo
print("Construyendo arquitectura del modelo...")
# Definimos una Red Neuronal Profunda (DNN) secuencial con capas densamente conectadas
# La arquitectura sigue un patrón de reducción progresiva (embudo): 256→128→64→32→3
# Esto permite que la red aprenda representaciones cada vez más abstractas de los datos
model = Sequential([
    # Primera capa densa: 256 neuronas con función de activación ReLU
    # ReLU (Rectified Linear Unit) es una función no lineal que devuelve max(0,x)
    # siendo muy efectiva para aprender patrones complejos sin sufrir el problema de
    # desvanecimiento del gradiente (vanishing gradient) que afecta a otras funciones
    Dense(256, activation='relu', input_shape=(input_dim,)),
    # Batch Normalization: normaliza las activaciones de la capa anterior
    # Esto acelera el entrenamiento y proporciona cierta regularización
    BatchNormalization(),
    # Dropout: desactiva aleatoriamente el 30% de las neuronas durante el entrenamiento
    # obligando a la red a aprender patrones redundantes y evitando la co-adaptación
    Dropout(dropout_rates[0]),
    
    # Segunda capa con 128 neuronas
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(dropout_rates[1]),
    
    # Tercera capa con 64 neuronas
    Dense(64, activation='relu'),
    BatchNormalization(), 
    Dropout(dropout_rates[2]),
    
    # Cuarta capa con 32 neuronas
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(dropout_rates[3]),
    
    # Capa de salida para clasificación multiclase con 3 posibles resultados
    # Softmax convierte las salidas en probabilidades (valores entre 0 y 1 que suman 1)
    # Cada neurona representa la probabilidad de una clase: empate, victoria local o visitante
    Dense(3, activation='softmax')
])

# Compilación con optimizador Adam y entropía cruzada
# El optimizador define cómo se actualizan los pesos de la red durante el entrenamiento
# Adam (Adaptive Moment Estimation) combina las ventajas de RMSProp y SGD con momento,
# adaptando automáticamente la tasa de aprendizaje para cada parámetro
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
    # La función de pérdida que queremos minimizar durante el entrenamiento
    # Categorical crossentropy es la elección estándar para clasificación multiclase
    # Mide la diferencia entre la distribución de probabilidad predicha y la real
    loss='categorical_crossentropy',
    # Métrica para evaluar el rendimiento del modelo durante el entrenamiento
    metrics=['accuracy']
)

# Early stopping para detener entrenamiento cuando la precisión apenas varía
# El early stopping es una técnica de regularización que evita el sobreajuste
# monitorizando una métrica en el conjunto de validación y deteniendo el entrenamiento
# cuando deja de mejorar significativamente.

early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Métrica que se monitoriza (precisión en validación)
    patience=3,              # Número de épocas a esperar sin mejora antes de parar
                             # Un valor bajo puede terminar el entrenamiento prematuramente
                             # Un valor alto podría permitir sobreajuste
    min_delta=0.001,         # Umbral mínimo de mejora para considerarse significativa (0.1%)
                             # Cambios menores se consideran ruido y no mejoras reales
    restore_best_weights=True,  # Recupera los mejores pesos encontrados durante el entrenamiento
                             # no los de la última época, sino los de mejor rendimiento
    mode='max',              # 'max' porque queremos maximizar la precisión
                             # (sería 'min' si monitorizáramos pérdida)
    verbose=1                # Mostrar mensaje cuando se active early stopping
)

# Callback para mostrar accuracy por época
accuracy_display = AccuracyDisplayCallback()

# Entrenar el modelo
print("\nEntrenando el modelo Keras...")
print("-" * 50)
# El método fit() realiza el entrenamiento real del modelo mediante:
# 1. Forward propagation: el modelo genera predicciones con los pesos actuales
# 2. Cálculo del error: se comparan las predicciones con los valores reales
# 3. Backpropagation: se calculan los gradientes para actualizar los pesos
# 4. Optimization: el optimizador (Adam) utiliza los gradientes para ajustar los pesos
history = model.fit(
    X_train_scaled,  # Datos de entrada (características) normalizados
    y_train_cat,     # Etiquetas codificadas en one-hot (3 columnas para las 3 clases)
    validation_split=0.2,   # 20% de los datos de entrenamiento se usan para validación
    batch_size=32,          # Procesa 32 muestras antes de actualizar los pesos
                            # Un batch más pequeño aumenta el ruido pero puede ayudar a generalizar
                            # Un batch más grande proporciona estimaciones más estables del gradiente
    epochs=150,             # Número máximo de iteraciones completas sobre el dataset
                            # El early stopping puede detener el proceso antes
    callbacks=[early_stopping, accuracy_display],  # Funciones que se ejecutan durante el entrenamiento
    verbose=0               # No mostrar la barra de progreso (el callback lo hará)
)
print("-" * 50)
print("Entrenamiento completado")

# Evaluar el modelo en datos de prueba
print("\nEvaluando el modelo...")
# Evaluamos el modelo con datos de prueba que no se utilizaron durante el entrenamiento
# Es fundamental utilizar datos nuevos para obtener una estimación realista del rendimiento
# ya que el modelo podría tener un rendimiento excelente en datos que ya ha visto pero
# generalizar mal con datos nuevos (lo que se conoce como sobreajuste/overfitting)
loss, accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
print(f"Precisión en datos de prueba: {accuracy:.4f}")

# Hacer predicciones
# Obtenemos las probabilidades predichas por el modelo para cada clase
# y seleccionamos la clase con mayor probabilidad (arg max)
y_pred_proba = model.predict(X_test_scaled, verbose=0)  # Devuelve matriz de probabilidades
# Convertimos las probabilidades a etiquetas de clase (índices 0, 1 o 2)
y_pred = np.argmax(y_pred_proba, axis=1)
# Convertimos las etiquetas codificadas (one-hot) a índices para compararlas
y_true = np.argmax(y_test_cat, axis=1)

# Calcular métricas
# Necesitamos métricas más allá de la precisión global para entender el comportamiento del modelo
# especialmente si las clases están desbalanceadas (p. ej. si hay más victorias locales que empates)
class_names = ['Empate', 'Victoria Local', 'Victoria Visitante']
# La matriz de confusión muestra:
# - Filas: clases reales
# - Columnas: clases predichas
# - Diagonal principal: predicciones correctas
# - Fuera de la diagonal: errores (confusiones)
conf_matrix = confusion_matrix(y_true, y_pred)
# El reporte de clasificación proporciona para cada clase:
# - Precision: de todas las predicciones de esa clase, qué porcentaje fue correcto (TP/(TP+FP))
# - Recall: de todos los casos reales de esa clase, qué porcentaje se identificó correctamente (TP/(TP+FN))
# - F1-score: media armónica de precisión y recall, buena métrica para balance entre ambas
class_report = classification_report(y_true, y_pred, target_names=class_names)

# Imprimir resultados
print("\nMatriz de confusión:")
print(conf_matrix)
print("\nReporte de clasificación:")
print(class_report)

# Visualizar historia de entrenamiento
# La visualización del proceso de entrenamiento nos ayuda a entender cómo evolucionó el modelo
# y puede ayudarnos a identificar problemas como sobreajuste o convergencia prematura
plt.figure(figsize=(12, 5))

# Gráfico de precisión
# Este gráfico muestra cómo evoluciona la precisión del modelo a lo largo del entrenamiento
# Comparar las curvas de entrenamiento y validación nos permite detectar:
# - Sobreajuste: si la precisión de entrenamiento sigue mejorando pero la de validación se estanca o empeora
# - Subajuste: si ambas curvas se estancan en valores bajos, indicando que el modelo es demasiado simple
# - Equilibrio óptimo: cuando ambas curvas convergen a un valor alto
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo por época')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# Gráfico de pérdida (loss)
# La función de pérdida es lo que el modelo intenta minimizar durante el entrenamiento
# Idealmente, queremos ver:
# - Ambas curvas descendiendo constantemente y estabilizándose en un valor bajo
# - Curvas relativamente cercanas entre sí (sin gran diferencia entre entrenamiento y validación)
# - Si la pérdida de entrenamiento sigue bajando pero la de validación sube, indica sobreajuste
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo por época')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()

# Generar nombres únicos para modelo y visualizaciones
# Usamos timestamps para generar nombres únicos para cada ejecución
# Esto permite:
# 1. Evitar sobrescribir modelos anteriores
# 2. Mantener un historial de diferentes versiones del modelo
# 3. Comparar rendimiento entre distintas sesiones de entrenamiento
model_version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'modelVersions/keras_model_{model_version}.h5'
scaler_path = f'modelVersions/keras_scaler_{model_version}.joblib'
graph_path = f'modelVersions/graphs/keras_training_{model_version}.png'

# Guardar gráfico
# Guardamos la visualización para documentación y análisis posterior
plt.savefig(graph_path)
plt.show()
print(f"Gráfico de entrenamiento guardado como {graph_path}")

# Guardar modelo y scaler
# Es crucial guardar tanto el modelo como el scaler, ya que:
# 1. El modelo contiene la arquitectura y los pesos aprendidos
# 2. El scaler es necesario para aplicar la misma transformación a nuevos datos
# Ambos son necesarios para hacer predicciones correctas en el futuro
model.save(model_path)  # Guarda arquitectura, pesos, estado del optimizador, etc.
joblib.dump(scaler, scaler_path)  # Guarda el scaler con sus parámetros (media, std, etc.)
print(f"Modelo y scaler guardados como keras_model_{model_version}.h5 y keras_scaler_{model_version}.joblib")

print("¡Entrenamiento completado con éxito!")

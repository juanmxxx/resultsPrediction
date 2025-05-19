import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import glob
import os

# Create a larger test DataFrame with 100 invented matches
np.random.seed(42)
n_samples = 100
n_teams = 30  # Assume 30 unique teams for encoding

# Creamos dos conjuntos de datos de prueba: uno con 6 características para MLP/SVM y otro con 10 para Keras
# Datos de prueba básicos con 6 características (para modelos MLP y SVM)
test_data_basic = pd.DataFrame({
    'home_team_enc': np.random.randint(0, n_teams, n_samples),
    'away_team_enc': np.random.randint(0, n_teams, n_samples),
    'neutral': np.random.randint(0, 2, n_samples),
    'home_goals': np.random.poisson(1.5, n_samples),
    'away_goals': np.random.poisson(1.2, n_samples),
    'shootout': np.random.randint(0, 2, n_samples)
})

# Datos de prueba completos con 10 características (para modelos Keras)
test_data_full = test_data_basic.copy()
# Añadimos las 4 características históricas adicionales
test_data_full['home_avg_goals_for'] = np.random.uniform(0.5, 2.5, n_samples)
test_data_full['home_avg_goals_against'] = np.random.uniform(0.5, 2.0, n_samples)
test_data_full['away_avg_goals_for'] = np.random.uniform(0.5, 2.0, n_samples)
test_data_full['away_avg_goals_against'] = np.random.uniform(0.5, 2.5, n_samples)

# Invented true labels: 0=draw, 1=home win, 2=away win
# Use a simple rule: if home_goals > away_goals -> 1, < -> 2, == -> 0
def invent_result(row):
    if row['home_goals'] > row['away_goals']:
        return 1
    elif row['home_goals'] < row['away_goals']:
        return 2
    else:
        return 0
y_true = test_data_basic.apply(invent_result, axis=1).values

print("\n--- Testing all saved models in modelVersions/ ---\n")

# Test all MLP models
mlp_models = glob.glob('modelVersions/mlp_model_*.joblib')  # Changed from hardcoded path
for model_path in mlp_models:
    print(f"Testing MLP model: {os.path.basename(model_path)}")
    mlp = joblib.load(model_path)
    y_pred = mlp.predict(test_data_basic)  # Usar versión básica con 6 características
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['Draw', 'Home Win', 'Away Win']))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("-"*50)

# Test all SVM models
svm_models = glob.glob('modelVersions/svm_model_*.joblib')
for model_path in svm_models:
    print(f"Testing SVM model: {os.path.basename(model_path)}")
    svm = joblib.load(model_path)
    y_pred = svm.predict(test_data_basic)  # Usar versión básica con 6 características
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=['Draw', 'Home Win', 'Away Win']))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("-"*50)

# Test all Keras models
keras_models = glob.glob('modelVersions/keras_model_*.h5')
for model_path in keras_models:
    print(f"Testing Keras model: {os.path.basename(model_path)}")
    # Find corresponding scaler
    version = os.path.basename(model_path).replace('keras_model_', '').replace('.h5', '')
    scaler_path = f'modelVersions/keras_scaler_{version}.joblib'
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_test_scaled = scaler.transform(test_data_full)  # Usar versión completa con 10 características
        model = load_model(model_path)
        y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred, target_names=['Draw', 'Home Win', 'Away Win']))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    else:
        print("Scaler not found for this Keras model.")
    print("-"*50)

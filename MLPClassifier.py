import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import joblib
import time
import datetime

# Load datasets
results = pd.read_csv('dataset/results.csv')
former_names = pd.read_csv('dataset/former_names.csv')

# Normalize team names using former_names.csv
name_map = dict(zip(former_names['former'], former_names['current']))
results['home_team'] = results['home_team'].replace(name_map)
results['away_team'] = results['away_team'].replace(name_map)

# Encode categorical variables
le_team = LabelEncoder()
all_teams = pd.concat([results['home_team'], results['away_team']]).unique()
le_team.fit(all_teams)
results['home_team_enc'] = le_team.transform(results['home_team'])
results['away_team_enc'] = le_team.transform(results['away_team'])

# Feature selection: home_team, away_team, neutral
X = results[['home_team_enc', 'away_team_enc', 'neutral']].values

# Label: result (0=draw, 1=home win, 2=away win)
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 1
    elif row['home_score'] < row['away_score']:
        return 2
    else:
        return 0
results['result'] = results.apply(get_result, axis=1)
y = results['result'].values

# Enrich features with goalscorers and shootouts data

goalscorers = pd.read_csv('dataset/goalscorers.csv')
shootouts = pd.read_csv('dataset/shootouts.csv')

# Feature 1: Number of goals scored by home and away teams per match
# Aggregate goals for home team
home_goals = goalscorers.groupby(['date', 'home_team', 'away_team']).apply(
    lambda df: (df['team'] == df['home_team'].iloc[0]).sum()
).reset_index(name='home_goals')
# Aggregate goals for away team
away_goals = goalscorers.groupby(['date', 'home_team', 'away_team']).apply(
    lambda df: (df['team'] == df['away_team'].iloc[0]).sum()
).reset_index(name='away_goals')
# Merge goals into results
results = results.merge(home_goals, on=['date', 'home_team', 'away_team'], how='left')
results = results.merge(away_goals, on=['date', 'home_team', 'away_team'], how='left')
results['home_goals'] = results['home_goals'].fillna(0)
results['away_goals'] = results['away_goals'].fillna(0)

# Feature 2: Was the match decided by shootout?
results['shootout'] = results.apply(
    lambda row: 1 if ((shootouts['date'] == row['date']) & (shootouts['home_team'] == row['home_team']) & (shootouts['away_team'] == row['away_team'])).any() else 0,
    axis=1
)

# Update feature selection
X = results[['home_team_enc', 'away_team_enc', 'neutral', 'home_goals', 'away_goals', 'shootout']].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network using scikit-learn (compatible with Python 3.13)
# Tune neural network parameters for better performance
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # more layers and neurons
    activation='relu',
    solver='adam',
    alpha=0.001,  # lower regularization
    learning_rate_init=0.002,  # higher learning rate
    max_iter=500,  # per epoch
    batch_size=32,
    random_state=42,
    early_stopping=False,  # Disable for partial_fit
    n_iter_no_change=20
)

# Custom training loop to print epoch timing
n_epochs = 5
max_iter_per_epoch = 500  # 1500/3
for epoch in range(n_epochs):
    start_time = time.time()
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))
    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{n_epochs} completed in {elapsed:.2f} seconds.")

# Save the trained model to a new file with a unique name per execution
model_version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = f'modelVersions/mlp_model_{model_version}.joblib'
joblib.dump(mlp, model_path)
print(f"Model saved to {model_path}")

# Evaluate the model
accuracy = mlp.score(X_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

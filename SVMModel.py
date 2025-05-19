import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import datetime

# Load datasets
results = pd.read_csv('dataset/results.csv')
former_names = pd.read_csv('dataset/former_names.csv')
goalscorers = pd.read_csv('dataset/goalscorers.csv')
shootouts = pd.read_csv('dataset/shootouts.csv')

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

# Feature 1: Number of goals scored by home and away teams per match
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

# Feature 2: Was the match decided by shootout?
results['shootout'] = results.apply(
    lambda row: 1 if ((shootouts['date'] == row['date']) & (shootouts['home_team'] == row['home_team']) & (shootouts['away_team'] == row['away_team'])).any() else 0,
    axis=1
)

# Feature selection
X = results[['home_team_enc', 'away_team_enc', 'neutral', 'home_goals', 'away_goals', 'shootout']].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train SVM model (with RBF kernel)
svm = SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.2f}')

# Save the model and scaler
model_version = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
joblib.dump(svm, f'modelVersions/svm_model_{model_version}.joblib')
joblib.dump(scaler, f'modelVersions/svm_scaler_{model_version}.joblib')
print(f"Model and scaler saved as svm_model_{model_version}.joblib and svm_scaler_{model_version}.joblib")

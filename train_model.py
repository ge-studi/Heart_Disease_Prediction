import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pickle

# --------------------
# 1. Load Dataset
# --------------------
url = 'https://raw.githubusercontent.com/ge-studi/Heart_Disease_Prediction/main/heart_disease_uci.csv'
df = pd.read_csv(url)

# --------------------
# 2. Preprocessing
# --------------------
df.dropna(inplace=True)  # remove missing values

# Encode categorical/binary features
label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])
df['dataset'] = label_encoder.fit_transform(df['dataset'])

# Define features and target
X = df.drop(columns=['id', 'num'])
y = df['num'].apply(lambda x: 1 if x > 0 else 0)  # binary classification

# One-hot encode categorical features
categorical_cols = ['cp', 'restecg', 'slope', 'thal']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# --------------------
# 3. Model Definitions
# --------------------
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

grid_search_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X_train, y_train)
best_gb = grid_search_gb.best_estimator_

ensemble_model = VotingClassifier(
    estimators=[('rf', best_rf), ('gb', best_gb), ('lr', LogisticRegression(max_iter=500))],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

# --------------------
# 4. Model Evaluation
# --------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_prob)
    }

results = evaluate_model(ensemble_model, X_test, y_test)
print("Ensemble Model Performance:", results)

# --------------------
# 5. Save Model, Scaler, Columns
# --------------------
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(ensemble_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model, scaler, and columns saved!")

import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score


os.makedirs('img', exist_ok=True)

# --- Chargement des données préparées
grouped = pd.read_csv('dataset_ml.csv')

X = grouped.drop(columns=['Table_Name', 'Table_Type'])
y = grouped['Table_Type']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gestion du déséquilibre
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

# Modèle XGBoost
xgb_model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

# Hyperparamètres pour RandomSearch
param_dist = {
    'n_estimators': [300, 500, 700],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.7, 0.8],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.05, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=60,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Entraînement SANS early stopping pendant la recherche
random_search.fit(X_train, y_train)

print("Meilleurs hyperparamètres :", random_search.best_params_)

# ===== Réentraînement final =====
best_params = random_search.best_params_
best_model = XGBClassifier(
    **best_params,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

best_model.fit(X_train, y_train)


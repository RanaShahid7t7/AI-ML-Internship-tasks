import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Telco-Customer-Churn.csv')

# Preprocessing
print("Preprocessing data...")
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Target
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify column types
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numeric_features]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Pipelines
log_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter grids
log_param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear']
}

rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

# GridSearch for Logistic Regression
print("Tuning Logistic Regression...")
log_grid = GridSearchCV(log_pipeline, log_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
log_grid.fit(X_train, y_train)

print(f"Best Logistic Regression params: {log_grid.best_params_}")
print(f"Best Logistic Regression score: {log_grid.best_score_:.4f}")

# GridSearch for Random Forest
print("Tuning Random Forest...")
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f"Best Random Forest params: {rf_grid.best_params_}")
print(f"Best Random Forest score: {rf_grid.best_score_:.4f}")

# Evaluate best models
models = {
    'Logistic Regression': log_grid.best_estimator_,
    'Random Forest': rf_grid.best_estimator_
}

for name, model in models.items():
    print(f"\n{name} Test Results:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Export the best pipeline (Random Forest as example)
print("Exporting pipeline...")
joblib.dump(rf_grid.best_estimator_, 'churn_pipeline.pkl')
print("Pipeline saved as 'churn_pipeline.pkl'")

# Optional: Feature importance for Random Forest
rf_model = rf_grid.best_estimator_.named_steps['classifier']
preprocessor = rf_grid.best_estimator_.named_steps['preprocessor']

# Get feature names
ohe = preprocessor.named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = numeric_features + list(cat_feature_names)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(feature_names)), importances[indices], align="center")
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show()
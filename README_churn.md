# End-to-End ML Pipeline for Customer Churn Prediction

This project implements a production-ready machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset.

## Dataset
- **Source**: IBM Telco Customer Churn Dataset
- **Size**: ~7,000 customers with 21 features
- **Target**: Churn (Yes/No)

## Pipeline Features
- **Preprocessing**: 
  - Standard scaling for numeric features (tenure, MonthlyCharges, TotalCharges)
  - One-hot encoding for categorical features
- **Models**: Logistic Regression and Random Forest
- **Hyperparameter Tuning**: GridSearchCV with 5-fold CV
- **Export**: Complete pipeline saved with joblib

## Skills Demonstrated
- ML pipeline construction with Scikit-learn Pipeline API
- Hyperparameter tuning with GridSearchCV
- Model export and reusability
- Production-readiness practices

## Usage
1. Ensure dataset is downloaded: `Telco-Customer-Churn.csv`
2. Run the pipeline: `python churn_pipeline.py`

## Output
- Best hyperparameters and CV scores for both models
- Test set classification reports and confusion matrices
- Exported pipeline: `churn_pipeline.pkl`
- Feature importance plot: `feature_importances.png`

## Loading the Pipeline
```python
import joblib
pipeline = joblib.load('churn_pipeline.pkl')
predictions = pipeline.predict(new_data)
```
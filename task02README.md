 Task 02 — House Prices (Regression)

Objective
 Build and compare regression models to predict house prices.

Dataset
`task 02/house_prices.csv`

Notebook
`task 02/01.ipynb`

Models applied
 Linear Regression (scikit-learn)
 Random Forest Regressor (scikit-learn)

Key results (from the notebook)
 Linear Regression — Training RMSE: $3.40, Testing RMSE: $3.48; Training MAE: $2.37, Testing MAE: $2.46.
 Random Forest — Training RMSE: $1.46, Testing RMSE: $11.36; Training MAE: $1.03, Testing MAE: $7.99.

Finding
 Random Forest achieved much lower training error but much higher test error, indicating overfitting on this split. Linear Regression showed more consistent generalization.

# Multimodal Housing Price Prediction

This project implements a multimodal machine learning model to predict housing prices using both tabular data and house images.

## Dataset
- **Tabular Data**: Boston Housing Dataset (506 samples, 13 features)
- **Image Data**: Houses Dataset from GitHub (535 house images, using first 506)

## Features
- Uses VGG16 CNN to extract features from images (25088 features per image)
- Combines image features with standardized tabular features
- Trains a neural network regressor
- Evaluates using MAE and RMSE

## Requirements
- Python 3.8+
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Pillow
- Matplotlib

## Usage
Run the script:
```bash
python multimodal_housing.py
```

The script will:
1. Load and preprocess tabular data
2. Extract features from images using VGG16
3. Combine features
4. Train a neural network
5. Evaluate and print MAE/RMSE
6. Save training history plot

## Output
- Console output with MAE and RMSE
- `training_history.png`: Plot of training/validation loss and MAE

## Skills Demonstrated
- Multimodal ML
- CNN feature extraction
- Feature fusion
- Regression modeling
- Model evaluation
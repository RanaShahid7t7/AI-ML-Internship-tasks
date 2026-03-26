import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# Load tabular data (Boston Housing dataset)
print("Loading tabular data...")
data = fetch_openml('boston', version=1, as_frame=True)
X_tab = data.data.values
y = data.target.values

# Standardize tabular data
scaler = StandardScaler()
X_tab = scaler.fit_transform(X_tab)

# Load and process images
print("Loading and processing images...")
image_dir = 'Houses-dataset-master/Houses Dataset'
image_files = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('.jpg'):
            image_files.append(os.path.join(root, file))

# Sort to ensure order
image_files.sort()

# Take first 506 images to match tabular data
image_files = image_files[:506]

# Load VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

features = []
for i, img_path in enumerate(image_files):
    if i % 50 == 0:
        print(f"Processing image {i+1}/{len(image_files)}")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = base_model.predict(img_array, verbose=0)
    feature = feature.flatten()
    features.append(feature)

X_img = np.array(features)

# Combine tabular and image features
print("Combining features...")
X_combined = np.concatenate([X_tab, X_img], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Build and train model
print("Building and training model...")
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Optional: Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.legend()
plt.title('MAE')

plt.savefig('training_history.png')
plt.show()
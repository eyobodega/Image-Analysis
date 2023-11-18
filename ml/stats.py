import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import os 
from tensorflow.keras.preprocessing import image

import pickle

os.makedirs('output', exist_ok=True)

# Load the images and labels
def load_data(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    images = []
    labels = []
    
    for img_file in image_files:
        # Load the image file
        img_path = os.path.join(image_dir, img_file)
        img = image.load_img(img_path, target_size=(512, 512))
        img = image.img_to_array(img)
        img /= 255.0  # Normalize the image
        images.append(img)
        
        # Load the label file
        label_path = os.path.join(label_dir, img_file.replace('.png', '.txt'))
        with open(label_path, 'r') as file:
            count = float(file.read().strip())
            labels.append(count)
    
    return np.array(images), np.array(labels)

# Load the saved model
model = tf.keras.models.load_model('staples_dataset/my_model1.h5')

# Load validation data
val_images, val_labels = load_data('staples_dataset/images/val', 'staples_dataset/val_labels')

# Predict the counts for validation images
predicted_counts = model.predict(val_images)

# Calculate residuals
residuals = val_labels - predicted_counts.flatten()

# Load the history from the pickle file
with open('staples_dataset/train_history5.pkl', 'rb') as file:
    history = pickle.load(file)

# Calculate error metrics
mse = mean_squared_error(val_labels, predicted_counts.flatten())
rmse = sqrt(mse)
mae_value = mean_absolute_error(val_labels, predicted_counts.flatten())
huber_loss = tf.keras.losses.Huber()
huber_loss_value = huber_loss(val_labels, predicted_counts.flatten()).numpy()

# Print error metrics
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"Huber Loss: {huber_loss_value}")
print(f"Mean Absolute Error (MAE): {mae_value}")

# Actual vs Predicted Plot
plt.figure(figsize=(12, 6))
plt.scatter(range(len(val_labels)), val_labels, color='blue', label='Actual Count')
plt.scatter(range(len(predicted_counts)), predicted_counts.flatten(), color='red', label='Predicted Count')
plt.title('Actual vs Predicted Count of Staples')
plt.xlabel('Image Index')
plt.ylabel('Count of Staples')
plt.legend()
plt.savefig("output/actual_vs_predicted.png")

# Residuals Plot
plt.figure(figsize=(12, 6))
plt.scatter(range(len(residuals)), residuals, color='orange', label='Residuals')
plt.axhline(y=0, color='green', linestyle='--')
plt.title('Residuals of Predictions')
plt.xlabel('Image Index')
plt.ylabel('Residuals (Actual Count - Predicted Count)')
plt.legend()
plt.savefig("output/residuals.png")

# Error Distribution
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.savefig("output/error_distribution.png")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss during Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("output/learning_curve.png")

# Plot the training and validation accuracy (if available)
if 'accuracy' in history:
    plt.figure(figsize=(12, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Plotting Actual vs Predicted Counts for each Image Index
plt.figure(figsize=(12, 6))
plt.scatter(range(len(val_labels)), val_labels, color='blue', label='Actual Count')
plt.scatter(range(len(predicted_counts)), predicted_counts.flatten(), color='red', label='Predicted Count')
plt.title('Actual vs Predicted Count of Staples')
plt.xlabel('Image Index')
plt.ylabel('Count of Staples')
plt.legend()
plt.savefig("output/actual_vs_predicted2.png")

# Plotting Error Metrics for each Image Index
plt.figure(figsize=(12, 6))
plt.axhline(y=mse, color='blue', linestyle='-', label='MSE')
plt.axhline(y=rmse, color='orange', linestyle='-', label='RMSE')
plt.axhline(y=mae_value, color='green', linestyle='-', label='MAE')
plt.axhline(y=huber_loss_value, color='red', linestyle='-', label='Huber Loss')
plt.title('Error Metrics for each Staple Count Prediction')
plt.xlabel('Image Index')
plt.ylabel('Error Metric Value')
plt.legend()
plt.savefig("output/error_metrics.png")
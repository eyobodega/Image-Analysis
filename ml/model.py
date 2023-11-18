import numpy as np
import os
from tensorflow.keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,UpSampling2D,Input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
import tensorflow as tf

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np
import os

import pickle
# Data directories
image_dir = 'staples_dataset/images/train'
label_dir = 'staples_dataset/train_labels'

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

# Use the load_data function to get your inputs and outputs
# Load training data
train_image_dir = 'staples_dataset/images/train'
train_label_dir = 'staples_dataset/train_labels'
train_images, train_labels = load_data(train_image_dir, train_label_dir)

# Check the shape of the loaded data
print(f'Image data shape: {train_images.shape}')
print(f'Label data shape: {train_labels.shape}')

def create_counting_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    # Flatten and use a Dense layer for counting
    x = Flatten()(x)
    count = Dense(1, activation='linear')(x)  # One neuron for one count value
    
    model = Model(inputs=inputs, outputs=count)
    return model

# Create and compile the model
model = create_counting_model(input_shape=(512, 512, 3))
model.compile(optimizer='adam', loss='mse',metrics=['mae'])

# Load and preprocess validation data
val_image_dir = 'staples_dataset/images/val'
val_label_dir = 'staples_dataset/val_labels'
val_images, val_labels = load_data(val_image_dir, val_label_dir)

# Train the model
# model.fit(train_images, train_labels, batch_size=32, epochs=30, validation_data=(val_images, val_labels))
# Train the model and capture the history
history = model.fit(train_images, train_labels, batch_size=32, epochs=30, validation_data=(val_images, val_labels))

# Saving the history

with open('staples_dataset/train_history5.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Predicting on a new image
# Make sure test_image is preprocessed to have shape (512, 512, 3)
# Load and preprocess the test image
# test_image_path = 'staples_dataset/images/val/staples_2.png'
# test_image = image.load_img(test_image_path, target_size=(512, 512))
# test_image = image.img_to_array(test_image)
# test_image /= 255.0  # Normalize the image

# Predict the count
# predicted_count = model.predict(np.expand_dims(test_image, axis=0))
# print("Predicted count:", predicted_count[0][0])

# The output is the predicted count, you can compare this with the actual count
# Predict and compare with actual counts
for i in range(len(val_images)):
    predicted_count = model.predict(np.expand_dims(val_images[i], axis=0))
    print(f"Image: {i}, Predicted count: {predicted_count[0][0]}, Actual count: {val_labels[i]}")



# Predict counts for validation images
predicted_counts = model.predict(val_images)

# Calculate Mean Squared Error
mse = mean_squared_error(val_labels, predicted_counts.flatten())
rmse = sqrt(mse)
mae_value = mean_absolute_error(val_labels, predicted_counts.flatten())

# Create a huber loss object
huber_loss = tf.keras.losses.Huber()

# Assuming 'val_labels' and 'predicted_counts' are numpy arrays
# You would convert these to tensors before calculating the loss
val_labels_tensor = tf.convert_to_tensor(val_labels, dtype=tf.float32)
predicted_counts_tensor = tf.convert_to_tensor(predicted_counts.flatten(), dtype=tf.float32)

# Calculate Huber loss
huber_loss_value = huber_loss(val_labels_tensor, predicted_counts_tensor).numpy()

# Plotting Predicted vs Actual Counts
plt.figure(figsize=(12, 6))
plt.scatter(range(len(val_labels)), val_labels, color='blue', label='Actual Count')
plt.scatter(range(len(predicted_counts)), predicted_counts.flatten(), color='red', label='Predicted Count')
plt.title('Actual vs Predicted Count of Staples')
plt.xlabel('Image Index')
plt.ylabel('Count of Staples')
plt.legend()
plt.show()

# Save the model to a file, makes it easy to reuse model later on 
model.save('staples_dataset/my_model1.h5')

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"Huber Loss: {huber_loss_value}")
print(f"Mean Absolute Error (MAE): {mae_value}")

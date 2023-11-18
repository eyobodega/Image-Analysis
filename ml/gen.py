import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

# Define paths
dataset_dir = 'staples_dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# Create directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Updated Function to generate the image with randomly tilted and thick staples
def generate_image_with_staples(num_staples, noise_level, height=512, width=512):
    bark_color = [139, 69, 19]  # RGB for brown
    image = np.ones((height, width, 3), dtype=np.uint8) * bark_color
    
    for _ in range(num_staples):
        staple_height = np.random.randint(5, 15)  # Random staple height
        staple_width = np.random.randint(1, 5)    # Random staple width
        staple_thickness = np.random.randint(1, 3)  # Random staple thickness
        
        # Create a staple with random rotation
        staple = np.zeros((staple_height, staple_width, 3), dtype=np.uint8) + 255
        angle = np.random.uniform(0, 180)  # Random angle
        staple = rotate(staple, angle, reshape=True)
        
        # Random position without going out of bounds
        top_left_y = np.random.randint(0, height - staple.shape[0])
        top_left_x = np.random.randint(0, width - staple.shape[1])
        
        # Embed the staple into the image
        for i in range(staple.shape[0]):
            for j in range(staple.shape[1]):
                if np.all(staple[i, j] == 255):  # If the pixel is white
                    image[top_left_y + i:top_left_y + i + staple_thickness,
                          top_left_x + j:top_left_x + j + staple_thickness] = staple[i, j]
        
    # Add noise
    noise = np.random.randint(0, noise_level, (height, width, 3), dtype=np.uint8)
    noisy_image = np.clip(image + noise - noise_level // 2, 0, 255).astype(np.uint8)
    
    return noisy_image
def save_image_and_label(num_staples, noise_level, image_idx):
    # Generate the image
    image = generate_image_with_staples(num_staples, noise_level)
    
    # Define file paths
    image_path = os.path.join(images_dir, f'staples_{image_idx}.png')
    label_path = os.path.join(labels_dir, f'staples_{image_idx}.txt')
    
    # Save the image
    plt.imsave(image_path, image)
    
    # Save the label
    with open(label_path, 'w') as label_file:
        label_file.write(str(num_staples))

# Generate dataset
staple_counts = np.arange(0, 300, 1)
constant_noise_level = 50

for idx, count in enumerate(staple_counts):
    save_image_and_label(count, constant_noise_level, idx)

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve, rotate
import os
from numpy.fft import fft2, fftshift, ifft2, ifftshift


# Ensure the folders for saving images exist
os.makedirs('test_counts_anistropic', exist_ok=True)
os.makedirs('test_noise', exist_ok=True)
os.makedirs('output', exist_ok=True)

# Updated Function to generate the image with randomly tilted and thick staples
def generate_image_with_staples_anistropic(num_staples, noise_level, height=512, width=512):
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

#isostropic image generator function
def generate_image_with_staples(num_staples, noise_level, height=512, width=512, staple_height=10, staple_width=2):
    bark_color = [139, 69, 19]  # RGB for brown
    image = np.ones((height, width, 3), dtype=np.uint8) * bark_color
    for _ in range(num_staples):
        top_left_y = np.random.randint(0, height - staple_height)
        top_left_x = np.random.randint(0, width - staple_width)
        image[top_left_y:top_left_y+staple_height, top_left_x:top_left_x+staple_width] = [255, 255, 255]  # Use full white color for staples
    noise = np.random.randint(0, noise_level, (height, width, 3), dtype=np.uint8)
    noisy_image = np.clip(image + noise - noise_level // 2, 0, 255).astype(np.uint8)  # Ensure the image is uint8
    return noisy_image

# Function to detect staples in the image - anistrpoic
def save_image_with_staples_anistropic(num_staples, noise_level, dir_name, filename):
    image = generate_image_with_staples_anistropic(num_staples, noise_level)
    plt.imsave(f"{dir_name}/{filename}", image)
    return image

# Function to detect staples in the image - isotrpoic
def save_image_with_staples(num_staples, noise_level, dir_name, filename):
    image = generate_image_with_staples(num_staples, noise_level)
    plt.imsave(f"{dir_name}/{filename}", image)
    return image

# Function to apply a spectral analysis approach to count the staples in an image
def detect_staples_spectral(image, low_threshold=50, high_threshold=200):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Fourier Transform
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    
    # Create a mask with high-pass filter
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-low_threshold:crow+low_threshold, ccol-low_threshold:ccol+low_threshold] = 1
    mask = 1 - mask  # Invert mask for high-pass filtering
    
    # Apply mask and Inverse Fourier Transform
    f_shift = f_shift * mask
    f_ishift = ifftshift(f_shift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Thresholding back to binary
    _, thresh = cv2.threshold(img_back, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours which correspond to the 'staples'
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count the number of contours found
    return len(contours)

# Experiment spectral: Varying number of staples with constant noise using spectral analysis
staple_counts = np.arange(50, 500, 50)
constant_noise_level = 50
detected_counts_exp1d = []
for count in staple_counts:
    img = save_image_with_staples_anistropic(count, constant_noise_level, 'test_counts_anistropic', f'staples_anistropic{count}.png')
    img = generate_image_with_staples_anistropic(num_staples=count, noise_level=constant_noise_level)
    detected = detect_staples_spectral(img)
    detected_counts_exp1d.append(detected)

# Plotting Experiment 1d
plt.figure(figsize=(10, 6))
plt.plot(staple_counts, detected_counts_exp1d, marker='o', linestyle='-', color='blue', label='Detected Count with Spectral')
plt.plot(staple_counts, staple_counts, marker='', linestyle='--', color='red', label='True Count with Spectral')
plt.title('Experiment : Spectral with Varying Staple Counts - Anistropic')
plt.xlabel('True Staple Count')
plt.ylabel('Detected Staple Count with Spectral')
plt.legend()
plt.grid(True)
plt.savefig('output/experiment-spectral-anistropic.png')
plt.show()

# Experiment spectral: Varying number of staples with constant noise using spectral analysis
staple_counts = np.arange(50, 500, 50)
constant_noise_level = 50
detected_counts_exp1d = []
for count in staple_counts:
    img = save_image_with_staples(count, constant_noise_level, 'test_counts_anistropic', f'staples_anistropic{count}.png')
    img = generate_image_with_staples(num_staples=count, noise_level=constant_noise_level)
    detected = detect_staples_spectral(img)
    detected_counts_exp1d.append(detected)

# Plotting Experiment 1d
plt.figure(figsize=(10, 6))
plt.plot(staple_counts, detected_counts_exp1d, marker='o', linestyle='-', color='blue', label='Detected Count with Spectral')
plt.plot(staple_counts, staple_counts, marker='', linestyle='--', color='red', label='True Count with Spectral')
plt.title('Experiment : Spectral with Varying Staple Counts - Isotropic')
plt.xlabel('True Staple Count')
plt.ylabel('Detected Staple Count with Spectral')
plt.legend()
plt.grid(True)
plt.savefig('output/experiment-spectral-isostropic.png')

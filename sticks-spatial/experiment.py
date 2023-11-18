import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import convolve
import os

from numpy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, rotate


# Ensure the folders for saving images exist
os.makedirs('test_counts', exist_ok=True)
os.makedirs('test_noise', exist_ok=True)
os.makedirs('test_counts_otsu', exist_ok=True)
os.makedirs('test_anistropic', exist_ok=True)
os.makedirs('output', exist_ok=True)


# Function to generate the image with staples
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

# Function to detect staples in the image
def detect_staples(image, staple_height=10, staple_width=2, threshold=127):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((staple_height, staple_width), dtype=np.float32)
    kernel /= np.sum(kernel)
    convolved = convolve(image_gray, kernel, mode='reflect')
    _, binary_image = cv2.threshold(convolved, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

# Function to detect staples in the image with simple adaptive thresholding - otsu
def detect_staples_otsu(image, staple_height=10, staple_width=2):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((staple_height, staple_width), dtype=np.float32)
    kernel /= np.sum(kernel)
    convolved = convolve(image_gray, kernel, mode='reflect')

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(convolved, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def detect_staples_anistropic(image, staple_height=10, staple_width=2, threshold=127):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define kernels for multiple orientations
    angles = [0, 45, 90, 135]  # Example angles, can be more
    kernels = [rotate(np.ones((staple_height, staple_width), dtype=np.float32), angle, reshape=True) for angle in angles]
    for kernel in kernels:
        kernel /= np.sum(kernel)  # Normalize kernel

    # Initialize an empty image to collect all detections
    detections = np.zeros_like(image_gray, dtype=np.float32)

    # Apply each kernel and combine the results
    for kernel in kernels:
        convolved = cv2.filter2D(image_gray, -1, kernel)
        detections = np.maximum(detections, convolved)  # Take the max response

    # Threshold the combined detections
    _, binary_image = cv2.threshold(detections, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return len(contours)

def save_image_with_staples(num_staples, noise_level, dir_name, filename):
    image = generate_image_with_staples(num_staples, noise_level)
    plt.imsave(f"{dir_name}/{filename}", image)
    return image

#anistropic version for image gen
def save_image_with_staples_anistropic(num_staples, noise_level, dir_name, filename):
    image = generate_image_with_staples_anistropic(num_staples, noise_level)
    plt.imsave(f"{dir_name}/{filename}", image)
    return image


# Experiment 1: Varying number of staples with constant noise
staple_counts = np.arange(0, 1000, 50)
detected_counts_exp1 = []
constant_noise_level = 50
for count in staple_counts:
    img = save_image_with_staples_anistropic(count, constant_noise_level, 'test_counts', f'staples_{count}.png')

    img = generate_image_with_staples(num_staples=count, noise_level=constant_noise_level)
    detected = detect_staples(img)
    detected_counts_exp1.append(detected)

# Plotting Experiment 1
plt.figure(figsize=(10, 6))
plt.plot(staple_counts, detected_counts_exp1, marker='o', linestyle='-', color='blue', label='Detected Count')
plt.plot(staple_counts, staple_counts, marker='', linestyle='--', color='red', label='True Count')
plt.title('Experiment 1: Detection Accuracy with Varying Staple Counts')
plt.xlabel('True Staple Count')
plt.ylabel('Detected Staple Count')
plt.legend()
plt.grid(True)
plt.savefig('output/experiment1.png')

# Experiment 1a: Varying number of staples with otsu
staple_counts = np.arange(50, 1000, 50)
detected_counts_exp1_otsu = []
constant_noise_level = 50
for count in staple_counts:
    img = save_image_with_staples(count, constant_noise_level, 'test_counts_otsu', f'staples_otsu_{count}.png')
    img = generate_image_with_staples(num_staples=count, noise_level=constant_noise_level)
    detected = detect_staples_otsu(img)
    detected_counts_exp1_otsu.append(detected)

# Plotting Experiment 1
plt.figure(figsize=(10, 6))
plt.plot(staple_counts, detected_counts_exp1_otsu, marker='o', linestyle='-', color='blue', label='Detected Count')
plt.plot(staple_counts, staple_counts, marker='', linestyle='--', color='red', label='True Count')
plt.title('Experiment 1: Detection Accuracy with Varying Staple Counts')
plt.xlabel('True Staple Count')
plt.ylabel('Detected Staple Count')
plt.legend()
plt.grid(True)
plt.savefig('output/experiment1_otsu.png')




# Experiment 2: Varying noise levels with constant staple count
noise_levels = np.arange(10, 200, 10)
detected_counts_exp2 = []
constant_staple_count = 200
for noise in noise_levels:
    img = save_image_with_staples(constant_staple_count, noise, 'test_noise', f'noise_{noise}.png')
    img = generate_image_with_staples(num_staples=constant_staple_count, noise_level=noise)
    detected = detect_staples(img)
    detected_counts_exp2.append(detected)

# Plotting Experiment 2
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, detected_counts_exp2, marker='o', linestyle='-', color='green', label='Detected Count')
plt.axhline(y=constant_staple_count, color='red', linestyle='--', label='True Count')
plt.title('Experiment 2: Detection Accuracy with Varying Noise Levels')
plt.xlabel('Noise Level')
plt.ylabel('Detected Staple Count')
plt.legend()
plt.grid(True)
plt.savefig('output/experiment2.png')

# Experiment 2: Varying noise levels with constant staple count
noise_levels = np.arange(10, 200, 10)
detected_counts_exp2_otsu = []
constant_staple_count = 200
for noise in noise_levels:
    img_otsu = save_image_with_staples(constant_staple_count, noise, 'test_noise', f'noise_{noise}.png')
    img_otsu = generate_image_with_staples(num_staples=constant_staple_count, noise_level=noise)
    detected_otsu = detect_staples(img_otsu)
    detected_counts_exp2_otsu.append(detected_otsu)

# Plotting Experiment 2
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, detected_counts_exp2, marker='o', linestyle='-', color='green', label='Detected Count')
plt.axhline(y=constant_staple_count, color='red', linestyle='--', label='True Count')
plt.title('Experiment 2: Detection Accuracy with Varying Noise Levels')
plt.xlabel('Noise Level')
plt.ylabel('Detected Staple Count')
plt.legend()
plt.grid(True)
plt.savefig('output/experiment2-otsu.png')

# Plotting both on the same graph for comparison
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, detected_counts_exp2, marker='o', linestyle='-', color='green', label='Detected Count without Otsu')
plt.plot(noise_levels, detected_counts_exp2_otsu, marker='o', linestyle='-', color='blue', label='Detected Count with Otsu')
plt.axhline(y=constant_staple_count, color='red', linestyle='--', label='True Count')
plt.title('Detection Accuracy with Varying Noise Levels: Otsu vs. No Otsu')
plt.xlabel('Noise Level')
plt.ylabel('Detected Staple Count')
plt.legend()
plt.grid(True)
plt.savefig('output/experiment2-comparison.png')

#anistropic kernel case 
# Experiment 1 anistropic new kernel : Varying number of staples with constant noise
staple_counts = np.arange(50, 500, 50)
detected_counts_exp1_anistropic_new_kernel = []
constant_noise_level = 50
for count in staple_counts:
    img = save_image_with_staples_anistropic(count, constant_noise_level, 'test_anistropic', f'staples_anistropic{count}.png')

    img = generate_image_with_staples_anistropic(num_staples=count, noise_level=constant_noise_level)
    detected = detect_staples_anistropic(img)
    detected_counts_exp1_anistropic_new_kernel.append(detected)

# Plotting Experiment 1
plt.figure(figsize=(10, 6))
plt.plot(staple_counts, detected_counts_exp1_anistropic_new_kernel, marker='o', linestyle='-', color='blue', label='Detected Count')
plt.plot(staple_counts, staple_counts, marker='', linestyle='--', color='red', label='True Count')
plt.title('Experiment 1 Anistropic with new kernel : Detection Accuracy with Varying Staple Counts')
plt.xlabel('True Staple Count')
plt.ylabel('Detected Staple Count')
plt.legend()
plt.grid(True)
plt.savefig('output/experiment1_anistropic_new_kernel.png')

#experiment 1 anistropic but still using old kernel that is istropic 

staple_counts = np.arange(50, 500, 50)
detected_counts_exp1_anistropic_old_kernel = []
constant_noise_level = 50
for count in staple_counts:
    img = save_image_with_staples_anistropic(count, constant_noise_level, 'test_anistropic', f'staples_anistropic{count}.png')

    img = generate_image_with_staples_anistropic(num_staples=count, noise_level=constant_noise_level)
    detected = detect_staples(img)
    detected_counts_exp1_anistropic_old_kernel.append(detected)

# Plotting Experiment 1
plt.figure(figsize=(10, 6))
plt.plot(staple_counts, detected_counts_exp1_anistropic_old_kernel, marker='o', linestyle='-', color='blue', label='Detected Count')
plt.plot(staple_counts, staple_counts, marker='', linestyle='--', color='red', label='True Count')
plt.title('Experiment 1 Anistropic with istropic kernel : Detection Accuracy with Varying Staple Counts')
plt.xlabel('True Staple Count')
plt.ylabel('Detected Staple Count')
plt.legend()
plt.grid(True)
plt.savefig('output/experiment1_anistropic_old_kernel.png')

# # Function to apply a spectral analysis approach to count the staples in an image
# def detect_staples_spectral(image, low_threshold=50, high_threshold=200):
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Fourier Transform
#     f_transform = fft2(gray)
#     f_shift = fftshift(f_transform)
    
#     # Create a mask with high-pass filter
#     rows, cols = gray.shape
#     crow, ccol = rows // 2, cols // 2
#     mask = np.zeros((rows, cols), np.uint8)
#     mask[crow-low_threshold:crow+low_threshold, ccol-low_threshold:ccol+low_threshold] = 1
#     mask = 1 - mask  # Invert mask for high-pass filtering
    
#     # Apply mask and Inverse Fourier Transform
#     f_shift = f_shift * mask
#     f_ishift = ifftshift(f_shift)
#     img_back = ifft2(f_ishift)
#     img_back = np.abs(img_back)
    
#     # Thresholding back to binary
#     _, thresh = cv2.threshold(img_back, 127, 255, cv2.THRESH_BINARY)
    
#     # Find contours which correspond to the 'staples'
#     contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Count the number of contours found
#     return len(contours)

# # Experiment 1d: Varying number of staples with constant noise using spectral analysis
# detected_counts_exp1d = []
# for count in staple_counts:
#     img = generate_image_with_staples(num_staples=count, noise_level=constant_noise_level)
#     detected = detect_staples_spectral(img)
#     detected_counts_exp1d.append(detected)

# # Plotting Experiment 1d
# plt.figure(figsize=(10, 6))
# plt.plot(staple_counts, detected_counts_exp1d, marker='o', linestyle='-', color='blue', label='Detected Count with Spectral')
# plt.plot(staple_counts, staple_counts, marker='', linestyle='--', color='red', label='True Count with Spectral')
# plt.title('Experiment 1d: Detection Accuracy with Varying Staple Counts using Spectral with staples in one direction')
# plt.xlabel('True Staple Count')
# plt.ylabel('Detected Staple Count with Spectral')
# plt.legend()
# plt.grid(True)
# plt.savefig('output/experiment1d.png')
# plt.show()
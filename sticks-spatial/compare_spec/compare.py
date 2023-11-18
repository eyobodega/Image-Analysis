import numpy as np
import cv2
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt

def apply_fourier_transform(image):
    f_transform = fft2(image)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
    return f_shift, magnitude_spectrum

def apply_high_pass_filter(f_shift, size):
    rows, cols = f_shift.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-size:crow+size, ccol-size:ccol+size] = 0
    f_shift_high_pass = f_shift * mask
    return f_shift_high_pass

def inverse_fourier_transform(f_shift_high_pass):
    f_ishift = ifftshift(f_shift_high_pass)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def count_staples(img_back):
    _, thresh = cv2.threshold(img_back, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours), thresh

def visualize_results(images, titles, output_filename):
    plt.figure(figsize=(15, 5))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1), plt.imshow(image, cmap='gray'), plt.title(title)
        plt.axis('off')
    plt.suptitle(output_filename)  # Set the figure's super title to be the output filename
    plt.savefig(output_filename)  # Save the figure with the specified filename
    plt.close()  # Close the figure after saving to free memory

def experiment(image_path, output_filename):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Fourier transform and magnitude spectrum
    f_shift, magnitude_spectrum = apply_fourier_transform(image)

    # Apply high-pass filter
    f_shift_high_pass = apply_high_pass_filter(f_shift, size=30)  # Example size, adjust as needed

    # Inverse Fourier Transform to spatial domain
    img_back = inverse_fourier_transform(f_shift_high_pass)

    # Count staples and get the thresholded image
    count, thresh = count_staples(img_back)

    # Visualize the results
    visualize_results(
        [image, magnitude_spectrum, np.log(np.abs(f_shift_high_pass)), thresh],
        ['Original Image', 'Magnitude Spectrum', 'High-Pass Filtered Spectrum', 'Detected Staples'],
        output_filename  # Pass the output filename to the function
    )
    
    print(f"Detected staple count: {count}")

# Perform the experiment on both types of images
experiment('staples_200_nt.png', 'staples_200_nt_output.png')
experiment('staples_200.png', 'staples_200_output.png')

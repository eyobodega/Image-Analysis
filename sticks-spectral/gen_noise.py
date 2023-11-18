import numpy as np
import matplotlib.pyplot as plt

# Create an image with tree bark color
height, width = 512, 512
bark_color = [139, 69, 19]  # RGB for brown
image = np.ones((height, width, 3), dtype=np.uint8) * bark_color

# Add staples
num_staples = 200
staple_height, staple_width = 10,2  
for _ in range(num_staples):
    top_left_y = np.random.randint(0, height - staple_height)
    top_left_x = np.random.randint(0, width - staple_width)
    image[top_left_y:top_left_y+staple_height, top_left_x:top_left_x+staple_width] = [200, 200, 200]

# Save the image without noise
plt.imshow(image)
plt.axis('off')
plt.savefig("tree_without_noise.png", dpi=1000, bbox_inches='tight', pad_inches=0)

# # Add some noise
# noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
# noisy_image = np.clip(image + noise - 25, 0, 255)


# Add increased noise
# Adjust these values to increase the noise level
noise_low = 0  # Lower bound of the noise
noise_high = 100  # Upper bound of the noise (increased from 50 to 100)
noise_reduction = 50  # This value is subtracted to create both positive and negative noise

noise = np.random.randint(noise_low, noise_high, (height, width, 3), dtype=np.uint8)
noisy_image = np.clip(image + noise - noise_reduction, 0, 255)

# Save the image with noise
plt.imshow(noisy_image)
plt.axis('off')
plt.savefig("tree_with_noise.png", dpi=1000, bbox_inches='tight', pad_inches=0)

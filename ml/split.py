import os
from sklearn.model_selection import train_test_split
import shutil

# Paths to your dataset directories
images_dir = 'staples_dataset/images'
labels_dir = 'staples_dataset/labels'

# Get a list of image file names
image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

# Ensure the files are sorted by their indices
image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# Splitting the dataset into training and validation sets
# Here we're not using the actual image data, just the file names
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# Create directories for the train/validation split if they don't exist
train_images_dir = os.path.join(images_dir, 'train')
val_images_dir = os.path.join(images_dir, 'val')
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)

# Move files into the respective directories
for f in train_files:
    os.rename(os.path.join(images_dir, f), os.path.join(train_images_dir, f))

for f in val_files:
    os.rename(os.path.join(images_dir, f), os.path.join(val_images_dir, f))

#we know move the labels
# Define the source and target directories for train labels
train_images_dir = 'staples_dataset/images/train'
train_labels_dir = 'staples_dataset/train_labels'
reference_labels_dir = 'staples_dataset/labels'

# Create the train labels target directory if it doesn't exist
os.makedirs(train_labels_dir, exist_ok=True)

# Loop through files in the train images directory
for filename in os.listdir(train_images_dir):
    # Construct the base filename by removing the .png extension
    base_filename = os.path.splitext(filename)[0]

    # Construct the source label file path with .txt extension
    source_label_path = os.path.join(reference_labels_dir, base_filename + '.txt')

    # Check if the corresponding .txt file exists in the reference labels directory
    if os.path.exists(source_label_path):
        # Move the .txt file from the reference labels directory to the train labels directory
        shutil.move(source_label_path, os.path.join(train_labels_dir, base_filename + '.txt'))

# Define the source and target directories for validation labels
val_images_dir = 'staples_dataset/images/val'
val_labels_dir = 'staples_dataset/val_labels'

# Create the validation labels target directory if it doesn't exist
os.makedirs(val_labels_dir, exist_ok=True)

# Loop through files in the validation images directory
for filename in os.listdir(val_images_dir):
    # Construct the base filename by removing the .png extension
    base_filename = os.path.splitext(filename)[0]

    # Construct the source label file path with .txt extension
    source_label_path = os.path.join(reference_labels_dir, base_filename + '.txt')

    # Check if the corresponding .txt file exists in the reference labels directory
    if os.path.exists(source_label_path):
        # Move the .txt file from the reference labels directory to the validation labels directory
        shutil.move(source_label_path, os.path.join(val_labels_dir, base_filename + '.txt'))

print(f"Training set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")

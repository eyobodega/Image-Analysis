from PIL import Image
import os

# Assuming all your images are in 'staples_dataset/images/train'
image_path = os.path.join('staples_dataset', 'images', 'train', 'staples_1.png')
with Image.open(image_path) as img:
    image_shape = img.size  # (width, height)
    image_shape_with_channels = image_shape + (3,)  # Assuming RGB images

print(f"Image shape: {image_shape_with_channels}")

labels = []
labels_dir = 'staples_dataset/val_labels'
for label_file in sorted(os.listdir(labels_dir)):
    with open(os.path.join(labels_dir, label_file), 'r') as file:
        labels.append(int(file.read().strip()))

print(f"Labels shape: ({len(labels)},)")

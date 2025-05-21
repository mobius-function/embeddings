import os
import random
import shutil
from pathlib import Path

# Source path (wide face extracted data)
source_path = "data/faces/output"

# Target paths
train_path = "data/train"
val_path = "data/val"
test_path = "data/test"

# Create directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Get all image files
image_files = []
for ext in ['.jpg', '.jpeg', '.png']:
    image_files.extend(list(Path(source_path).glob(f'**/*{ext}')))

# Ensure we found images
if not image_files:
    print(f"No images found in {source_path}! Check file extensions and directory.")
    exit(1)

print(f"Found {len(image_files)} images")

# Shuffle the files
random.shuffle(image_files)

# Calculate split points (70% train, 15% val, 15% test)
train_split = int(len(image_files) * 0.7)
val_split = int(len(image_files) * 0.85)  # 70% + 15% = 85%

# Copy files to train directory
for img_path in image_files[:train_split]:
    shutil.copy(img_path, os.path.join(train_path, img_path.name))

# Copy files to validation directory
for img_path in image_files[train_split:val_split]:
    shutil.copy(img_path, os.path.join(val_path, img_path.name))

# Copy files to test directory
for img_path in image_files[val_split:]:
    shutil.copy(img_path, os.path.join(test_path, img_path.name))

print(f"Split {len(image_files)} images into:")
print(f" - Training: {train_split} images")
print(f" - Validation: {val_split - train_split} images")
print(f" - Test: {len(image_files) - val_split} images")
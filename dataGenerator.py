import cv2
import numpy as np
import os
import random
from glob import glob
from tqdm import tqdm
import albumentations as A  # Powerful augmentation library



# Define augmentation pipeline
augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.7),  # Adjust brightness & contrast
    A.GaussianBlur(blur_limit=2, p=0.5),  # Slight blur
    A.GaussNoise((0.02, 0.04), p=0.2),  # Add Gaussian noise
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.025, 
        rotate_limit=5.0,
        border_mode=cv2.BORDER_REFLECT,
        p=0.5  # Apply to 50% of images
    )
    # A.Perspective(scale=(0.02, 0.07), p=0.4),  # Apply minor perspective warping
    # A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), p=0.5),  # Random crops
    # A.Downscale(scale_min=0.7, scale_max=0.9, p=0.3),  # Simulate lower-quality capture
])

# Input and output directories
INPUT_FOLDER = "0_Raw_Images/"
OUTPUT_FOLDER = "0_Augmented_Images/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load original PCB images
image_paths = glob(os.path.join(INPUT_FOLDER, "*.jpg"))  # Modify based on image type
generated_images = 0
num_augmentations = 500  # Number of augmented images per original

# Generate augmented images
for img_path in tqdm(image_paths, desc="Generating Data"):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (Albumentations uses RGB)

    for i in range(num_augmentations):
        augmented = augmentations(image=image)["image"]
        save_path = os.path.join(OUTPUT_FOLDER, f"pcb_{generated_images}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2GRAY))  # Convert back to BGR for saving
        generated_images += 1

print(f"Generated {generated_images} training images in '{OUTPUT_FOLDER}'")

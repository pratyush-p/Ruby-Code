import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras.src.utils import image_dataset_from_directory
from keras.src.saving import load_model
import matplotlib.pyplot as plt
import cv2
import os

# --- Load the Autoencoder Model ---
MODEL_PATH = "pcb_autoencoder.keras"  # Path to your trained model
autoencoder = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

image_path = "0_Raw_Images/test0.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # For grayscale images
image = cv2.resize(image, (144, 224))  # Resize to match model's input size
image = image / 255.0  # Normalize to [0, 1]
# image = np.expand_dims(image, axis=0)  # Add batch dimension
image = np.expand_dims(image, axis=-1)  # Add channel dimension (if grayscale)


output = autoencoder.predict(image)
print(image.shape)
print(output.shape)


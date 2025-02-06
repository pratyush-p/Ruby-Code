import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.src.models import Model
from keras.src.optimizers import Adam
from keras.src.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import cv2
import os

# Define the input shape of your images
IMG_HEIGHT = 144  # Adjust as per your dataset
IMG_WIDTH = 224
CHANNELS = 1  # 1 for grayscale
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

DATASET_PATH = "0_Augmented_Images"

# Load images into a dataset
train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    labels=None,  # Unsupervised, so no labels
    color_mode="grayscale",
    image_size=(IMG_HEIGHT, IMG_WIDTH),  # Resize images
    batch_size=32,  # Batch size
    shuffle=True  # Shuffle the dataset
)

# Normalize pixel values to [0, 1]
def normalize_image(image):
    return image / 255.0

# Convert the dataset into a NumPy array (if required)
X = tf.concat([x for x in train_dataset.map(normalize_image)], axis=0).numpy()

print("yahoo")
print(X.shape)

# --- Autoencoder Architecture ---
def build_autoencoder(input_shape):
    # Encoder
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)  # Latent space

    # Decoder
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(CHANNELS, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

# Build the autoencoder
autoencoder = build_autoencoder(INPUT_SHAPE)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.summary()

history = autoencoder.fit(
    X,
    X,
    batch_size=32,
    epochs=10,
)

# # --- Save the Trained Model ---
autoencoder.save("pcb_autoencoder.keras")

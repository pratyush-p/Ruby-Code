import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.src.models import Model
from keras.src.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import os

# Load and preprocess training images
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        images.append(img / 255.0)  # Normalize
    return np.array(images)

# Assume "Raw Images" contains only unaltered PCBs
normal_pcbs = load_images("Raw Images").reshape(-1, 128, 128, 1)

# Define Autoencoder architecture
input_img = Input(shape=(128, 128, 1))

# Encoder
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
encoded = MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = Conv2D(16, (3, 3), activation="relu", padding="same")(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train the model
autoencoder

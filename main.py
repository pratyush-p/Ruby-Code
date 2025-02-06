import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# Constants for display resolution
HEIGHT = 480
X = int(HEIGHT * (4.0 / 3.0))
Y = HEIGHT

# Camera labels
camera_labels = ["Angle0", "Angle1", "Angle2", "Angle3"]

for label in camera_labels:
    os.makedirs(label + "_Raw_Images", exist_ok=True)

# Function to capture frames from multiple cameras
def capture_frames(cams):
    frames = []
    for i, cam in enumerate(cams):
        if cam is not None:
            ret, frame = cam.read()
            if ret:
                frame = cv2.resize(frame, (X, Y))  # Resize for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                frames.append(frame)
            else:
                frames.append(np.zeros((Y, X, 3), dtype=np.uint8))  # Placeholder
        else:
            frames.append(np.zeros((Y, X, 3), dtype=np.uint8))  # Placeholder
    return frames

# Function to save images to "Raw Images" directory
def save_images(frames):
    timestamp = datetime.now().strftime("%d_%H%M%S")  # Unique timestamp for filenames
    for i, frame in enumerate(frames):
        filename = os.path.join("Angle" + str(i) + "_Raw_Images", f"{timestamp}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert back to BGR before saving
    st.success("Images saved to 'Raw Images' folder.")

# Streamlit App
st.set_page_config(layout="wide")  # Use wide layout to maximize space
st.title("Multi-Camera Live Stream with Image Capture")

# Initialize Streamlit session state
if "running" not in st.session_state:
    st.session_state.running = False

# Start/Stop buttons
if st.button("Start Stream"):
    st.session_state.running = True

if st.button("Stop Stream"):
    st.session_state.running = False

st.text("Running" if st.session_state.running else "Not Running")

# Camera initialization
cams = [cv2.VideoCapture(i) for i in range(len(camera_labels))]
for i, cam in enumerate(cams):
    if not cam.isOpened():
        st.warning(f"Error: Could not open camera {i}")
        cams[i] = None  # Mark as None if it fails

# Create placeholders for the camera feeds
col1, col2 = st.columns([1, 1])
feed1 = col1.empty()
feed2 = col2.empty()
feed3 = col1.empty()
feed4 = col2.empty()

# Button to capture and save images
if st.button("Capture Images"):
    frames = capture_frames(cams)
    save_images(frames)

# Camera feed loop
while st.session_state.running:
    frames = capture_frames(cams)

    # Update Streamlit placeholders with the frames
    feed1.image(frames[0], channels="RGB", use_container_width=True)
    feed2.image(frames[1], channels="RGB", use_container_width=True)
    feed3.image(frames[2], channels="RGB", use_container_width=True)
    feed4.image(frames[3], channels="RGB", use_container_width=True)

    # Stop the loop temporarily to prevent freezing
    if not st.session_state.running:
        break

# Release all cameras when stopping
for cam in cams:
    if cam is not None:
        cam.release()

st.success("Camera feed stopped.")

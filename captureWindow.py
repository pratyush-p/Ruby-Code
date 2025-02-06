import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# Constants for display resolution
HEIGHT = 480
X = int(HEIGHT * (4.0 / 3.0))
Y = HEIGHT

CASE_SLICES = {
        "Top": lambda: [(70, 100), (570, 470)], # 500 x 370
        "Left": lambda: [(40, 130), (460, 470)], # 420 x 340
        "Right": lambda: [(180, 130), (600, 470)], # 420 x 340
        "Front": lambda: [(80, 180), (520, 470)], # 440 x 290
}

CAMERA_COUNT = 3

for i in range(CAMERA_COUNT):
    os.makedirs(str(i) + "_Raw_Images", exist_ok=True)

st.set_page_config(layout="wide")  # Use wide layout for better visibility
st.title("Multi-Camera Capture Grid")

if "frames" not in st.session_state:
    st.session_state.frames = [np.zeros((Y, X, 3), dtype=np.uint8) for _ in range(CAMERA_COUNT)]
    
if "cams" not in st.session_state:
    st.session_state.cams = [None] * CAMERA_COUNT

def get_camera(index):
    if st.session_state.cams[index] is None:
        camera = cv2.VideoCapture(index)  # Open the camera
        if camera.isOpened():
            st.session_state.cams[index] = camera
            return camera
        else:
            st.warning(f"Camera {index} could not be opened.")
            return None
    return st.session_state.cams[index]

# Function to capture a frame from a specific camera
def capture_frame(cam_index):
    camera: cv2.VideoCapture = get_camera(cam_index)
    if camera.isOpened():
        ret, frame = camera.read()
        if ret:
            frame = cv2.resize(frame, (X, Y))  # Resize for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            return frame
    return np.zeros((Y, X, 3), dtype=np.uint8)  # Placeholder if capture fails

# Function to save an image
def save_image(frame, cam_index):
    timestamp = datetime.now().strftime("%d_%H%M%S")  # Unique timestamp
    filename = os.path.join(f"{cam_index}_Raw_Images", f"{timestamp}.jpg")
    cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert back to BGR before saving

def drawBoxOnCapture(angle: str, frame: np.ndarray, index: int) -> np.ndarray:
    slice = CASE_SLICES.get(angle, lambda: [(0, 0), (10, 10)])()
    return cv2.rectangle(frame, slice[0], slice[1], (255, 0, 0), 3)

def cameraModule(index):
    angle = st.selectbox(
        "View Angle - " + str(index),
        ["Top", "Left", "Right", "Front"]
    )
    
    if st.button("Save Image " + str(index)):
        slice = CASE_SLICES.get(angle, lambda: [(0, 0), (10, 10)])()
        x1, y1 = slice[0]
        x2, y2 = slice[1]
        save_image(capture_frame(index)[y1:y2, x1:x2], index)
    
    if angle: 
        st.session_state.frames[index] = capture_frame(index)
        drawBoxOnCapture(angle, st.session_state.frames[index], index)

    st.image(st.session_state.frames[index], channels="RGB", use_container_width=True)
    
# Create 2x2 grid layout
col1, col2 = st.columns(2)
with col1:
    cameraModule(0)
    cameraModule(2)

with col2:
    cameraModule(1)

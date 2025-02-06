import cv2

def check_cameras(num_cameras=4):
    for i in range(num_cameras):
        print(f"Testing Camera {i}...")

        # Try to open the camera
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"Camera {i} could not be opened.")
            continue

        # Try to capture a frame
        ret, frame = cap.read()
        if not ret:
            print(f"Camera {i} could not capture a frame.")
            cap.release()
            continue

        # Display the frame
        cv2.imshow(f"Camera {i}", frame)
        print(f"Camera {i} is working. Press any key to close the preview.")
        cv2.waitKey(0)  # Wait for a key press to close the window

        # Release the camera
        cap.release()
        cv2.destroyAllWindows()

# Run the test
check_cameras(num_cameras=4)

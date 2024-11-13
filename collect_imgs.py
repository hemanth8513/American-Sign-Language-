import os
import cv2
import time

# Directory setup
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 27
dataset_size = 100

# Initialize camera
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if 0 doesn't work

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

print("Camera successfully opened. Waiting to initialize...")
time.sleep(1)  # Allow camera time to initialize

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Display "Ready" message to start capture
    while True:
        ret, frame = cap.read()
        
        # Check if frame was captured successfully
        if not ret:
            print("Failed to grab frame. Ensure the camera is properly connected.")
            break
        
        # Show the ready message on the frame
        cv2.putText(frame, 'Press "q" to start, "c" to capture, "q" again to stop', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Wait for 'q' key to enter capture mode
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Starting capture mode...")
            break

    # Capture images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        
        # Confirm successful frame capture
        if not ret:
            print("Failed to grab frame during capture.")
            break

        cv2.imshow('frame', frame)
        
        # Press 'c' to capture and save each frame manually
        key = cv2.waitKey(25) & 0xFF
        if key == ord('c'):
            img_path = os.path.join(class_dir, f'{counter}.jpg')
            cv2.imwrite(img_path, frame)
            print(f"Captured image {counter + 1} for class {j}")
            counter += 1
        elif key == ord('q'):  # Press 'q' to stop capturing for this class
            print("Stopping capture for this class.")
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

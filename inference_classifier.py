import pickle
import cv2
import mediapipe as mp
import numpy as np


# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the camera with potential compatibility settings for macOS
# Try different indices if 0 doesn't work. CAP_AVFOUNDATION helps on macOS.
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open video. Check camera permissions and index.")
    exit()

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels for hand gestures
labels_dict = {
    0: 'NULL', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 
    19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Check if frame was captured successfully
    if not ret or frame is None:
        print("Error: Could not read frame from camera. Please check the camera connection or index.")
        break

    H, W, _ = frame.shape

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Process hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Image to draw on
                hand_landmarks,  # Model output
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Calculate bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Predict the gesture
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()

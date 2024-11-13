# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# # Load the model
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # Initialize the camera
# cap = cv2.VideoCapture(0)

# # Initialize MediaPipe hands and drawing utilities
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# # Labels for hand gestures
# labels_dict = {
#     0: 'NULL', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
#     10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R',
#     19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'
# }

# # Variables for recognized text and word formation
# recognized_text = ""  # To store the full recognized text (words)
# current_word = []  # List to store letters of the current word
# last_predicted = None  # Store last predicted letter
# last_prediction_time = None  # Time when last letter was predicted
# gesture_start_time = None  # Time when the gesture was first detected
# hold_time = 3  # Time (in seconds) required to hold the gesture to register (now 3 seconds)
# gesture_change_threshold = 1  # Time threshold to consider a gesture change (in seconds)
# gesture_in_progress = False  # Flag to indicate if a gesture is in progress

# while True:
#     data_aux = []  # Reset features list
#     x_ = []
#     y_ = []

#     # Read a frame from the camera
#     ret, frame = cap.read()

#     # Check if frame was captured successfully
#     if not ret or frame is None:
#         print("Error: Could not read frame from camera. Please check the camera connection or index.")
#         break

#     H, W, _ = frame.shape

#     # Convert the frame to RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     # Process hand landmarks if detected
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # Image to draw on
#                 hand_landmarks,  # Model output
#                 mp_hands.HAND_CONNECTIONS,  # Hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )

#             # Collect landmark coordinates
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 x_.append(x)
#                 y_.append(y)

#             # Normalize the coordinates and store in data_aux
#             min_x, min_y = min(x_), min(y_)
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append((x - min_x))  # Normalize x coordinates
#                 data_aux.append((y - min_y))  # Normalize y coordinates

#             # Ensure data_aux has exactly 42 features (21 landmarks * 2 coordinates)
#             print(len(data_aux))  # Should print 42 for each gesture

#             # Predict the gesture
#             if len(data_aux) == 42:  # Only predict if data_aux has the correct number of features
#                 prediction = model.predict([np.asarray(data_aux)])
#                 predicted_character = labels_dict[int(prediction[0])]

#                 # Only process if a valid gesture is detected (ignore NULL)
#                 if predicted_character != 'NULL':
#                     current_time = time.time()

#                     # Check if the gesture is still the same as the last predicted one
#                     if predicted_character == last_predicted:
#                         # If the gesture has been held for more than 'hold_time' seconds, confirm the prediction
#                         if current_time - gesture_start_time >= hold_time:
#                             # Add letter to the current word
#                             current_word.append(predicted_character)
#                             last_predicted = None  # Reset last predicted gesture after confirming
#                             print(f"Letter {predicted_character} added.")
#                     else:
#                         # If the gesture changes, reset the timer and set the new gesture start time
#                         last_predicted = predicted_character
#                         gesture_start_time = current_time
#                         print(f"Started holding gesture: {predicted_character}")

#                 # Draw bounding box and predicted character
#                 x1 = int(min(x_) * W) - 10
#                 y1 = int(min(y_) * H) - 10
#                 x2 = int(max(x_) * W) + 10
#                 y2 = int(max(y_) * H) + 10
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                 cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

#     # If no gesture is in progress and timeout is reached, finalize the current word
#     if len(current_word) > 0 and (time.time() - gesture_start_time) > gesture_change_threshold:
#         recognized_text += "".join(current_word) + " "  # Add word to recognized text
#         current_word = []  # Clear current word

#     # Display the recognized text on the screen
#     cv2.putText(frame, "Text: " + recognized_text, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     # Display the frame
#     cv2.imshow('frame', frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()




import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Labels for hand gestures
labels_dict = {
    0: 'NULL', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
    10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R',
    19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'
}

# Variables for recognized text and word formation
recognized_text = ""  # To store the full recognized text (words)
current_word = []  # List to store letters of the current word
last_predicted = None  # Store last predicted letter
last_prediction_time = None  # Time when last letter was predicted
gesture_start_time = None  # Time when the gesture was first detected
hold_time = 3  # Time (in seconds) required to hold the gesture to register (now 3 seconds)
gesture_change_threshold = 1  # Time threshold to consider a gesture change (in seconds)
gesture_in_progress = False  # Flag to indicate if a gesture is in progress
inactivity_timeout = 10  # Timeout (in seconds) after which text will be cleared if no gesture is detected
last_gesture_time = time.time()  # Time when the last gesture was detected

while True:
    data_aux = []  # Reset features list
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

            # Normalize the coordinates and store in data_aux
            min_x, min_y = min(x_), min(y_)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append((x - min_x))  # Normalize x coordinates
                data_aux.append((y - min_y))  # Normalize y coordinates

            # Ensure data_aux has exactly 42 features (21 landmarks * 2 coordinates)
            print(len(data_aux))  # Should print 42 for each gesture

            # Predict the gesture
            if len(data_aux) == 42:  # Only predict if data_aux has the correct number of features
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Only process if a valid gesture is detected (ignore NULL)
                if predicted_character != 'NULL':
                    current_time = time.time()

                    # Check if the gesture is still the same as the last predicted one
                    if predicted_character == last_predicted:
                        # If the gesture has been held for more than 'hold_time' seconds, confirm the prediction
                        if current_time - gesture_start_time >= hold_time:
                            # Add letter to the current word
                            current_word.append(predicted_character)
                            last_predicted = None  # Reset last predicted gesture after confirming
                            print(f"Letter {predicted_character} added.")
                    else:
                        # If the gesture changes, reset the timer and set the new gesture start time
                        last_predicted = predicted_character
                        gesture_start_time = current_time
                        print(f"Started holding gesture: {predicted_character}")

                # Draw bounding box and predicted character
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                # Reset inactivity timer as gesture was detected
                last_gesture_time = time.time()

    # If no gesture is detected and inactivity timeout has passed, reset recognized_text
    if time.time() - last_gesture_time > inactivity_timeout:
        recognized_text = ""  # Clear the recognized text if no gesture is detected for the timeout duration
        current_word = []  # Clear the current word

    # If no gesture is in progress and timeout is reached, finalize the current word
    if len(current_word) > 0 and (time.time() - gesture_start_time) > gesture_change_threshold:
        recognized_text += "".join(current_word) + " "  # Add word to recognized text
        current_word = []  # Clear current word

    # Display the recognized text on the screen
    cv2.putText(frame, "Text: " + recognized_text, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

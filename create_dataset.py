import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Get the list of items in DATA_DIR
dir_list = os.listdir(DATA_DIR)
print(f"Found items in data directory: {dir_list}")

# Process directories and images
for dir_ in dir_list:
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip if it's a file (not a directory) or hidden file
    if not os.path.isdir(dir_path) or dir_.startswith('.'):
        print(f"Skipping non-directory or hidden file: {dir_}")
        continue

    print(f"Processing directory: {dir_}")

    for img_path in os.listdir(dir_path):
        img_path_full = os.path.join(dir_path, img_path)

        # Skip hidden files and non-file items
        if img_path.startswith('.') or not os.path.isfile(img_path_full):
            print(f"Skipping non-file or hidden file: {img_path}")
            continue

        data_aux = []
        x_ = []
        y_ = []

        # Read the image and process it
        img = cv2.imread(img_path_full)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                min_x, min_y = min(x_), min(y_)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

            data.append(data_aux)
            labels.append(dir_)

# Save the dataset into a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Dataset saved to data.pickle")

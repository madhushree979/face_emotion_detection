import os
import cv2
from keras.models import model_from_json
import numpy as np

# Paths
json_file_path = "facialemotionmodel.json"
weights_file_path = "facialemotionmodel.h5"

# Verify JSON file
if not os.path.exists(json_file_path) or not os.path.getsize(json_file_path):
    raise FileNotFoundError(f"JSON file is missing or empty: {json_file_path}")

# Verify weights file
if not os.path.exists(weights_file_path):
    raise FileNotFoundError(f"Weights file is missing: {weights_file_path}")

# Load the model architecture
with open(json_file_path, "r") as json_file:
    model_json = json_file.read()

# Initialize model
model = model_from_json(model_json)

# Load weights
model.load_weights(weights_file_path)

# Load Haar cascade
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Webcam setup
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Real-time detection loop
try:
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (48, 48))
            features = extract_features(face_resized)
            prediction = model.predict(features)
            emotion = labels[prediction.argmax()]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    webcam.release()
    cv2.destroyAllWindows()

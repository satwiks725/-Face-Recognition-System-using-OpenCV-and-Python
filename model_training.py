import cv2
import os
import numpy as np
from PIL import Image

# Path to your dataset directory
dataset_dir = 'images'  # Each subfolder should be named after the person

# Output model file
model_file = 'training_data.yml'

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize lists to hold training data
faces = []
labels = []
label_ids = {}
current_id = 0

# Walk through the dataset directory
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            path = os.path.join(root, file)
            label = os.path.basename(root)

            # Assign a numeric ID to each label
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            # Open image and convert to grayscale
            image = Image.open(path).convert('L')  # grayscale
            image_np = np.array(image, 'uint8')

            # Detect faces
            faces_rects = face_cascade.detectMultiScale(image_np, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_rects:
                roi = image_np[y:y+h, x:x+w]
                faces.append(roi)
                labels.append(id_)

# Train recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save the model to file
recognizer.save(model_file)

# Optionally save label map to a file
with open("labels.txt", "w") as f:
    for name, id_ in label_ids.items():
        f.write(f"{id_}:{name}\n")

print(f"Training complete. Model saved as '{model_file}' and labels in 'labels.txt'")

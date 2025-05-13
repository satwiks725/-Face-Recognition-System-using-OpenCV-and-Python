import cv2
import numpy as np

# Load face recognizer and label mappings
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training_data.yml")

# Load label-ID map
label_dict = {}
with open("labels.txt", "r") as f:
    for line in f:
        id_str, name = line.strip().split(":")
        label_dict[int(id_str)] = name

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

print("[INFO] Starting face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        id_, confidence = recognizer.predict(roi_gray)

        name = label_dict.get(id_, "Unknown")
        confidence_text = f"{100 - int(confidence)}%"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence_text})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

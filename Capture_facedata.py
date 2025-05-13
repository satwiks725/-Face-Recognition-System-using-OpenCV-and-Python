import cv2
import os

# Configuration
num_samples = 50  # Number of face images to capture per person
dataset_dir = "images"  # Base directory to store images
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Input: Get person's name
person_name = input("Enter the name of the person: ").strip()

# Create person's folder if not exists
person_dir = os.path.join(dataset_dir, person_name)
os.makedirs(person_dir, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

print(f"\n[INFO] Starting face capture for '{person_name}'. Press 'q' to quit early.")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        file_path = os.path.join(person_dir, f"{person_name}_{count:02d}.jpg")
        cv2.imwrite(file_path, face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{count}/{num_samples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Face Capture', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Manual quit.")
        break

    if count >= num_samples:
        print(f"[INFO] Collected {num_samples} face samples for '{person_name}'.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

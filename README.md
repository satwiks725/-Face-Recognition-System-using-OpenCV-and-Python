
# 👤 Face Recognition System using OpenCV and Python

This project is a real-time **Face Recognition System** built using **OpenCV** and **Python 3.9**. It allows you to capture face data, train a recognition model, and then recognize faces in real-time using a webcam.

---

## 🛠️ Tech Stack
- Python 3.9  
- OpenCV (with `opencv-contrib-python`)  
- NumPy  
- Haar Cascade Classifier  
- LBPH Face Recognizer (Local Binary Patterns Histograms)

---

## 📂 Project Structure

```bash
├── images/                # Stores captured face images for each person
├── capturefacedata.py     # Step 1: Face data collection script
├── modeltraining.py       # Step 2: Model training script
├── facerecognition.py     # Step 3: Face recognition app
├── training_data.yml      # Model file generated after training
├── README.md              # Project documentation
```

---

## ✅ Steps to Run the Project

### 📸 STEP 1: Capture Face Data
Run the following script to capture face data through your webcam and save it to the `images/` folder:
```bash
python Capture_facedata.py
```
- The script will ask for a user ID or name.
- Face images will be saved automatically.
- Make sure your webcam is connected.

---

### 🧠 STEP 2: Train the Model
Train the face recognizer using the captured data:
```bash
python model_training.py
```
- This will process all images inside the `images/` folder.
- The script will create a `training_data.yml` file for use in face recognition.

---

### 🕵️ STEP 3: Recognize Faces in Real-Time
Start the face recognition app:
```bash
python Face_Recognition_with_name.py
```
- The webcam feed will display recognized faces along with their names or show "Unknown" if not trained.

---

## ⚠️ Prerequisites

Install dependencies using pip:
```bash
pip install opencv-contrib-python numpy
```

---

## 📌 Notes
- Ensure `haarcascade_frontalface_default.xml` is correctly loaded using OpenCV's built-in path.
- The model works best in good lighting and consistent head orientation.
- This implementation uses the LBPH recognizer — best suited for small datasets and local applications.

---

## 🚀 Future Improvements
- Use deep learning-based models like FaceNet or Dlib for more accuracy.
- Add a GUI using Tkinter or PyQt.
- Implement database integration for managing user profiles.



## 🤝 Contribution
Feel free to fork the repository, raise issues, or open pull requests if you have improvements or suggestions!

---

## 📎 License
This project is open-source and available under the [MIT License](LICENSE).


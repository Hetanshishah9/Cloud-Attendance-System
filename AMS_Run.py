from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
from PIL import Image
import datetime

app = Flask(__name__)

# Path constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_IMAGE_PATH = os.path.join(BASE_DIR, 'TrainingImage')
RECOGNIZER_PATH = os.path.join(BASE_DIR, 'TrainingImageLabel', 'Trainer.yml')
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
ATTENDANCE_PATH = os.path.join(BASE_DIR, 'Attendance')

# Ensure folders exist
os.makedirs(TRAINING_IMAGE_PATH, exist_ok=True)
os.makedirs(os.path.dirname(RECOGNIZER_PATH), exist_ok=True)
os.makedirs(ATTENDANCE_PATH, exist_ok=True)


def train_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(CASCADE_PATH)

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []
        for imagePath in image_paths:
            if imagePath.endswith('.jpg') or imagePath.endswith('.png'):
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')
                try:
                    id = int(os.path.split(imagePath)[-1].split(".")[1])
                except:
                    continue
                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
        return face_samples, ids

    faces, ids = get_images_and_labels(TRAINING_IMAGE_PATH)
    recognizer.train(faces, np.array(ids))
    recognizer.save(RECOGNIZER_PATH)


def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 60:
                student_id = str(id_)
                ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = os.path.join(ATTENDANCE_PATH, f'attendance_{ts}.csv')
                with open(filename, 'a') as f:
                    f.write(f'{student_id},{ts}\n')
                label = f"ID: {student_id}"
            else:
                label = "Unknown"

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), font, 0.8, (255, 255, 255), 2)

        cv2.imshow('Recognition', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


@app.route('/')
def index():
    return "Cloud Attendance System is running!"


@app.route('/train', methods=['POST'])
def train_model():
    try:
        train_faces()
        return jsonify({"status": "Training complete"})
    except Exception as e:
        return jsonify({"status": "Training failed", "error": str(e)})


@app.route('/recognize', methods=['POST'])
def recognize_model():
    try:
        recognize_faces()
        return jsonify({"status": "Recognition process finished"})
    except Exception as e:
        return jsonify({"status": "Recognition failed", "error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

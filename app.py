from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return "Cloud Attendance System is running!"

@app.route('/train', methods=['POST'])
def train_model():
    # You can use LBPHFaceRecognizer or FisherFaceRecognizer
    # Uncomment the one you are using, but ensure opencv-contrib-python is installed

    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer = cv2.face.FisherFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        Ids = []
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(Id)
        return faceSamples, Ids

    faces, Ids = getImagesAndLabels('TrainingImage')
    recognizer.train(faces, np.array(Ids))
    if not os.path.exists('TrainingImageLabel'):
        os.makedirs('TrainingImageLabel')
    recognizer.save('TrainingImageLabel/trainner.yml')

    return jsonify({"status": "Training complete"})

@app.route('/recognize', methods=['POST'])
def recognize():
    os.system('python AMS_Run.py')
    return jsonify({"status": "Recognition started"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

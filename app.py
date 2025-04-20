from flask import Flask, request, jsonify
import os
import cv2,os
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return "Cloud Attendance System is running!"

@app.route('/train', methods=['POST'])
def train_model():

# recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer=cv2.face.createFisherFaceRecognizer_create()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids

faces,Ids = getImagesAndLabels('TrainingImage')
recognizer.train(faces, np.array(Ids))
recognizer.save('TrainingImageLabel/trainner.yml')
    os.system('python training.py')
    return jsonify({"status": "Training complete"})

@app.route('/recognize', methods=['POST'])
def recognize():
    # You can import AMS_Run logic here
    os.system('python AMS_Run.py')
    return jsonify({"status": "Recognition started"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

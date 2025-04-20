from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "Cloud Attendance System is running!"

@app.route('/train', methods=['POST'])
def train_model():
    # You can import your training.py logic here
    os.system('python training.py')
    return jsonify({"status": "Training complete"})

@app.route('/recognize', methods=['POST'])
def recognize():
    # You can import AMS_Run logic here
    os.system('python AMS_Run.py')
    return jsonify({"status": "Recognition started"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

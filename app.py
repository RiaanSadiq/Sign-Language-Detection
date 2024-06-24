from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import logging

app = Flask(__name__)

# Load the sign language model
model = load_model("signlanguagedetectionmodel50x50.h5")

# Labels for prediction
label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'blank']

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 50, 50, 1)
    return feature / 255.0

camera = cv2.VideoCapture(0)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    
    pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)


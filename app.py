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

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            app.logger.error("Failed to read frame from camera")
            break
        else:
            app.logger.info("Frame captured")
           
            cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
            crop_frame = frame[40:300, 0:300]
            crop_frame_gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            crop_frame_resized = cv2.resize(crop_frame_gray, (50, 50))
            crop_frame_normalized = extract_features(crop_frame_resized)
            
            # Prediction
            pred = model.predict(crop_frame_normalized)
            prediction_label = label[pred.argmax()]
            
            # Display prediction
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
            if prediction_label == 'blank':
                cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                accu = "{:.2f}".format(np.max(pred) * 100)
                cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                app.logger.error("Failed to encode frame")
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)


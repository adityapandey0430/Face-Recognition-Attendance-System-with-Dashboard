from flask import Flask, render_template, Response
import cv2
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Load model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

student_df = pd.read_csv("student_details.csv")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

marked_ids = set()

def generate_frames():
    while True:
        success, img = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:

            id, conf = recognizer.predict(gray[y:y+h,x:x+w])

            if conf < 60:
                row = student_df[student_df["ID"].astype(int) == int(id)]

                if not row.empty:
                    name = row["Name"].values[0]

                    label = name
                else:
                    label = "Unknown"
            else:
                label = "Unknown"

            # Draw rectangle + name
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import pandas as pd
from datetime import datetime
import time

app = Flask(__name__)

# 🔥 Global result storage
result_data = {}

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Face detector
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Student data
student_df = pd.read_csv("student_details.csv")

# Camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

marked_ids = set()


def generate_frames(user_id):
    global result_data

    while True:
        success, img = camera.read()

        if not success:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 60 and str(id) == str(user_id):
                row = student_df[
                    student_df["ID"].astype(str) == str(id)
                ]

                if not row.empty:
                    name = row["Name"].values[0]
                    roll = row["Roll"].values[0]

                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time_now = now.strftime("%H:%M:%S")

                    # Save attendance
                    if roll not in marked_ids:
                        df = pd.DataFrame(
                            [[name, roll, date, time_now]],
                            columns=["Name", "Roll", "Date", "Time"]
                        )
                        df.to_csv("attendance.csv", mode='a',
                                  header=False, index=False)

                        marked_ids.add(roll)

                    # 🔥 Save result
                    result_data = {
                        "name": name,
                        "roll": roll,
                        "date": date,
                        "time": time_now
                    }

                    # Stop camera
                    camera.release()

                    return

            # Draw rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.03)


@app.route('/')
def index():
    return render_template('camera.html')


@app.route('/video')
def video():
    user_id = request.args.get("user")

    # Start stream
    return Response(generate_frames(user_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result():
    return render_template("result.html",
                           name=result_data.get("name"),
                           roll=result_data.get("roll"),
                           date=result_data.get("date"),
                           time=result_data.get("time"))


# 🔥 AUTO REDIRECT CHECK
@app.route('/check')
def check():
    if result_data:
        return redirect(url_for('result'))
    return "waiting"


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)

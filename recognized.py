import cv2
import pandas as pd
from datetime import datetime
import os

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load student details
student_df = pd.read_csv("student_details.csv")

# Start camera
cam = cv2.VideoCapture(0)

attendance_file = "attendance.csv"

# Create attendance file if not exists
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name","Roll","Date","Time"])
    df.to_csv(attendance_file, index=False)

marked_ids = set()

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:

        # 🔥 PROFILE CARD BOX
        cv2.rectangle(img, (400, 50), (630, 200), (0, 0, 0), -1)
        cv2.rectangle(img, (400, 50), (630, 200), (0, 255, 0), 2)

        id, conf = recognizer.predict(gray[y:y+h,x:x+w])

        if conf < 60:
            row = student_df[student_df["ID"].astype(int) == int(id)]

            if not row.empty:
                name = row["Name"].values[0]
                roll = row["Roll"].values[0]

                now = datetime.now()
                time = now.strftime("%H:%M:%S")

                # Mark attendance once
                if roll not in marked_ids:
                    date = now.strftime("%Y-%m-%d")
                    df = pd.DataFrame([[name, roll, date, time]])
                    df.to_csv(attendance_file, mode='a', header=False, index=False)
                    marked_ids.add(roll)

                #  SHOW PROFILE CARD TEXT
                cv2.putText(img, "STUDENT PROFILE", (410, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.putText(img, f"Name: {name}", (410, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                cv2.putText(img, f"Roll: {roll}", (410, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                cv2.putText(img, "Status: Present", (410, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.putText(img, f"Time: {time}", (410, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                label = name
            else:
                label = "Unknown"
        else:
            label = "Unknown"

        # Face box + label
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Attendance System', img)

    # Press ENTER to exit
    if cv2.waitKey(1) == 13:
        break

cam.release()
cv2.destroyAllWindows()
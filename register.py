import cv2
import os

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

user_id = input("Enter User ID: ")
count = 0

os.makedirs("dataset", exist_ok=True)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('Register Face', img)

    if cv2.waitKey(1) == 13 or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
print("Dataset Created")
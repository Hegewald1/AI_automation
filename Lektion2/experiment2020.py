import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# https://github.com/opencv/opencv/tree/master/data/haarcascades

#To download them, right click “Raw” => “Save link as”. Make sure they are in your working directory.

cap = cv2.VideoCapture(0)
cv2.namedWindow('image')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

static_back = None
while True:
    _, frame = cap.read()

    # detect motion
    motion = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if static_back is None:
        static_back = gray
        continue
    diff_frame = cv2.absdiff(static_back, gray)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    #thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
    cnts, _ = cv2.findContours(thresh_frame.copy(),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        #motion = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # find full body
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    rects, weights = hog.detectMultiScale(gray_frame)
    for i, (x, y, w, h) in enumerate(rects):
        if weights[i] < 0.7:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Full body", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #print('Full body found\n')

    # face
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.putText(frame, "Face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        gray_face = gray_frame[y:y + h, x:x + w]  # cut the gray face frame out
        face = frame[y:y + h, x:x + w]  # cut the face frame out

        # eyes
        eyes = eye_cascade.detectMultiScale(gray_face)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 225, 255), 2)
            # print('Face found\n')

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
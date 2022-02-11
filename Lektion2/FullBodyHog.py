import numpy as np
import cv2

cap = cv2.VideoCapture(0)
print(cap.isOpened())
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
while True:
    r, frame = cap.read()
    if r:
        frame = cv2.resize(frame, (640, 360))  # Downscale to improve frame rate
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # HOG needs a grayscale image

        rects, weights = hog.detectMultiScale(gray_frame)


        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.7:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("preview", frame)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"):  # Exit condition
      break

cap.release()
cv2.destroyAllWindows()

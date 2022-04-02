import cv2
import numpy as np

# load cascades for full  body and faces
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def motion_detect(capture):
    _, img_1 = capture.read()
    _, img_2 = capture.read()

    diff = cv2.absdiff(img_1, img_2)

    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    diff_blur = cv2.GaussianBlur(diff_gray, (21, 21), 0)

    thresh_frame = cv2.threshold(diff_blur, 30, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 1000:
            cv2.rectangle(img_1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_1


lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])


def pedestrian_detect(capture):
    r, frame = capture.read()
    if r:
        frame = cv2.resize(frame, (640, 360))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # for color detecting
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, lower_red, upper_red)
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        rects, weights = hog.detectMultiScale(gray_frame)

        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] < 0.5:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # print('Full body detected')


        # TODO detect color - maybe of what they are wearing

        return frame


def face_detect(capture):
    _, frame = capture.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame


# capture video from device - standard 0
v_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mode, img = 'motion', None

while v_capture.isOpened():
    # different detection types - mode is set to motion as standard
    if mode == 'motion':
        img = motion_detect(v_capture)
        # start with detecting motion
    elif mode == 'full body':
        img = pedestrian_detect(v_capture)
    elif mode == 'face':
        img = face_detect(v_capture)

    # show the results from detection
    cv2.imshow(mode, img)

    # quit or switch detection mode
    # destroy old window on switch
    key = cv2.waitKey(1)
    if key == ord('q'):  # quit
        v_capture.release()
    elif key == ord('p'):  # switch to pedestrian detection
        cv2.destroyWindow(mode)
        mode = 'full body'
    elif key == ord('f'):  # switch to face detection
        cv2.destroyWindow(mode)
        mode = 'face'
    elif key == ord('m'):  # swtich back to motion
        cv2.destroyWindow(mode)
        mode = 'motion'

cv2.destroyAllWindows()
print('Done')

import cv2
import numpy as np
import sys


# ----- Exercise 1 -----
def exercise1():
    a = np.array([1, 2, 3])
    print(a)

    print("The Python version is %s.%s.%s" % sys.version_info[:3])

    img = cv2.imread('peopleselbow.jpg', 0)
    cv2.imshow('image', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ----- Exercise 2 -----
def exercise2():
    img = cv2.imread('peopleselbow.jpg', 0)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == 1:  # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'):  # wait for 's' key to save and exit
        cv2.imwrite('therockgray.png', img)
        cv2.destroyAllWindows()

def videocapture():
    cap = cv2.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        color = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # Display the resulting frame
        cv2.imshow('frame', color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


exercise2()

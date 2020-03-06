import time
import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2 as cv

cap = cv.VideoCapture('test.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv.imshow('frame',frame)
        time.sleep(1/25.0)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()

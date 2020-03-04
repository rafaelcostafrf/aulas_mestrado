import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2 as cv

cap = cv.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	cv.imshow('frame',gray)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv.destroyAllWindows()

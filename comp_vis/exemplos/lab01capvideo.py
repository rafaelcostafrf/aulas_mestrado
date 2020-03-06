import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2 as cv

cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('saidavideo.avi',fourcc,20.0,(640,480))

while (cap.isOpened()):
	ret, frame = cap.read()
	if ret == True :
		out.Write(frame)
		cv.imshow('frame',frame)
		if cv.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
cap.release()
out.release()
cv.destroyAllWindows()

import numpy as np
import sys
import time
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2

cap = cv2.VideoCapture(0)
fps = 15.0
print("O framerate da webcam: " + str(fps))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('saidavideo.avi',fourcc,fps,(640,480))

inicio = time.perf_counter()
while (cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:		
		if time.perf_counter()-inicio > 1/fps:
			inicio = time.perf_counter()
			out.write(frame)		
			cv2.imshow('frame',frame)	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
        
cap.release()
out.release()
cv2.destroyAllWindows()

cap = cv2.VideoCapture('saidavideo.avi')

while(cap.isOpened()):
	inicio = time.perf_counter()
	ret, frame = cap.read()
	if ret == True:		
		cv2.imshow('frame',frame)
		cv2.waitKey(1)
	else:
		break
	if 1/fps-(time.perf_counter()-inicio) < 0:
		dorme = 0
		print("atraso")
	else:
		dorme = 1/fps-(time.perf_counter()-inicio)
	time.sleep(dorme)
	
cap.release()
cv2.destroyAllWindows()

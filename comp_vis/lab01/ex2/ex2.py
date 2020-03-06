import numpy as np
import sys
import time
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2
from setup_camera import set_camera

fps = 20.0
largura = 320
altura = 240
brilho = 120

cap = cv2.VideoCapture(0)

set_camera(largura, altura, fps, brilho, cap)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('saidavideo.avi',fourcc,fps,(largura,altura))

while (cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
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
	ret, frame = cap.read()
	if ret == True:
		inicio = time.perf_counter()
		cv2.imshow('frame',frame)
	else:
		break
	if 1/fps-(time.perf_counter()-inicio) < 0:
		cv2.waitKey(1)
	else:
		dorme = 1/fps-(time.perf_counter()-inicio)
		cv2.waitKey(int(dorme*1000))
cap.release()
cv2.destroyAllWindows()

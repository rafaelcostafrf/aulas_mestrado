import cv2 as cv
import numpy as np
import glob

names = glob.glob('./drone_camera/*.jpg')
h, w, l = cv.imread(names[0]).shape
size = (w, h)
out = cv.VideoWriter('./drone_camera/drone_camera_video.avi', cv.VideoWriter_fourcc(*'MJPG'), 10, size)

for name in names:
    img = cv.imread(name)
    out.write(img)

out.release()    


import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2


img = cv2.imread('image1.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
coef = np.array([[0, 1, 1]])
gray_coef = cv2.transform(img, coef) 


h, w, _ = img.shape

src_ver = np.zeros((h, w, 3), np.uint8)
src_ver[:] = (120,0,20)

gray_alt = cv2.cvtColor(cv2.absdiff(img,src_ver), cv2.COLOR_BGR2GRAY)

cv2.imshow('image',img)
cv2.imshow('gray',gray)
cv2.imshow('gray_alt',gray_alt)
cv2.imshow('coef_custom',gray_coef)

cv2.waitKey()
cv2.destroyAllWindows()

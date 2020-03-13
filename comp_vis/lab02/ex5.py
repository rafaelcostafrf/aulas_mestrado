import cv2
import numpy as np
from matplotlib import pyplot as plt

img_c = cv2.imread('arquivos_auxiliares/peppers.png')
(largura, altura) = img_c.shape[:2]

hist = np.zeros([256,3])
ind = np.arange(256)

for ii in range(largura):
    for jj in range(altura):
        for k in range(3):
            hist[img_c[ii,jj,k],k] += 1
            
b=plt.plot(ind,hist[:,0],color='b')
g=plt.plot(ind,hist[:,1],color='g')
r=plt.plot(ind,hist[:,2],color='r')
plt.show()

img_c_b = img_c[:,:,0]
img_c_g = img_c[:,:,1]
img_c_r = img_c[:,:,2]

img_eq_b = cv2.equalizeHist(img_c_b)
cv2.imshow('teste',img_eq_b)
cv2.waitKey(0)
img_eq_g = cv2.equalizeHist(img_c_g)
cv2.imshow('teste',img_eq_g)
cv2.waitKey(0)
img_eq_r = cv2.equalizeHist(img_c_r)
cv2.imshow('teste',img_eq_r)
cv2.waitKey(0)

img_eq = np.zeros([largura,altura,3], dtype=np.uint8)
img_eq[:,:,0] = img_eq_b
img_eq[:,:,1] = img_eq_g
img_eq[:,:,2] = img_eq_r

cv2.imshow('histograma equalizado',img_eq)
cv2.waitKey(0)

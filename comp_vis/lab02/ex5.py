import cv2
import numpy as np
from matplotlib import pyplot as plt

img_c = cv2.imread('arquivos_auxiliares/peppers.png')
(largura, altura) = img.shape[:2]

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
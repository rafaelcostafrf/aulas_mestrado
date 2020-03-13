import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('arquivos_auxiliares/coins.png')
(largura, altura) = img.shape[:2]

hist = np.zeros(256)
ind = np.arange(256)

for ii in range(largura):
    for jj in range(altura):
        hist[img[ii,jj]] += 1
plt.plot(ind,hist)
plt.show()


img2 = cv2.imread('arquivos_auxiliares/rice.png')
(largura, altura) = img.shape[:2]

hist = np.zeros(256)
ind = np.arange(256)

for ii in range(largura):
    for jj in range(altura):
        hist[img2[ii,jj]] += 1
plt.plot(ind,hist)
plt.show()




import numpy as np
import cv2 

img = cv.imread('messi5.jpg')

cv.imshow('Imagem Original', img)
cv.waitKey(0)

px = img[100,100]
print(px)

#acessando apenas um valor de cor
azul = img[100,100,0]
print(azul)
red = img.item(10,11,2)
print(red)

#alterando um pixel
img[100,100] = [255,255,255]
img.itemset((10,11,2), 100)

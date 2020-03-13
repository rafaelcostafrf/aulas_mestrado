import numpy as np
import cv2

img = cv2.imread('arquivos_auxiliares/Tile_WazirKhanmosque.jpg')
ruido = np.uint8((1.5*img.std()*np.random.random(img.shape)))
img = img+ruido

size=5

k_f = cv2.GaussianBlur(img,(size,size),10)

cv2.imshow('teste2',img)
cv2.imshow('teste',k_f)
cv2.waitKey(0)


size=5

k_f = cv2.blur(img,(size,size))

cv2.imshow('teste2',img)
cv2.imshow('teste',k_f)
cv2.waitKey(0)

size=5

k_f = cv2.medianBlur(img,size)

cv2.imshow('teste2',img)
cv2.imshow('teste',k_f)
cv2.waitKey(0)
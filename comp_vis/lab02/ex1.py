import numpy as np
import cv2 

img = cv2.imread('fotos_ex/rosto.jpg')
obj = cv2.imread('fotos_ex/objeto.jpg')
cv2.imshow('Imagem Original', img)
cv2.imshow('Objeto', obj)

in_h = 470
in_v = 230
fim_h = 470+250
fim_v = 230+250

ROI = img[in_v:fim_v, in_h:fim_h]
obj[in_v:fim_v, in_h-120:fim_h-120]= ROI
cv2.imshow('ROI',ROI)
cv2.imshow('Mistura',obj)
cv2.imwrite('fotos_ex/mistura_ROI.jpg',obj)
cv2.waitKey(0)


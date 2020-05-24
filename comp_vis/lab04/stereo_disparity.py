import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

"""
INF209B − TÓPICOS ESPECIAIS EM PROCESSAMENTO DE SINAIS:

VISAO COMPUTACIONAL

PRÁTICA 04

RA: 21201920754
NOME: RAFAEL COSTA FERNANDES
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIÇÃO:
Exercício n.4

Algoritmo para obtenção de mapa de disparidade baseado em um par de imagems estereoscópicas. 
Ajustar os parâmetros de setup do algoritmo SGBM, conforme necessidade.


Fonte parcial do código: 
https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
"""
plt.close('all')


#Utilização de uma imagem estéreo já calibrada
img1 = cv.imread('.\\imagens\\disparities\\view0.png',0)

img2 = cv.imread('.\\imagens\\disparities\\view1.png',0)

#Equalização da imagem
img2 = cv.equalizeHist(img2)
img1 = cv.equalizeHist(img1)


#Plot da imagem original
plt.close('all')
plt.figure('Original', figsize=(18,16))
ax1=plt.subplot(121)
ax1.imshow(img1, cmap='gray')
ax2=plt.subplot(122)
ax2.imshow(img2, cmap='gray')
plt.show()


#Setup do algoritmo StereoSGBM
win_size = 2
min_disp = -1
max_disp = 16*8-1
num_disp = max_disp - min_disp

  
stereo = cv.StereoSGBM_create(minDisparity= min_disp,
     numDisparities = num_disp,
     blockSize = 5,
     uniquenessRatio = 5,
     speckleWindowSize = 5,
     speckleRange = 5,
     disp12MaxDiff = 1,
     P1 = 8*4*win_size**2,
     P2 =32*4*win_size**2)

#Cálculo do mapa de disparidade
disparity = stereo.compute(img1, img2)
#Normalização do resultado entre 0 e 255 e uint8
disparity -= np.min(disparity)
disparity = np.multiply(disparity,255.0/np.max(disparity))
disparity = disparity.astype(np.uint8)

#Plot do resultado
plt.figure(figsize=(12,8))
string = f'Disparity Map'
plt.title(string)
plt.imshow(disparity, cmap='gray')
plt.draw()
plt.pause(0.5)

#Salva o mapa obtido
caminho_mapa = '.\\imagens\\disparities\\Disparidade_Calculada.png'
cv.imwrite(caminho_mapa, disparity)
print('Mapa de disparidade salvo em: '+caminho_mapa)
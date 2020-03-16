# INF209B − TOPICOS ESPECIAIS EM PROCESSAMENTO DE SINAIS:
# VISAO COMPUTACIONAL
#
# PRATICA 02
#
# RA: 21201920754
# NOME: RAFAEL COSTA FERNANDES
#
# E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR
#
# DESCRICAO:
# Exercicio n.5 - Histograma BGR e equalizacao
# Abre uma foto e calcula o histograma pixel a pixel para cada canal de cor.
# Separa cada canal de cor em uma imagem independente,
# Realiza a equalizacao em cada canal de cor separadamente, depois realoca os canais em uma imagem unica


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

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
# Exercicio n.7 - Tecnicas de Filtragem
# Abre uma foto, adiciona ruido aleatorio, e Blur Gaussiano, blur, e um blur por media.
# Mostra as imagens, para poder comparar entre elas.


import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.cvtColor(cv2.imread('fotos_ex/image1.jpeg'),cv2.COLOR_BGR2RGB)
ruido = np.uint8((1.5*img.std()*np.random.random(img.shape)))
img = cv2.add(img,ruido)

size=3

k_f = cv2.GaussianBlur(img,(size,size),10)

plt.figure(figsize=(20,20))
ax1=plt.subplot(121)
plt.imshow(img)
ax2=plt.subplot(122)
plt.imshow(k_f)



size=5

k_f = cv2.blur(img,(size,size))

plt.figure(figsize=(20,20))
ax3=plt.subplot(121)
plt.imshow(img)
ax4=plt.subplot(122)
plt.imshow(k_f)


size=5

k_f = cv2.medianBlur(img,size)

plt.figure(figsize=(20,20))
ax5=plt.subplot(121)
plt.imshow(img)
ax6=plt.subplot(122)
plt.imshow(k_f)


ax1.title.set_text('Ruido')
ax2.title.set_text('Blur Gaussiano')
ax3.title.set_text('Ruido')
ax4.title.set_text('Blur')
ax5.title.set_text('Ruido')
ax6.title.set_text('Blur por Mediana')
plt.show()
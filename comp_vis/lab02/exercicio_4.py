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
# Exercicio n.4 - Histograma
# Abre uma foto e calcula o histograma pixel a pixel.
# utiliza a biblioteca matplotlib para expor o histograma em 8 bits.


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('arquivos_auxiliares/coins.png')

plt.figure(figsize=(10,10))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Moedas')
plt.show()

(largura, altura) = img.shape[:2]

hist = np.zeros(256)
ind = np.arange(256)

for ii in range(largura):
    for jj in range(altura):
        hist[img[ii,jj]] += 1
plt.figure(figsize=(10,10))
plt.plot(ind,hist)
plt.title('Histograma - Moedas')
plt.xlabel('Valor')
plt.ylabel('Ocorrencia')
plt.show()


img2 = cv2.imread('arquivos_auxiliares/rice.png')
plt.figure(figsize=(10,10))
plt.imshow(img2, cmap='gray', vmin=0, vmax=255)
plt.title('Arroz')
plt.show()

(largura, altura) = img2.shape[:2]

hist = np.zeros(256)
ind = np.arange(256)

for ii in range(largura):
    for jj in range(altura):
        hist[img2[ii,jj]] += 1
plt.figure(figsize=(10,10))
plt.plot(ind,hist)
plt.title('Histograma - Arroz')
plt.xlabel('Valor')
plt.ylabel('Ocorrencia')
plt.show()




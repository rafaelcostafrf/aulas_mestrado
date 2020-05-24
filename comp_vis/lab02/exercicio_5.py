import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
INF209B − TÓPICOS ESPECIAIS EM PROCESSAMENTO DE SINAIS:
VISÃO COMPUTACIONAL

PRÁTICA 02

RA: 21201920754
NOME: RAFAEL COSTA FERNANDES
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIÇÃO:
Exercício n.5 
    Histograma BGR e equalizacao
    Abre uma foto e calcula o histograma pixel a pixel para cada canal de cor.
    Separa cada canal de cor em uma imagem independente,
    Realiza a equalizacao em cada canal de cor separadamente, depois realoca os canais em uma imagem unica
"""  

img_c = cv2.imread('arquivos_auxiliares/peppers.png')

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB))
plt.title("Foto Original")
plt.show()
(largura, altura) = img_c.shape[:2]

hist = np.zeros([256,3])
ind = np.arange(256)

for ii in range(largura):
    for jj in range(altura):
        for k in range(3):
            hist[img_c[ii,jj,k],k] += 1
plt.figure(figsize=(10,10))            
b=plt.plot(ind,hist[:,0],color='b')
g=plt.plot(ind,hist[:,1],color='g')
r=plt.plot(ind,hist[:,2],color='r')
plt.title("Histograma foto original")
plt.xlabel("valores")
plt.ylabel("ocorrencia")
plt.show()

img_c_b = img_c[:,:,0]
img_c_g = img_c[:,:,1]
img_c_r = img_c[:,:,2]

img_eq_b = cv2.equalizeHist(img_c_b)
plt.figure(figsize=(10,10))
plt.imshow(img_eq_b, cmap='gray', vmin=0, vmax=255)
plt.title("Canal azul equalizado")
plt.show()

img_eq_g = cv2.equalizeHist(img_c_g)
plt.figure(figsize=(10,10))
plt.imshow(img_eq_g, cmap='gray', vmin=0, vmax=255)
plt.title("Canal  verde equalizado")
plt.show()

img_eq_r = cv2.equalizeHist(img_c_r)
plt.figure(figsize=(10,10))
plt.imshow(img_eq_r, cmap='gray', vmin=0, vmax=255)
plt.title("Canal vermelho equalizado")
plt.show()

img_eq = np.zeros([largura,altura,3], dtype=np.uint8)
img_eq[:,:,0] = img_eq_b
img_eq[:,:,1] = img_eq_g
img_eq[:,:,2] = img_eq_r

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(img_eq,cv2.COLOR_BGR2RGB))
plt.title("Imagem final equalizada")
plt.show()
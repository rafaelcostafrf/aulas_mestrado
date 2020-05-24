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
Exercício n.1
Importa a duas fotos tiradas no experimento anterior
Seleciona o ROI da foto do rosto e aplica em uma posicao escolhida no objeto
Salva a imagem final
"""



img = cv2.cvtColor(cv2.imread('fotos_ex/rosto.jpg'),cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.title('Rosto Original')
plt.show()

obj = cv2.cvtColor(cv2.imread('fotos_ex/objeto.jpg'),cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(obj)
plt.title('Objeto')
plt.show()

#escolha de retirada do ROI
in_h = 470
in_v = 230
fim_h = 470+250
fim_v = 230+250
ROI = img[in_v:fim_v, in_h:fim_h]

#colocação do ROI na posição
obj[in_v:fim_v, in_h-120:fim_h-120]= ROI

plt.figure(figsize=(10,10))
plt.imshow(obj)
plt.title('Foto Mesclada')
plt.show()
cv2.imwrite('fotos_ex/mistura_ROI.jpg',cv2.cvtColor(obj, cv2.COLOR_RGB2BGR))
print('Foto Salva em fotos_ex/mistura_ROI.jpg')
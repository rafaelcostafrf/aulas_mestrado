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
# Exercicio n.1
# Importa a duas fotos tiradas no experimento anterior
# Seleciona o ROI da foto do rosto e aplica em uma posicao escolhida no objeto
# Salva a imagem final

from matplotlib import pyplot as plt
import numpy as np
import cv2 

img = cv2.cvtColor(cv2.imread('fotos_ex/rosto.jpg'),cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()
obj = cv2.cvtColor(cv2.imread('fotos_ex/objeto.jpg'),cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(obj)
plt.show()

in_h = 470
in_v = 230
fim_h = 470+250
fim_v = 230+250

ROI = img[in_v:fim_v, in_h:fim_h]
obj[in_v:fim_v, in_h-120:fim_h-120]= ROI
plt.figure(figsize=(10,10))
plt.imshow(obj)
plt.show()
cv2.imwrite('fotos_ex/mistura_ROI.jpg',obj)


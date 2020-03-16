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
# Exercicio n.3 - Limiarizacao
# Abre uma foto e realiza dez passos de limiarizacao, de 0 a 100%.
# Operacao pixel a pixel, o openCv trabalha com intensidades 0 para o valor Falso e 255 para o valor True.

import cv2
import numpy as np

img = cv2.imread('arquivos_auxiliares/rice.png')
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lim = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]



(altura,largura) = img.shape[:2]
thr = np.zeros([altura,largura], dtype=np.uint8)

for k in lim:
    for ii in range(largura):
        for jj in range(altura):
            thr_v = k*255
            if img_g[ii,jj] < thr_v:
                thr[ii,jj] = 0
            else:
                thr[ii,jj] = 255
    cv2.imshow(("limiarizacao em %.1f" %k),thr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
                
        
            
    

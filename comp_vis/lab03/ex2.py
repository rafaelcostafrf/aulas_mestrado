import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
"""
INF209B − TÓPICOS ESPECIAIS EM PROCESSAMENTO DE SINAIS:

VISAO COMPUTACIONAL

PRÁTICA 03

RA: 21201920754
NOME: RAFAEL COSTA FERNANDES
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIÇÃO:
Exercício n.2

Determina as bordas de uma imagem baseado no algoritmo watershed, que pode ser separado em diversas etapas:
1. Limiarização da imagem a partir do algoritmo de Otsu
2. Remoção de Ruídos da imagem
3. Determinação da área de fundo da imagem
4. Determinação da área de frente da imagem
5. Determinação da região desconhecida 
6. Preenchimento da imagem com o algoritmo watershed (bacia hidrográfica)
"""
plt.close('all')


images = (cv.cvtColor(cv.imread('./arq_aux/rafael.jpeg'), cv.COLOR_BGR2RGB),
          cv.imread('./arq_aux/rice.png'),
          cv.imread('./arq_aux/cameraman.tif'),
          cv.cvtColor(cv.imread('./arq_aux/peppers.png'), cv.COLOR_BGR2RGB))

titles = ('Algoritmo Watershed - Rafael',
          'Algoritmo Watershed - Arroz',
          'Algoritmo Watershed - Cameraman',
          'Algoritmo Watershed - Pimentas')

t_f = []
for img, title in zip(images, titles):
    t_i = time.time()
    plt.figure(figsize=(18, 16))
    
    plt.suptitle(title)
    plt.subplot(331).imshow(img)
    plt.title('Imagem Original')
    
    cinza = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(cinza, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
    plt.subplot(332).imshow(thresh, cmap='gray')
    plt.title('Limiarizacao')
    
    # Remoção de ruídos
    kernel = np.ones((3,3), np.uint8)
    abertura = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)
    
    plt.subplot(333).imshow(abertura, cmap='gray')
    plt.title('Remoção de Ruído')
    
    # Areas de Fundo
    fundo = cv.dilate(abertura, kernel, iterations=2)
    
    plt.subplot(334).imshow(fundo, cmap='gray')
    plt.title('Área de Fundo')
    
    # Áreas de frente
    dist_transform = cv.distanceTransform(abertura, cv.DIST_L2, 5)
    
    plt.subplot(335).imshow(dist_transform, cmap='gray')
    plt.title('Transformada de Distância')
    
    ret, frente = cv.threshold(dist_transform, 0.15*dist_transform.max(), 255, 0)
    
    plt.subplot(336).imshow(frente, cmap='gray')
    plt.title('Limiarização na transformada de distância')
    
    # Região desconhecida
    frente = np.uint8(frente)
    desconhecida = cv.subtract(fundo, frente)
    
    plt.subplot(337).imshow(desconhecida, cmap='gray')
    plt.title('Região desconhecida')
    
    # Marcadores
    ret, marcadores = cv.connectedComponents(frente)
    # Soma um em todos os marcadores (para que o fundo seja 1 e não zero)
    marcadores = marcadores+1
    # região desconhecida é zero
    marcadores[desconhecida==255] = 0
    
    plt.subplot(338).imshow(marcadores, cmap='jet')
    plt.title('Marcadores')
    
    marcadores = cv.watershed(img, marcadores)
    img[marcadores == -1] = [255,0,0]
    
    plt.subplot(339).imshow(img)
    plt.title('Imagem Final')
    t_f.append(time.time()-t_i)
    plt.show()

m_t = np.mean(t_f)
string = f'O tempo médio para o algoritmo foi de {m_t:.2f} segundos'
print(string)
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
plt.close('all')


images = (cv.cvtColor(cv.imread('./arq_aux/rafael.jpeg'), cv.COLOR_BGR2RGB),
          cv.imread('./arq_aux/rice.png'),
          cv.imread('./arq_aux/cameraman.tif'),
          cv.cvtColor(cv.imread('./arq_aux/peppers.png'), cv.COLOR_BGR2RGB))

titles = ('Algoritmo Watershed - Rafael',
          'Algoritmo Watershed - Arroz',
          'Algoritmo Watershed - Cameraman',
          'Algoritmo Watershed - Pimentas')


for img, title in zip(images, titles):    
    plt.figure()
    
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
    
    ret, frente = cv.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    
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
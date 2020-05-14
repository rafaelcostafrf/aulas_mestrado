import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
plt.close('all')


images = (~cv.imread('./arq_aux/rice.png'),
          cv.imread('./arq_aux/cameraman.tif'),
          cv.cvtColor(cv.imread('./arq_aux/peppers.png'), cv.COLOR_BGR2RGB))

titles = ('Algoritmo Watershed - Arroz',
          'Algoritmo Watershed - Cameraman',
          'Algoritmo Watershed - Pimentas')


for img, title in zip(images, titles):    
    plt.figure()
    
    plt.suptitle(title)
    plt.subplot(331).imshow(img)
    plt.title('Imagem Original')
    
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
    plt.subplot(332).imshow(thresh, cmap='gray')
    plt.title('Limiarizacao')
    
    # noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)
    
    plt.subplot(333).imshow(opening, cmap='gray')
    plt.title('Remoção de Ruído')
    
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=2)
    
    plt.subplot(334).imshow(sure_bg, cmap='gray')
    plt.title('Área de Fundo')
    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    
    plt.subplot(335).imshow(dist_transform, cmap='gray')
    plt.title('Transformada de Distância')
    
    ret, sure_fg = cv.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    
    plt.subplot(336).imshow(sure_fg, cmap='gray')
    plt.title('Limiarização na transformada de distância')
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    
    plt.subplot(337).imshow(unknown, cmap='gray')
    plt.title('Região desconhecida')
    
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    plt.subplot(338).imshow(markers, cmap='jet')
    plt.title('Marcadores')
    
    markers = cv.watershed(img, markers)
    img[markers == -1] = [255,0,0]
    
    plt.subplot(339).imshow(img)
    plt.title('Imagem Final')
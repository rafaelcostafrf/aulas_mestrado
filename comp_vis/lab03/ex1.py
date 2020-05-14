import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

plt.close('all')

## METODO DE CANNY

#COLORIDO
img = cv.imread('./arq_aux/objeto.jpg', 1)
edges = cv.Canny(img,100,200)


fig = plt.figure()
fig.suptitle('Método de Canny - Colorido')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.subplot(311).imshow(img_rgb)
plt.title('Imagem Original')

plt.subplot(312).imshow(edges, cmap='gray')
plt.title('Bordas')

height, width, channels = img_rgb.shape    
edges_rgb = np.ones((height, width, channels), np.uint8)
for i in range(channels):
    edges_rgb[:, :, i] = np.multiply(edges_rgb[:, :, i], edges)

img_sum = cv.add(img_rgb, edges_rgb)
plt.subplot(313).imshow(img_sum)
plt.title('Composicao')
plt.show()


#ESCALA DE CINZAS
img = cv.imread('./arq_aux/objeto.jpg', 0)
edges = cv.Canny(img,100,200)

fig2 = plt.figure()
fig2.suptitle('Método de Canny - Escala de Cinzas')
plt.subplot(311).imshow(img, cmap='gray')
plt.title('Imagem original em escalas de Cinza')

plt.subplot(312).imshow(edges, cmap='gray')
plt.title('Bordas')

img_sum = cv.add(img, edges)
plt.subplot(313).imshow(img_sum, cmap='gray')
plt.title('Composicao')
plt.show()


## METODO DE LAPLACE
#COLORIDO
img = cv.imread('./arq_aux/objeto.jpg', 1)
heigth, width, channels = img.shape
edges = cv.Laplacian(img, cv.CV_64F, ksize = 25)

#NORMALIZACAO DAS BORDAS PARA UINT8
edges = np.abs(edges)
edges = edges/edges.max()*255
edges = np.clip(edges, 0, 255)
edges = edges.astype(np.uint8)


fig = plt.figure()
fig.suptitle('Método de Laplace - Colorido')
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.subplot(311).imshow(img_rgb)
plt.title('Imagem Original')


img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
edges_rgb = cv.cvtColor(edges, cv.COLOR_BGR2RGB)
plt.subplot(312).imshow(edges_rgb)
plt.title('Bordas')

img_sum = cv.add(img_rgb, edges_rgb)
plt.subplot(313).imshow(img_sum)
plt.title('Composicao')
plt.show()


#ESCALA DE CINZAS
img = cv.imread('./arq_aux/objeto.jpg', 0)
edges = cv.Laplacian(img, cv.CV_64F, ksize = 25)
edges = np.abs(edges)
edges = edges/edges.max()*255
edges = np.clip(edges, 0, 255)
edges = edges.astype(np.uint8)

fig2 = plt.figure()
fig2.suptitle('Método de Laplace - Escala de Cinzas')
plt.subplot(311).imshow(img, cmap='gray')
plt.title('Imagem original em escalas de Cinza')

edges = np.clip(edges, 0, 255)
plt.subplot(312).imshow(edges, cmap='gray')
plt.title('Bordas')

img_sum = cv.add(img, edges)
plt.subplot(313).imshow(img_sum, cmap='gray')
plt.title('Composicao')
plt.show()
import numpy as np
import cv2 as cv
import time
from matplotlib import pyplot as plt

"""
INF209B − TÓPICOS ESPECIAIS EM PROCESSAMENTO DE SINAIS:

VISAO COMPUTACIONAL

PRÁTICA 03

RA: 21201920754
NOME: RAFAEL COSTA FERNANDES
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIÇÃO:
Exercício n.1

Determinação de bordas de uma imagem baseado em dois algoritmos: 
    Algoritmo de Canny:
        O algoritmo de canny segue uma lista de critérios para a detecção de bordas. 
        Entre os critérios mais importantes:
            Não pode haver falsos positivos.
            As bordas devem estar bem localizadas (A distancia entre a borda e a borda real deve ser mínima)
            Apenas uma resposta por borda
        
    Algoritmo Laplaciano:
        A detecção de bordas pelo algoritmo laplaciano segue uma premissa simples, 
        pixels em regiões de borda terão um pico de gradiente, em relação ao seu contorno.         

"""


plt.close('all')


def canny(caminho_arquivo, cor=True):
    t_i = time.time()
    # Leitura da Imagem
    img = cv.imread(caminho_arquivo, cor)
    # Reconhecimento das bordas pelo algoritmo de canny
    bordas = cv.Canny(img,100,200)
    # Configuração do plot da imagem original
    fig = plt.figure(figsize=(18, 16))
    titulo = 'Método de Canny - ' + ('Colorido' if cor else 'Escala de Cinzas')
    fig.suptitle(titulo)
    cmap = None if cor else 'gray'
    if cor:
        # Conversão para de BGR para RGB (convenção do matplotlib)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.subplot(311).imshow(img, cmap=cmap)
    
    # Plot das bordas em escala de cinzas
    plt.title('Imagem Original')
    plt.subplot(312).imshow(bordas, cmap='gray')
    plt.title('Bordas')
    
    # Composição das imagens e plot
    if cor:
        # Borda em escala de cinza precisa ser convertida para 3 canais RGB para soma
        bordas_d = bordas
        height, width, channels = img.shape    
        bordas = np.ones((height, width, channels), np.uint8)
        for i in range(channels):
            bordas[:, :, i] = np.multiply(bordas[:, :, i], bordas_d)
    
    # plot da composição da imagem original e bordas detectadas
    img_sum = cv.add(img, bordas)
    plt.subplot(313).imshow(img_sum, cmap=cmap)        
    plt.title('Composicao')
    plt.show()
    t_f = time.time() - t_i
    string =f'O algoritmo de Canny levou {t_f:.2f}s para uma ' + ('imagem colorida' if cor else 'imagem em escala de cinzas')
    print(string)


def laplace(caminho_arquivo, cor=True):
    t_i = time.time()
    # Leitura da Imagem
    img = cv.imread(caminho_arquivo, cor)
    cmap = None if cor else 'gray'    
    
    # Reconhecimento das bordas pelo algoritmo de laplace
    bordas = cv.Laplacian(img, cv.CV_64F, ksize = 25)
    
    # Normalização das bordas para uint8
    bordas = np.abs(bordas)
    bordas = bordas/bordas.max()*255
    bordas = np.clip(bordas, 0, 255)
    bordas = bordas.astype(np.uint8)
    
    #Plot da imagem original, das bordas e da composição
    titulo = 'Método de Laplace - ' + ('Colorido' if cor else 'Escala de Cinzas')
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(titulo)
    if cor:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        bordas = cv.cvtColor(bordas, cv.COLOR_BGR2RGB)
    plt.subplot(311).imshow(img, cmap=cmap)
    plt.title('Imagem Original')  
    plt.subplot(312).imshow(bordas, cmap=cmap)
    plt.title('Bordas')
    img_sum = cv.add(img, bordas)
    plt.subplot(313).imshow(img_sum, cmap=cmap)
    plt.title('Composicao')
    plt.show()
    t_f = time.time() - t_i
    string =f'O algoritmo Laplaciano levou {t_f:.2f}s para uma ' + ('imagem colorida' if cor else 'imagem em escala de cinzas')
    print(string)
    
## METODO DE CANNY

#COLORIDO
canny('./arq_aux/objeto.jpg', True)
#ESCALA DE CINZAS
canny('./arq_aux/objeto.jpg', False)


## METODO DE LAPLACE
#COLORIDO
laplace('./arq_aux/objeto.jpg', True)
#ESCALA DE CINZAS
laplace('./arq_aux/objeto.jpg', False)

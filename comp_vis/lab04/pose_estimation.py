import numpy as np
import cv2 as cv
import glob
import time
from matplotlib import pyplot as plt

"""
INF209B − TÓPICOS ESPECIAIS EM PROCESSAMENTO DE SINAIS:

VISAO COMPUTACIONAL

PRÁTICA 04

RA: 21201920754
NOME: RAFAEL COSTA FERNANDES
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIÇÃO:
Exercício n.2

Algoritmo de detecção de pose de um padrão xadrez. 
Utiliza técnica de PnP para a determinação da pose por vetores de rotação rvecs e translação através de um vetor tvecs.
Utiliza um randomizador para a figura que será desenhada na imagem (sistema de eixos ou cubo)


Fonte parcial do código: https://docs.opencv.org/master/d7/d53/tutorial_py_pose.html
"""
plt.close('all')
cameras = ['c270', 'c920']
t_chess = []
t_pnp = []
for camera in cameras:
    # Carrega as propriedades da camera
    caminho_propriedades = './camera_properties/'+camera+'.npz'
    with np.load(caminho_propriedades) as X:
        mtx, dist = [X[i] for i in ('mtx','dist')]
    
    # Define as funções de desenho (Sistema de eixos ou Cubo)
    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    
    def cube(img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
        # draw ground floor in green
        img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
        # draw pillars in blue color
        for i,j in zip(range(4),range(4,8)):
            img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        # draw top layer in red color
        img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
        return img
    
    axis_cube = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                        [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
    
    #Critério de determinação de vértices em subpixel
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    #Pontos do tabuleiro de xadrez
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    caminho_fotos = './imagens/'+camera+'/*.jpg'
    for i, fname in enumerate(glob.glob(caminho_fotos)):
        #Lê uma imagem no caminho das imagens
        img = cv.imread(fname)
        
        #Converte para escala de cinzas
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
        #Tempo inicial do algoritmo de busca do padrão xadrez
        t_i = time.time()
        #Algoritmo de busca do padrão xadrez
        ret, corners = cv.findChessboardCorners(gray, (9,6),None)
        t_chess.append(time.time()-t_i)
        
        #Se achar o padrão xadrez
        if ret == True: 
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            # Tempo inicial do algoritmo PnP
            t_i = time.time()
            
            # Determinação da atitude e posição (rvecs e tvecs)
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            t_pnp.append(time.time()-t_i)
            
            # Projeta os pontos 3D na imagem
            if np.random.random() < 0.5:
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                img = draw(img,corners2,imgpts)
            else:
                imgpts, jac = cv.projectPoints(axis_cube, rvecs, tvecs, mtx, dist)
                img = cube(img,corners2,imgpts)
            
            # Mostra a imagem
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            plt.title('Webcam: ' + camera + ' foto: ' +fname[15:-4])
            plt.show()
            plt.pause(0.5)
            
            # Salva a imagem
            cv.imwrite(fname[0:-4]+'_vec.png', img)
    print('Imagens salvas em: ' + caminho_fotos)
print(f'Tempo médio Detecção Tabuleiro {np.mean(t_chess)}')
print(f'Tempo médio Solução PnP {np.mean(t_pnp)}')
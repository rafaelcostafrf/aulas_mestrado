import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
"""
INF209B − TÓPICOS ESPECIAIS EM PROCESSAMENTO DE SINAIS:

VISAO COMPUTACIONAL

PRÁTICA 04

RA: 21201920754
NOME: RAFAEL COSTA FERNANDES
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIÇÃO:
Exercício n.1

Algoritmo de calibração de câmera baseado em detecção de padrão xadrez. 
O padrão utilizado tem 9 x 6 vértices internos (totalizando 10x7 "quadrados").
Para que a calibração seja boa, tirar a maior quantidade de fotos possível (pelo menos 10), com diversas poses do padrão xadrez.
Utilizar o padrão em uma superfície plana e rígida.


Fonte parcial do código: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
"""

plt.close('all')
# critério de finalização do algoritmo de subpixel
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 120, 0.0001)

# preparação dos pontos do objeto (tabuleiro de xadrez)
x_col = 9
x_lin = 6
objp = np.zeros((x_col*x_lin,3), np.float32)
objp[:,:2] = np.mgrid[0:x_col,0:x_lin].T.reshape(-1,2)

# Inicialização das listas
objpoints = [] # Pontos do objeto no mundo 3D
imgpoints = [] # Pontos do objeto no plano da imagem

cameras = ['c270', 'c920']
for camera in cameras:
    caminho = './imagens/'+camera+'/*.jpg'
    propriedades = './camera_properties/'+camera+'.npz'
    caminho_salva = './imagens/'+camera+'/resultado_calibr.png'
    images = glob.glob(caminho)
    for i, fname in enumerate(images):
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Acha os vértices do tabuleiro de xadrez
        ret, corners = cv.findChessboardCorners(gray, (x_col,x_lin), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Mostra o resultado obtido
            cv.drawChessboardCorners(img, (x_col,x_lin), corners2, ret)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            plt.title('Webcam: '+camera+' Foto: '+fname[15:-4])
            plt.show()
            plt.pause(1)
    # Determinação das matrizes da camera (matriz intrínseca e matriz de distorção)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    img = cv.imread(images[1])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    np.savez(propriedades, mtx=newcameramtx, dist=dist)
    print("Webcam " + camera + ":\nMatriz Intrínseca: ")
    print(mtx)
    print("Coeficientes de Distorção: ")
    print(dist)
    # tira a distorção da imagem
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # corta as partes da imagem sem informação (por causa do processo de retirada de distorção)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(caminho_salva, cv.cvtColor(dst, cv.COLOR_RGB2BGR))
    
    plt.figure(figsize=(18, 16))
    plt.title(camera)
    ax1 = plt.subplot(121)
    ax1.imshow(img)
    ax2 = plt.subplot(122)
    ax2.imshow(dst)
    ax1.set_title('Original')
    ax2.set_title('Sem Distorção')
    plt.show()  
    
    print('Imagem planificada salva em: ' + caminho_salva)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print('Webcam: ' + camera + " calibrada!\nErro Total: {}".format(mean_error/len(objpoints)) )
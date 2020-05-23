import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

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
    caminho_salva = './imagens/'+camera+'/resultado_calibração.png'
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
            plt.title('Webcam: '+camera+' Foto: '+str(i))
            plt.show()
            plt.pause(1)
    # Determinação das matrizes da camera (matriz intrínseca e matriz de distorção)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    img = cv.imread(images[1])
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    np.savez(propriedades, mtx=newcameramtx, dist=dist)
    # tira a distorção da imagem
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # corta as partes da imagem sem informação (por causa do processo de retirada de distorção)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(caminho_salva, dst)
    print('Imagem planificada salva em: ' + caminho_salva)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print('Webcam: ' + camera + " calibrada!\nErro Total: {}".format(mean_error/len(objpoints)) )
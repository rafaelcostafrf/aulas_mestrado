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
Exercécio n.3

Plota as epilinhas, dado duas imagens de um par de câmeras em disposição estéreo.
Preferencialmente deve haver um plano bem definido na imagem. 
Este plano será utilizado para a determinação do plano epipolar, e será a base para as epilinhas.

Fonte parcial do código: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
A função SIFT não existe mais no opencv 4.2 por questões de Direitos Autorais
O opencv deve ser revertido para a versão 3.4.2 contrib para compatibilidade com a função.
Aparentemente a versão 3.4.2 roda todos os algoritmos utilizados na disciplina, então sem problemas.
"""
plt.close('all')
cam_name = glob.glob('.\\camera_properties\\*.npz')
mtx = []
dist = []
for cam in cam_name:
    cam_par = np.load(cam)
    mtx.append(cam_par['mtx'])
    dist.append(cam_par['dist'])
   
    


# Imagem da camera esquerda
img1 = cv.imread('.\\imagens\\epipolar\\left.jpg',0)  

# Processo de igualar resolução e retirar as distorções
img1 = cv.resize(img1, (640, 480))
h,  w = img1.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx[0], dist[0], (w,h), 1, (w,h))
dst = cv.undistort(img1, mtx[0], dist[0], None, newcameramtx)
x, y, w, h = roi
img1 = dst[y:y+h, x:x+w]


# Imagem da camera direita
img2 = cv.imread('.\\imagens\\epipolar\\right.jpg',0) 
img2 = cv.resize(img2, (640, 480))

# Processo de igualar resolução e retirar as distorções
h,  w = img2.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx[1], dist[1], (w,h), 1, (w,h))
dst = cv.undistort(img2, mtx[1], dist[1], None, newcameramtx)
x, y, w, h = roi
img2 = dst[y:y+h, x:x+w]

plt.figure('Original', figsize=(16, 18))
ax1=plt.subplot(121)
ax1.imshow(img1, cmap='gray')
ax2=plt.subplot(122)
ax2.imshow(img2, cmap='gray')
ax1.set_title('Original - C920')
ax2.set_title('Original - C270')
plt.show()


# Determinação dos pontos chave por algoritmo SIFT
sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Parâmetros FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []

# Teste de Razão
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Matriz Fundamental
F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
print('Matriz Fundamental: ')
print(F)
# Matriz Essencial
E = []
nomes = ['c920', 'c270']
for mtx_i, nome in zip(mtx, nomes):
    E_i = np.dot(mtx_i.T, np.dot(F, mtx_i))
    print('Matriz essencial webcam '+name)
    print(E_i)
    E.append(E_i)

# Selecionando apenas pontos internos
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

# Função para desenhar as epilines
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Achando epilinhas da imagem direira e desenhando as linhas na imagem esquerda, vice versa.

lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)


#Plotando os resultados

plt.figure('Epilines', figsize=(16, 18))
ax1 =plt.subplot(121)
ax1.imshow(img5)
ax2=plt.subplot(122)
ax2.imshow(img3)
ax1.set_title('Epilines - C920')
ax2.set_title('Epilines - C270')
plt.show()
import numpy as np
import cv2

#aumenta exposicao

img_camera = cv2.imread('arquivos_auxiliares/cameraman.tif')
img_camera = cv2.cvtColor(img_camera, cv2.COLOR_BGR2GRAY)
(largura,altura) = img_camera.shape[:2]


cte = 100
for ii in range(largura):
    for jj in range(altura):
            if img_camera[ii,jj]/2 + cte/2>255/2:
                img_camera[ii,jj] = 255
            else:
                img_camera[ii,jj] = img_camera[ii,jj] + cte
cv2.imshow('Imagem exposição alterada',img_camera)
cv2.waitKey(0)


#soma ponderada carrinhos pb
img4 = cv2.imread('arquivos_auxiliares/toycars1.png')
img5 = cv2.imread('arquivos_auxiliares/toycars2.png')
img6 = cv2.imread('arquivos_auxiliares/toycars3.png')

a = 0.2989
b = 0.5870
c = 0.1140
cte = 0.3

img4_c = np.uint8(a*img4[:,:,0] + b*img4[:,:,1] + c*img4[:,:,2])
img5_c = np.uint8(a*img5[:,:,0] + b*img5[:,:,1] + c*img5[:,:,2])
img6_c = np.uint8(a*img6[:,:,0] + b*img6[:,:,1] + c*img6[:,:,2])
imgp_c = img4_c

(largura,altura) = img4_c.shape[:2]
for ii in range(largura):
    for jj in range(altura):
        if cte*img4_c[ii,jj]/3+cte*img5_c[ii,jj]/3+cte*img6_c[ii,jj]/3>255/3:
            imgp_c[ii,jj] = cte*img4_c[ii,jj]+cte*img5_c[ii,jj]+cte*img6_c[ii,jj]
        else:
            imgp_c[ii,jj] = cte*img4_c[ii,jj]+cte*img5_c[ii,jj]+cte*img6_c[ii,jj]
            
cv2.imshow('soma ponderada pb',imgp_c)
cv2.waitKey(0)

#soma ponderada carrinhos colorida

imgp_col = img4
(largura,altura) = img4.shape[:2]
for ii in range(largura):
    for jj in range(altura):
        for k in range(3):
            if cte*img4[ii,jj,k]/3+cte*img5[ii,jj,k]/3+cte*img6[ii,jj,k]/3>255/3:
                imgp_col[ii,jj,k] = cte*img4[ii,jj,k]+cte*img5[ii,jj,k]+cte*img6[ii,jj,k]
            else:
                imgp_col[ii,jj,k] = cte*img4[ii,jj,k]+cte*img5[ii,jj,k]+cte*img6[ii,jj,k]
                
cv2.imshow('soma ponderada colorida',imgp_col)
cv2.waitKey(0)

#subtracao coca cola

img7 = cv2.imread('arquivos_auxiliares/cola1.png')
img8 = cv2.imread('arquivos_auxiliares/cola2.png')

(largura,altura) = img7.shape[:2]
img_sub = np.zeros([largura,altura,3], dtype=np.uint8)

for ii in range(largura):
    for jj in range(altura):
        for k in range(3):
            if img7[ii,jj,k]/2-img8[ii,jj,k]/2+255/2<255/2:
                img_sub[ii,jj,k] = 0
            else:
                img_sub[ii,jj,k] = img7[ii,jj,k]-img8[ii,jj,k]
cv2.imshow('subtracao coca cola',img_sub)
cv2.waitKey(0)

#subtracao coca cola em cinzas

a = 0.2989
b = 0.5870
c = 0.1140

img7_c = np.uint8(a*img7[:,:,0] + b*img7[:,:,1] + c*img7[:,:,2])
img8_c = np.uint8(a*img8[:,:,0] + b*img8[:,:,1] + c*img8[:,:,2])


(largura,altura) = img7_c.shape[:2]
img_sub_c = np.zeros([largura,altura], dtype=np.uint8)
for ii in range(largura):
    for jj in range(altura):
        if (img7_c[ii,jj]/2-img8_c[ii,jj]/2)+255/2<255/2:
            img_sub_c[ii,jj] = 0
        else:
            img_sub_c[ii,jj] = img7_c[ii,jj]-img8_c[ii,jj]
              
            
cv2.imshow('subtracao pb',img_sub_c)
cv2.waitKey(0)


#soma opencv
#im3 = cv2.add(img1,img2)
#cv2.imshow('soma opencv',img3)
#cv2.waitKey(0)


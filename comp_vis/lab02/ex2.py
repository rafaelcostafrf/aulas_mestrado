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


#soma ponderada carrinhos
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
            
cv2.imshow('soma ponderada',imgp_c)
cv2.waitKey(0)


#soma sem overflow
#(largura,altura) = img1.shape[:2]
#for ii in range(largura):
#    for jj in range(altura):
#        for k in range(3):
#            if img1[ii,jj,k]/2+img2[ii,jj,k]/2>255/2:
#                img3[ii,jj,k] = 255
#            else:
#                img3[ii,jj,k] = img1[ii,jj,k]+img2[ii,jj,k]
#cv2.imshow('soma sem overflow',img3)
#cv2.waitKey(0)

#soma opencv
#im3 = cv2.add(img1,img2)
#cv2.imshow('soma opencv',img3)
#cv2.waitKey(0)


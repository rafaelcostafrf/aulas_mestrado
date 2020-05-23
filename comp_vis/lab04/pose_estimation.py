import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
plt.close('all')
cameras = ['c270', 'c920']
for camera in cameras:
    # Load previously saved data
    caminho_propriedades = './camera_properties/'+camera+'.npz'
    with np.load(caminho_propriedades) as X:
        mtx, dist = [X[i] for i in ('mtx','dist')]
    
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
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    caminho_fotos = './imagens/'+camera+'/*.jpg'
    for i, fname in enumerate(glob.glob(caminho_fotos)):
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9,6),None)
        if ret == True:
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            # project 3D points to image plane
            if np.random.random() < 0.5:
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                img = draw(img,corners2,imgpts)
            else:
                imgpts, jac = cv.projectPoints(axis_cube, rvecs, tvecs, mtx, dist)
                img = cube(img,corners2,imgpts)
            plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            plt.title('Webcam: ' + camera + ' foto: ' + str(i))
            plt.show()
            plt.pause(0.5)
            cv.imwrite(fname[0:-4]+'_vec.png', img)
    print('Imagens salvas em: ' + caminho_fotos)

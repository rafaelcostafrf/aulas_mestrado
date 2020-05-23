from panda3d.core import loadPrcFile
loadPrcFile('./config/conf.prc')

from tabulate import tabulate
import numpy as np
import cv2 as cv
import panda3d
import sys, os
import torch
import time

#Panda 3D Imports
from panda3d.core import Filename
from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from direct.showbase.ShowBase import ShowBase
from scipy.spatial.transform import Rotation as R

#Custom Functions
from environment.quadrotor_env import quad, sensor
from environment.quaternion_euler_utility import quat_euler, euler_quat, quat_rot_mat, deriv_quat
from controller.model import ActorCritic
from controller.dl_auxiliary import dl_in_gen


## ALGORITMO DE TESTE PPO ##
time_int_step = 0.01
max_timesteps = 3000
T = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_POS_DETER = True

# Get the location of the 'py' file I'm running:
mydir = os.path.abspath(sys.path[0])

# Convert that to panda's unix-style notation.
mydir = Filename.fromOsSpecific(mydir).getFullpath()

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.calibrated = False
        self.disableMouse()
        self.time_total_img = []
        self.time_total_sens = []
        
        # ENV SETUP
        self.env = quad(time_int_step, max_timesteps, direct_control=1, T=T)
        self.sensor = sensor(self.env)
        self.aux_dl = dl_in_gen(T, self.env.state_size, self.env.action_size)    
        self.error = []
        state_dim = self.aux_dl.deep_learning_in_size
        self.policy = ActorCritic(state_dim, action_dim=4, action_std=0).to(device)
        
        #CONTROLLER SETUP
        try:
            self.policy.load_state_dict(torch.load('./controller/PPO_continuous_solved_drone.pth',map_location=device))
            print('Saved policy loaded')
        except:
            print('Could not load policy')
            sys.exit(1)
            
        n_parameters = sum(p.numel() for p in self.policy.parameters())
        print('Neural Network Number of Parameters: %i' %n_parameters)

        # Load the environment model.
        self.scene = self.loader.loadModel(mydir + "/models/city.egg")
        self.scene.reparentTo(self.render)
        self.scene.setScale(1, 1, 1)
        self.scene.setPos(0, 0, 0)
        
        # Load the skybox
        self.skybox = self.loader.loadModel(mydir + "/models/skybox.egg")
        self.skybox.setScale(100,100,100)
        self.skybox.setPos(0,0,-500)
        self.skybox.reparentTo(self.render)

        # Also add an ambient light and set sky color.
        skycol = panda3d.core.VBase3(135 / 255.0, 206 / 255.0, 235 / 255.0)
        self.set_background_color(skycol)
        alight = AmbientLight("sky")
        alight.set_color(panda3d.core.VBase4(skycol * 0.04, 1))
        alight_path = render.attachNewNode(alight)
        render.set_light(alight_path)

        # 4 perpendicular lights (flood light)
        dlight1 = DirectionalLight('directionalLight')
        dlight1.setColor(panda3d.core.Vec4(0.3,0.3,0.3,0.3))
        dlight1NP = render.attachNewNode(dlight1)
        dlight1NP.setHpr(0,0,0)

        dlight2 = DirectionalLight('directionalLight')
        dlight2.setColor(panda3d.core.Vec4(0.3,0.3,0.3,0.3))
        dlight2NP = render.attachNewNode(dlight2)
        dlight2NP.setHpr(-90,0,0)

        dlight3 = DirectionalLight('directionalLight')
        dlight3.setColor(panda3d.core.Vec4(0.3,0.3,0.3,0.3))
        dlight3NP = render.attachNewNode(dlight3)
        dlight3NP.setHpr(-180,0,0)

        dlight4 = DirectionalLight('directionalLight')
        dlight4.setColor(panda3d.core.Vec4(0.3,0.3,0.3,0.3))
        dlight4NP = render.attachNewNode(dlight4)
        dlight4NP.setHpr(-270,0,0)
        render.setLight(dlight1NP)
        render.setLight(dlight2NP)
        render.setLight(dlight3NP)
        render.setLight(dlight4NP)

        # 1 directional light (Sun)
        dlight = DirectionalLight('directionalLight')
        dlight.setColor(panda3d.core.Vec4(1, 1, 1, 1)) # directional light is dim green
        dlight.getLens().setFilmSize(panda3d.core.Vec2(50, 50))
        dlight.getLens().setNearFar(-100, 100)
        dlight.setShadowCaster(True, 4096*2, 4096*2)
        # dlight.show_frustum()
        dlightNP = render.attachNewNode(dlight)
        dlightNP.setHpr(0,-65,0)
        #Turning shader and lights on
        render.setShaderAuto()
        render.setLight(dlightNP)



        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.calibrate, 'Camera Calibration')
        self.taskMgr.add(self.drone_position_task, 'Drone Position')
        

        # Load and transform the quadrotor actor.
        self.quad = self.loader.loadModel(mydir + '/models/quad.egg')
        self.quad.reparentTo(self.render)
        self.prop_1 = self.loader.loadModel(mydir + '/models/prop.egg')
        self.prop_1.setPos(-0.26,0,0)
        self.prop_1.reparentTo(self.quad)
        self.prop_2 = self.loader.loadModel(mydir + '/models/prop.egg')
        self.prop_2.setPos(0,0.26,0)
        self.prop_2.reparentTo(self.quad)
        self.prop_3 = self.loader.loadModel(mydir + '/models/prop.egg')
        self.prop_3.setPos(0.26,0,0)
        self.prop_3.reparentTo(self.quad)
        self.prop_4 = self.loader.loadModel(mydir + '/models/prop.egg')
        self.prop_4.setPos(0,-0.26,0)
        self.prop_4.reparentTo(self.quad)
        
        # Load the checkerboard actor
        self.checker = self.loader.loadModel(mydir+ '/models/checkerboard.egg')
        self.checker.reparentTo(self.render)
        self.checker_scale = 0.5
        self.checker_sqr_size = 0.2046
        self.checker.setScale(self.checker_scale, self.checker_scale, 1)
        self.checker.setPos(3*self.checker_scale*self.checker_sqr_size, 2.5*self.checker_scale*self.checker_sqr_size, 0.001)
        
        #env cam
        self.cam.node().getLens().setFilmSize(36, 24)
        self.cam.node().getLens().setFocalLength(45)
        self.cam.setPos(5,5,7)
        self.cam.reparentTo(self.render)
        self.cam.lookAt(self.quad)
        
        if IMG_POS_DETER:
            self.taskMgr.add(self.pos_deter, 'Position Determination')
            window_size = (self.win.getXSize(), self.win.getYSize())     
            self.buffer = self.win.makeTextureBuffer('Buffer', *window_size, None, True)
            self.tex = self.buffer.getTexture()
            self.cam_1 = self.makeCamera(self.buffer)
            self.cam_1.setName('cam_1')     
            
            self.cam_1.node().getLens().setFilmSize(36, 24)
            self.cam_1.node().getLens().setFocalLength(45)
            self.cam_1.reparentTo(self.quad)
            self.cam_1.setPos(0,0,0.01)
            self.cam_1.lookAt(self.quad)
            
            self.pipe = panda3d.core.GraphicsPipeSelection.get_global_ptr().make_module_pipe('pandagl')

    

        
    def drone_position_task(self, task):
        if self.calibrated:
            if task.frame == 0 or self.env.done:
                # in_state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
                in_state = None
                states, action = self.env.reset(in_state)
                self.network_in = self.aux_dl.dl_input(states, action)
                self.sensor.reset()
                pos = self.env.state[0:5:2]
                ang = self.env.ang
                self.a = np.zeros(4)
            else:
                action = self.policy.actor(torch.FloatTensor(self.network_in).to(device)).cpu().detach().numpy()
                states, _, done = self.env.step(action)
                time_iter = time.time()
                _, self.velocity_accel, self.pos_accel = self.sensor.accel_int()
                self.quaternion_gyro = self.sensor.gyro_int()
                self.ang_vel = self.sensor.gyro()
                quaternion_vel = deriv_quat(self.ang_vel, self.quaternion_gyro)
                self.pos_gps, self.vel_gps = self.sensor.gps()
                self.quaternion_triad, _ = self.sensor.triad()
                self.time_total_sens.append(time.time() - time_iter)
                #SENSOR CONTROL
                pos_vel = np.array([self.pos_accel[0], self.velocity_accel[0],
                                    self.pos_accel[1], self.velocity_accel[1],
                                    self.pos_accel[2], self.velocity_accel[2]])
                
                states_sens = [np.concatenate((pos_vel, self.quaternion_gyro, quaternion_vel))]
                
                self.network_in = self.aux_dl.dl_input(states_sens, [action])
                
                error_i = np.concatenate((self.env.state[0:5:2]-self.pos_accel, self.env.state[6:10]-self.quaternion_gyro))
                self.error.append(error_i)   
                if len(self.error) > 1e4:
                    error_final = np.array(self.error)
                    # np.savez('./data/hover_img_det_results', error_final)
                    time_total_img = np.mean(np.array(self.time_total_img))
                    time_total_sens = np.mean(np.array(self.time_total_sens))
                    print('Iteration Time: Img: %.5fms Sens: %.5fms' %(time_total_img*1000, time_total_sens*1000))
                    sys.exit()
                pos = self.env.state[0:5:2]
                ang = self.env.ang
                for i, w_i in enumerate(self.env.w):
                    self.a[i] += (w_i*time_int_step )*180/np.pi/30
        
            ang_deg = (ang[2]*180/np.pi, ang[0]*180/np.pi, ang[1]*180/np.pi)
            pos = (0+pos[0], 0+pos[1], 5+pos[2])
            
            self.quad.setHpr(*ang_deg)
            self.quad.setPos(*pos)
            self.cam.lookAt(self.quad)
            # self.quad.setHpr(0, -45, 0)
            # self.quad.setPos(0, 5, 6)
            for v in self.env.w:
                if v<0:
                    print('negativo')
            self.prop_1.setHpr(self.a[0],0,0)
            self.prop_2.setHpr(self.a[1],0,0)
            self.prop_3.setHpr(self.a[2],0,0)
            self.prop_4.setHpr(self.a[3],0,0)
        return task.cont
    
    def get_image(self):
        tex = self.buffer.getTexture()
        img = tex.getRamImage()
        image = np.frombuffer(img, np.uint8)
        if len(image) > 0:
            image = np.reshape(image, (tex.getYSize(), tex.getXSize(), 4))
            # image = np.flipud(image)
            # image = cv.resize(image, (640, 360))
            return True, image
        else:
            return False, None
   
    def calibrate(self, task):
        if task.frame == 0:
            self.fast = cv.FastFeatureDetector_create()
            self.fast.setThreshold(20)
            self.criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 1, 0.0001) 
            self.nCornersCols = 9
            self.nCornersRows = 6
            self.objp = np.zeros((self.nCornersCols*self.nCornersRows, 3), np.float32)
            self.objp[:,:2] = (np.mgrid[0:self.nCornersCols, 0:self.nCornersRows].T.reshape(-1,2))*self.checker_scale*self.checker_sqr_size
            try: 
                npzfile = np.load('./config/camera_calibration.npz')
                self.mtx = npzfile[npzfile.files[0]]
                self.dist = npzfile[npzfile.files[1]]
                self.calibrated = True
                print('Calibration File Loaded')
                return task.done
            except:
                print('Could Not Load Calibration File, Calibrating... ')
                self.calibrated = False
                self.quad.setPos(10,10,10)
                self.cam_pos = []
                self.objpoints = []
                self.imgpoints = []
                
        rand_pos = (np.random.random(3)-0.5)*5
        rand_pos[2] = np.random.random()*3+2
        cam_pos = tuple(rand_pos)
        self.cam.reparentTo(self.render)
        self.cam_1.reparentTo(self.render)
        self.cam.setPos(*cam_pos)
        self.cam.lookAt(self.checker)
        self.cam_1.setPos(*cam_pos)
        self.cam_1.lookAt(self.checker)
        ret, image = self.get_image()
        if ret:
            img = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
            self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                
            ret, corners = cv.findChessboardCorners(self.gray, (self.nCornersCols, self.nCornersRows), 
                                                    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                self.objpoints.append(self.objp)             
                # corners2 = cv.cornerSubPix(self.gray, corners, (1, 1), (-1, -1), self.criteria)
                self.imgpoints.append(corners)
                img = cv.drawChessboardCorners(img, (self.nCornersCols, self.nCornersRows), corners, ret)
                cv.imshow('img',img)

        if len(self.objpoints) > 70:
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
            if ret:
                h,  w = img.shape[:2]
                
                newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
                dist = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) #Camera Perfeita (Simulada), logo não há distorção
                dst = cv.undistort(img, mtx, dist, None, newcameramtx)    
                cv.imshow('img', dst)
                self.mtx = mtx
                self.dist = dist
                print('Calibration Complete')
                self.calibrated = True
                np.savez('./config/camera_calibration', mtx, dist)
                print('Calibration File Saved')
                self.cam.reparentTo(self.quad)
                self.cam.setPos(0,0,0.01)
                self.cam_1.reparentTo(self.quad)
                self.cam_1.setPos(0,0,0.01)
                return task.done                
            else:
                return task.cont
        else:
            return task.cont  
    
    def tabulate_gen(self, real, image, accel, gps, gyro, triad):
        data = []
        header = ['---', 'State', 'Image State', 'Accelerometer State', 'GPS State', 'Gyro State', 'Triad State']
        data_name = ['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3']

        for i in range(3):
            data.append((data_name[i], str(real[i]), str(image[i]), str(accel[i]), str(gps[i]), "0", "0"))
        
        for i in range(4):
            data.append((data_name[i+3], str(real[i+3]), str(image[i+3]), "0", "0", str(gyro[i]), str(triad[i])))
            
        return data, header
    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 1)
        img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 1)
        img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 1)
        return img    
    
    def pos_deter(self, task):
        time_iter = time.time()
        if self.calibrated and task.frame % 10 == 0 and self.env.i > T:           
            ret, image = self.get_image()
            cv.imshow('img', image)
            if ret:
                img = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                fast_gray = cv.resize(gray, None, fx=1, fy=1)
                corner_good = self.fast.detect(fast_gray)
                if len(corner_good) > 83:
                    ret, corners = cv.findChessboardCorners(img, (self.nCornersCols, self.nCornersRows),
                                                            cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE)
                    
                    if ret:
                        # corners2 = cv.cornerSubPix(self.gray, corners, (1, 1), (-1, -1), self.criteria)
                        ret, rvecs, tvecs = cv.solvePnP(self.objp, corners, self.mtx, self.dist)
                        
                        if ret:
                            axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
                            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)
                            imgpts = imgpts.astype(np.int)
    
                            real_state = np.concatenate((self.env.state[0:5:2], self.env.state[6:10]))
                            rvecs[2] *= -1 
                            r = R.from_rotvec(rvecs.flatten()).inv()
                            euler = r.as_euler('zyx')
                            r = R.from_euler('zyx', -euler)
                            quaternion = r.as_quat()   
                            quaternion = np.concatenate(([quaternion[3]],quaternion[0:3]))
                            trans = np.dot(r.as_matrix().T, tvecs).flatten() 
                            trans[0] *= -1
                            trans[1] *= -1
                            trans[2] += -5.01
                            self.image_pos = trans
                            self.image_quat = quaternion
                            self.sensor.position_t0 = self.image_pos
                            self.sensor.quaternion_t0 = self.sensor.quaternion_t0
                            image_state = np.concatenate((trans.flatten(), quaternion.flatten()))                    
                            data, header = self.tabulate_gen(real_state, image_state, self.pos_accel, self.pos_gps, self.quaternion_gyro, self.quaternion_triad)
                            print(tabulate(data, headers = header, numalign='center', stralign='center', floatfmt='.3f'))
                            print('\n')
                            self.draw(img, corners, imgpts)
                            cv.imshow('img',img)
        self.time_total_img.append(time.time()-time_iter)
        return task.cont
    
app = MyApp()
app.run()

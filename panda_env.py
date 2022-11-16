import time
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import cv2

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class PandaSim():
  def __init__(self, bullet_client, offset, fps):
    self.bullet_client = bullet_client
    self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
    self.offset = np.array(offset)
    
    flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    # self.bullet_client.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)
    # self.bullet_client.loadURDF("urdf/hole/hole.urdf", np.array([0.3, 0.0, 0.0])+self.offset, flags=flags)
    # self.hole = self.bullet_client.loadURDF("urdf/hole_single/hole_single.urdf", np.array([0.5, 0.0, 0.0])+self.offset, flags=flags)
    self.hole = self.bullet_client.loadURDF("urdf/hole_square/hole_square.urdf", np.array([0.7, 0.0, 0.02])+self.offset, flags=flags)
    self.plane = self.bullet_client.loadURDF("plane.urdf", useFixedBase=True)
    self.bullet_client.changeDynamics(self.plane,-1,restitution=.95)
    self.peg = self.bullet_client.loadURDF("urdf/peg_square/peg_square.urdf", np.array([0.5, 0., 0.06])+self.offset, flags=flags) 
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, useFixedBase=True, flags=flags)
    self.state = 0
    self.index = 0 
    self.control_dt = 1./fps
    self.finger_target = 0
    self.gripper_height = 0.2

    self.initial_joints = [-0.038, 0.42, 0.047, -1.50, -0.020, 1.92, 0.797, 0.02, 0.02]
    self.peg_pos, self.peg_orn = self.bullet_client.getBasePositionAndOrientation(self.peg)
    self.hole_pos, self.hole_orn = self.bullet_client.getBasePositionAndOrientation(self.hole)
    self.yaw_hole = 0.    
    self.state_durations = np.array([0.4, 0.6, 0.4, 0.6, 0.6, 200000, 200000])*0.5
    self.t = 0. 
    self.target_joints = []
    self.scan_path = self.generate_tragectory()
    self.move = True 
    self.generate_tra = True 
    self.num_tra = 0
    self.new_orn_peg = [1.,0.,0.,0.]

    #add camera
    self.viewMatrix_1 = self.bullet_client.computeViewMatrix(
                          cameraEyePosition=[0.9, 0.2, 0.2],
                          cameraTargetPosition=[0.7, 0.0, 0.05],
                          cameraUpVector=[0, 0, 1])

    self.viewMatrix_2 = self.bullet_client.computeViewMatrix(
                          cameraEyePosition=[0.9, -0.2, 0.2],
                          cameraTargetPosition=[0.7, 0.0, 0.05],
                          cameraUpVector=[0, 0, 1])

    self.projectionMatrix_1 = self.bullet_client.computeProjectionMatrixFOV(
                          fov=45.0,
                          aspect=1.0,
                          nearVal=0.1,
                          farVal=2.1)

    #create a constraint to keep the fingers centered
    c = self.bullet_client.createConstraint(self.panda,
                       9,
                       self.panda,
                       10,
                       jointType=self.bullet_client.JOINT_GEAR,
                       jointAxis=[1, 0, 0],
                       parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
    
    self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    self.reset()
    

  def reset(self):
    index = 0
    for j in range(self.bullet_client.getNumJoints(self.panda)):
      self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
      info = self.bullet_client.getJointInfo(self.panda, j)
      #print("info=",info)
      # jointName = info[1]
      jointType = info[2]
      if (jointType == self.bullet_client.JOINT_PRISMATIC):
        self.bullet_client.resetJointState(self.panda, j, self.initial_joints[index]) 
        index=index+1
      if (jointType == self.bullet_client.JOINT_REVOLUTE):
        self.bullet_client.resetJointState(self.panda, j, self.initial_joints[index]) 
        index=index+1
    self.t = 0.

  def close_gripper(self):
    # self.bullet_client.submitProfileTiming("Close Gripper")
    for i in [9,10]:
      self.bullet_client.setJointMotorControl2(self.panda, i, \
        self.bullet_client.POSITION_CONTROL, 0.0001 ,force= 50)
    self.bullet_client.submitProfileTiming()

  def open_gripper(self):
    # self.bullet_client.submitProfileTiming("Open Gripper")
    for i in [9,10]:
      self.bullet_client.setJointMotorControl2(self.panda, i, \
        self.bullet_client.POSITION_CONTROL, 0.04 ,force= 50)
    self.bullet_client.submitProfileTiming()

  def set_joints(self,jointPoses, velocity=None):
    # self.bullet_client.submitProfileTiming("set joints")
    if velocity: 
      for i in range(7):
          self.bullet_client.setJointMotorControl2(self.panda, i, \
            self.bullet_client.POSITION_CONTROL, jointPoses[i],targetVelocity = velocity, force=5 * 240., positionGain=0.1,
                                velocityGain=1.0)
    else:
      for i in range(7):
          self.bullet_client.setJointMotorControl2(self.panda, i, \
            self.bullet_client.POSITION_CONTROL, jointPoses[i], force=5 * 240.)
    self.bullet_client.submitProfileTiming()

  def get_IK(self, pos, orn=[1,0,0,0]):
    self.bullet_client.submitProfileTiming("IK")
    jointPoses = self.bullet_client.calculateInverseKinematics(self.panda,11, pos, orn, ll, ul,
        jr, rp, maxNumIterations=100)  
    return jointPoses

  def get_cart(self): 
    # pos_w, quat_w, pos_r, quat_r
    result = self.bullet_client.getLinkState(self.panda, 11, computeLinkVelocity=1) 
    return result[0], result[1], result[-2], result[-1]


  def get_peg_cart(self):
    # self.bullet_client.submitProfileTiming("step")
    pos, o = self.bullet_client.getBasePositionAndOrientation(self.peg)
    return pos, o

  def take_picture(self):
    width, height, rgbImg, depthImg, segImg  = self.bullet_client.getCameraImage(224,224,self.viewMatrix_1,self.projectionMatrix_1)
    width2, height2, rgbImg2, depthImg2, segImg2 = self.bullet_client.getCameraImage(224,224,self.viewMatrix_2,self.projectionMatrix_1)
    # width3, height3, rgbImg3, depthImg3, segImg3 = self.bullet_client.getCameraImage(224,224,self.viewMatrix_3,self.projectionMatrix_1)
    return rgbImg, depthImg, rgbImg2, depthImg2 #, rgbImg3, depthImg3

  def rocking(self):
    self.update_state()

    if self.state == 0:
      current_pos, current_orn, _, _ = self.get_cart() 
      



  def grasp_step(self):
    # step:0) get the ik 1) go to the position 2) grasp 3) lift 
    self.update_state()
    # print(time.time()-t0)

    if self.state == 0: 
      # get the pog postion and open gripper 
      target_pos = [self.peg_pos[0], self.peg_pos[1], self.peg_pos[2] + 0.005]
      euler_angle = R.from_quat(self.peg_orn).as_euler('zyx', degrees=True)
      euler_angle[2] += 0
      target_orn = R.from_euler('zyx', euler_angle, degrees=True).as_quat()
      target_orn_ = [target_orn[3], target_orn[0],target_orn[1],target_orn[2]]
      # print("target_orn", target_pos)
      # print('target pos', target_pos)
      self.target_joints = self.get_IK(target_pos)
      self.open_gripper()
      self.hole_pos, self.hole_orn = self.bullet_client.getBasePositionAndOrientation(self.hole)
      
    elif self.state == 1:
      # go to the peg position
      self.set_joints(self.target_joints) 

    elif self.state == 2:

      self.close_gripper()

    elif self.state == 3:
      # lift up
      target_pos = [self.peg_pos[0], self.peg_pos[1], self.peg_pos[2] + 0.105]
      euler_angle = R.from_quat(self.peg_orn).as_euler('zyx', degrees=True)
      euler_angle[2] += 0
      target_orn = R.from_euler('zyx', euler_angle, degrees=True).as_quat()
      target_orn_ = [target_orn[3], target_orn[0],target_orn[1],target_orn[2]]
      self.target_joints = self.get_IK(target_pos, target_orn_)

    elif self.state == 4:
      self.set_joints(self.target_joints) 
      # print(self.target_joints)


    elif self.state == 5:

      if self.num_tra > 400:
        self.state += 1
      target_pos = [self.hole_pos[0], self.hole_pos[1] , self.peg_pos[2] + 0.045]
      taregt_orn = [np.pi, 0., R.from_quat(self.hole_orn).as_euler('xyz')[2]] 

      # print('target pose', target_pos)
      # self.target_joints = self.get_IK(target_pos)
      # self.set_joints(self.target_joints)

      if self.move: 
        current_pos, current_orn, _, _ = self.get_cart()
        current_orn_eular = R.from_quat(current_orn).as_euler('xyz')

        if current_orn_eular[0] < 0:
          current_orn_eular[0] += np.pi*2

        # print(current_orn_eular)
        # import pdb; pdb.set_trace()
        self.end_effector = np.array(current_pos + current_orn) 
        self.vel = np.array(target_pos) - np.array(current_pos)
        self.vel_max = 0.01
        self.avel_max = np.pi/45.
        self.vel_angle = np.array(taregt_orn) - np.array(current_orn_eular)

        
        if np.max(np.abs(self.vel_angle)) > self.avel_max:
          self.vel_angle = self.vel_angle/np.max(np.abs(self.vel_angle))*self.avel_max
        if np.linalg.norm(self.vel) > self.vel_max:
          self.vel = self.vel/np.linalg.norm(self.vel)*self.vel_max

        target_pos_temp = current_pos + self.vel 
        target_orn_temp_eular = np.array(current_orn_eular) + self.vel_angle
        target_orn_temp = R.from_euler('xyz',target_orn_temp_eular).as_quat()

        self.target_joints = self.get_IK(target_pos_temp,target_orn_temp) 
        self.set_joints(self.target_joints) 
      # print(self.target_joints)

      # reset to a new position 
        if np.linalg.norm(self.vel) < self.vel_max*0.1 and np.max(np.abs(self.vel_angle)) < self.avel_max*0.1:
          self.move = False   

      else:

        if self.generate_tra:
          self.num_tra += 1
          print('num_of_trac', self.num_tra, end=' ')

          ### reset the peg position pose 
          off_len = 0.05

          x_range = [-0.05, 0.05]
          y_range = [-0.05, 0.05]
          z_range = [0.04, 0.1]
          x_off = np.random.random() * (x_range[1] - x_range[0])+ x_range[0]
          y_off = np.random.random() * (y_range[1] - y_range[0])+ y_range[0]
          z_off = np.random.random() * (z_range[1] - z_range[0])+ z_range[0]

          off_angle_rp = 5./180.*np.pi 
          off_angle_y = 30./180.*np.pi 
          rand_r = (np.random.random()-0.5)*off_angle_rp*2 
          rand_p = (np.random.random()-0.5)*off_angle_rp*2 
          rand_y = (np.random.random()-0.5)*off_angle_y*2 
          self.new_orn_peg = self.bullet_client.getQuaternionFromEuler([np.pi+rand_r, rand_p, rand_y])
          self.new_pos_peg = [0.7+x_off, y_off, 0.105+z_off] 

          # self.bullet_client.resetBasePositionAndOrientation(self.hole, [0.7+off_x, 0.+off_y, 0.01], [1,0,0,0]) #
          
          ### reset the hole position pose 
          off_x = (np.random.random()-0.5)*off_len*2
          off_y = (np.random.random()-0.5)*off_len*2
          off_angle_y = 30./180.*np.pi 
          rand_theta_hole = (np.random.random()-0.5)*off_angle_y*2 
          new_orn_hole = self.bullet_client.getQuaternionFromEuler([0.0, 0.0, rand_theta_hole])
          self.yaw_hole = rand_theta_hole
          self.bullet_client.resetBasePositionAndOrientation(self.hole, [0.7+off_x, 0.+off_y, self.hole_pos[2]], new_orn_hole) 
          self.hole_pos, self.hole_orn = self.bullet_client.getBasePositionAndOrientation(self.hole)
          self.generate_tra = False

        self.target_joints = self.get_IK(self.new_pos_peg) #, self.new_orn_peg) 
        self.set_joints(self.target_joints) 
        self.hole_pos, self.hole_orn = self.bullet_client.getBasePositionAndOrientation(self.hole)
        self.hole_pos = np.array(self.hole_pos)
        # self.hole_pos[2] += 0.02
        current_pos, _, _, _ = self.get_cart() 


        if np.linalg.norm(np.array(self.new_pos_peg) - np.array(current_pos)) < self.vel_max*0.1:
          self.move = True    
          self.generate_tra = True 


    elif self.state == 6:  # start scan for keypoints 
      # self.open_gripper()
      target_pos = [self.hole_pos[0] + self.scan_path[self.index][0], self.hole_pos[1] + self.scan_path[self.index][1],\
                                   self.hole_pos[2] + 0.085 + self.scan_path[self.index][2]]

      # print('hole orientation', self.peg_orn)
      current_pos, current_orn, _, _ = self.get_cart() 
      # print('gripper orientation', current_orn)
      # import pdb; pdb.set_trace()

      if self.move: 
        self.target_joints = self.get_IK(target_pos, self.new_orn_peg)
        self.set_joints(self.target_joints)
        self.move = False 


      pos = self.bullet_client.getJointStates(self.panda, range(self.bullet_client.getNumJoints(self.panda)))

      joint_pos = [pos[i][0] for i in range(7)]
      joint_pos = np.array(joint_pos)
      target_pos = np.array(self.target_joints)
      # print('diff', np.linalg.norm(target_pos[:7] - joint_pos))
      if np.linalg.norm(target_pos[:7] - joint_pos) < 0.01:
        self.index += 1
        self.move = True 
        
        off_len = 0.05
        off_angle_rp = 5./180.*np.pi 
        off_angle_y = 30./180.*np.pi 
        off_x = (np.random.random()-0.5)*off_len*2
        off_y = (np.random.random()-0.5)*off_len*2
        rand_r = (np.random.random()-0.5)*off_angle_rp*2 
        rand_p = (np.random.random()-0.5)*off_angle_rp*2 
        rand_y = (np.random.random()-0.5)*off_angle_y*2 
        self.new_orn_peg = self.bullet_client.getQuaternionFromEuler([np.pi+rand_r, rand_p, rand_y])
        # self.new_orn_peg = [self.new_orn_peg[3], self.new_orn_peg[0], self.new_orn_peg[1], self.new_orn_peg[2]]
        # print('new_orn_peg', self.new_orn_peg)

        off_angle_y = 30./180.*np.pi 
        rand_theta_hole = (np.random.random()-0.5)*off_angle_y*2 
        new_orn_hole = self.bullet_client.getQuaternionFromEuler([0.0, 0.0, rand_theta_hole])
        self.bullet_client.resetBasePositionAndOrientation(self.hole, [0.7+off_x, 0.+off_y, self.hole_pos[2]], new_orn_hole) 


        self.hole_pos, self.hole_orn = self.bullet_client.getBasePositionAndOrientation(self.hole)
        self.yaw_hole = rand_theta_hole
        self.hole_pos = np.array(self.hole_pos)
        # self.hole_pos[2] += 0.02

      if self.index == len(self.scan_path):
        self.index = 0 
        


  def generate_tragectory(self):

    # define some grids
    xgrid = np.arange(-5,6,1) 
    ygrid = [-5, 5]
    zgrid = np.arange(0,8,2) 

    scan = []
    for z in zgrid:
      for x in xgrid:
        for y in ygrid:
          scan.append([x*0.01,y*0.01,z*0.01])
    return scan 


  def update_state(self):
    self.t += self.control_dt
    if self.t > self.state_durations[self.state]:
      self.state += 1
      if self.state>=len(self.state_durations):
        self.state = 0
      self.t = 0.
      # self.state=self.states[self.cur_state]


 

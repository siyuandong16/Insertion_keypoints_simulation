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
    self.hole = self.bullet_client.loadURDF("urdf/hole_square/hole_square.urdf", np.array([0.7, 0., 0.02])+self.offset, flags=flags)
    self.plane = self.bullet_client.loadURDF("plane.urdf", useFixedBase=True)
    self.bullet_client.changeDynamics(self.plane,-1,restitution=.95)
    self.peg = self.bullet_client.loadURDF("urdf/peg_square/peg_square.urdf", np.array([0.5, 0., 0.06])+self.offset, flags=flags) 
    self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, useFixedBase=True, flags=flags)
    self.state = 0
    self.control_dt = 1./fps
    self.finger_target = 0
    self.gripper_height = 0.2

    self.initial_joints = [-0.038, 0.42, 0.047, -1.50, -0.020, 1.92, 0.797, 0.02, 0.02]
    self.peg_pos, self.peg_orn = self.bullet_client.getBasePositionAndOrientation(self.peg)
    self.hole_pos, self.hole_orn = self.bullet_client.getBasePositionAndOrientation(self.hole)
    self.state_durations = np.array([0.4, 0.6, 0.4, 0.6, 0.6, 0.4, 0.6, 0.6, 0.4])*0.5
    self.t = 0.;
    self.target_joints = []
    self.offset_2 = [0.0, 0.0, 0.005]
    # self.offset_3 = [-0.02, -0.02, 0.09]
    self.offset_3 = [0.00, 0.00, 0.05]

    #add camera
    self.viewMatrix_1 = self.bullet_client.computeViewMatrix(
                          cameraEyePosition=[0.9, 0.2, 0.2],
                          cameraTargetPosition=[0.6, 0, 0],
                          cameraUpVector=[0, 0, 1])
    self.viewMatrix_2 = self.bullet_client.computeViewMatrix(
                          cameraEyePosition=[0.9, -0.2, 0.2],
                          cameraTargetPosition=[0.6, 0, 0],
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
            self.bullet_client.POSITION_CONTROL, jointPoses[i],targetVelocity = velocity, force=5 * 240.)
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

  def get_peg_cart(self):
    # self.bullet_client.submitProfileTiming("step")
    pos, o = self.bullet_client.getBasePositionAndOrientation(self.peg)
    return pos, o

  def take_picture(self):
    width, height, rgbImg, depthImg, segImg  = self.bullet_client.getCameraImage(224,224,self.viewMatrix_1,self.projectionMatrix_1)
    width2, height2, rgbImg2, depthImg2, segImg2 = self.bullet_client.getCameraImage(224,224,self.viewMatrix_2,self.projectionMatrix_1)
    return rgbImg, depthImg, rgbImg2, depthImg2

  def grasp_step(self):
    # step:0) get the ik 1) go to the position 2) grasp 3) lift 
    self.update_state()
    # print(time.time()-t0)

    if self.state == 0: 
      # get the pog postion and open gripper 
      target_pos = [self.peg_pos[0]+self.offset_2[0], self.peg_pos[1]+self.offset_2[1], self.peg_pos[2] +self.offset_2[2]]
      euler_angle = R.from_quat(self.peg_orn).as_euler('zyx', degrees=True)
      euler_angle[2] += 0
      target_orn = R.from_euler('zyx', euler_angle, degrees=True).as_quat()
      target_orn_ = [target_orn[3], target_orn[0],target_orn[1],target_orn[2]]
      # print("target_orn", target_pos)
      # print('target pos', target_pos)
      self.target_joints = self.get_IK(target_pos)
      self.open_gripper()
      
    elif self.state == 1:
      # go to the peg position
      self.set_joints(self.target_joints) 

    elif self.state == 2:
      self.close_gripper()

    elif self.state == 3:
      # lift up
      target_pos = [self.peg_pos[0]+self.offset_2[0], self.peg_pos[1]+self.offset_2[1], \
              self.peg_pos[2] +self.offset_2[2] + 0.1]
      euler_angle = R.from_quat(self.peg_orn).as_euler('zyx', degrees=True)
      euler_angle[2] += 0
      target_orn = R.from_euler('zyx', euler_angle, degrees=True).as_quat()
      target_orn_ = [target_orn[3], target_orn[0],target_orn[1],target_orn[2]]
      self.target_joints = self.get_IK(target_pos, target_orn_)

    elif self.state == 4:
      self.set_joints(self.target_joints) 
      # print(self.target_joints)


    elif self.state == 5:
      target_pos = [self.hole_pos[0] + self.offset_3[0], self.hole_pos[1] + self.offset_3[1],\
                                         self.peg_pos[2] + self.offset_3[2]]
      euler_angle = R.from_quat(self.peg_orn).as_euler('zyx', degrees=True)
      euler_angle[2] += 0
      target_orn = R.from_euler('zyx', euler_angle, degrees=True).as_quat()
      target_orn_ = [target_orn[3], target_orn[0],target_orn[1],target_orn[2]]
      self.target_joints = self.get_IK(target_pos, target_orn_)

    elif self.state == 6:
      self.set_joints(self.target_joints) 
      # print(self.target_joints)

    elif self.state == 7:
      # input("press Enter")
      target_pos = [self.hole_pos[0] + self.offset_3[0], self.hole_pos[1] + self.offset_3[1],\
                                         self.hole_pos[2] + self.offset_3[2] -0.015]
      self.target_joints = self.get_IK(target_pos)
      self.set_joints(self.target_joints) #, 1.0) 
      # print(self.target_joints)
    elif self.state == 8:
      self.open_gripper()



  def update_state(self):
    self.t += self.control_dt
    if self.t > self.state_durations[self.state]:
      self.state += 1
      if self.state>=len(self.state_durations):
        self.state = 0
      self.t = 0.
      # self.state=self.states[self.cur_state]


 
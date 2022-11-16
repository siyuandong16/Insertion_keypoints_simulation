import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
# import pybullet_robots.panda.panda_sim_grasp as panda_sim
import panda_env as panda_sim
import cv2
import copy 
from utils import * 
#video requires ffmpeg available in path
collect_data = True         
createVideo=False
fps=240.
timeStep = 1./fps

if createVideo:
	p.connect(p.GUI, options="--minGraphicsUpdateTimeMs=0 --mp4=\"pybullet_grasp.mp4\" --mp4fps="+str(fps) )
else:
	p.connect(p.GUI)


p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=-300, cameraPitch=-30, cameraTargetPosition=[0.7,0.0,0.0])
p.setAdditionalSearchPath(pd.getDataPath())   #-22

p.setTimeStep(timeStep)
p.setGravity(0,0,-9.8)

panda = panda_sim.PandaSim(p,[0,0,0],fps)
panda.control_dt = timeStep


logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")
panda.bullet_client.submitProfileTiming("start")


offset_peg1 = np.array([[0.019, 0.02, -0.038], 
					  [0.019, -0.018, -0.038],
					  [-0.019, 0.02, -0.038]
					  ]) 

offset_peg2 = np.array([[0.02, 0.02, -0.038], 
					  [0.02, -0.02, -0.038],
					  [-0.02, -0.02, -0.038]
					  ])  

# key_peg_list = [] 
# key_hole_list = []

# print('peg posiiton', panda.peg_pos, panda.peg_orn)

time.sleep(1)

for i in range (50000, 120000):
	panda.bullet_client.submitProfileTiming("full_step")

	panda.grasp_step()
	# if i % 3 == 0 and panda.state == 5 and panda.move and collect_data: 
		# rgb, depth, rgb2, depth2 = panda.take_picture()
		# rgb = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
		# rgb2 = cv2.cvtColor(np.array(rgb2), cv2.COLOR_RGB2BGR)
		# rgb_save1 = copy.deepcopy(rgb)
		# rgb_save2 = copy.deepcopy(rgb2)
		# peg_pos, peg_orn = panda.get_peg_cart()
		# hole_pos = np.array(panda.hole_pos)
		# hole_orn = panda.yaw_hole
		#detect key points 

		# rm_peg = rotation_matrix(peg_orn, "quat")

		# peg_pos_img1, hole_pos_img1 = key_detector(np.array([0.9, 0.2, 0.2]), peg_pos, hole_pos, hole_orn, rgb, depth, offset_peg1.dot(rm_peg))
		# peg_pos_img2, hole_pos_img2 = key_detector(np.array([0.9, -0.2, 0.2]), peg_pos, hole_pos, hole_orn, rgb2, depth2, offset_peg2.dot(rm_peg))
		#label key points 
		# img2show1 = show_keypoint(rgb, peg_pos_img1, hole_pos_img1)
		# img2show2 = show_keypoint(rgb2, peg_pos_img2, hole_pos_img2)
		# cv2.imwrite("rgb_left.jpg", rgb)
		# cv2.imwrite("rgb_right.jpg", rgb2)

		# cv2.imwrite('data_policy/img1_'+str(i)+'.jpg', rgb_save1)
		# cv2.imwrite('data_policy/img2_'+str(i)+'.jpg', rgb_save2)
		# key_peg_list.append(peg_pos_img1)
		# key_hole_list.append(hole_pos_img1)
		# np.savez('data_policy/keypoint'+str(i)+'.npz', peg1 =peg_pos_img1, hole1 = hole_pos_img1, peg2 =peg_pos_img2, hole2 = hole_pos_img2,\
		# 	vel = panda.vel, avel = panda.vel_angle, end_effector = panda.end_effector)
		# np.savez('data_key/keypoint1'+str(i)+'.npz', peg =peg_pos_img1, hole = hole_pos_img1)
		# np.savez('data_key/keypoint2'+str(i)+'.npz', peg =peg_pos_img2, hole = hole_pos_img2)

		# cv2.imshow('rgb', img2show1)
		# cv2.imshow('rgb2', img2show2)
		# cv2.imshow('depth', depth3)
		# cv2.waitKey(1)
		# plt.imshow(cv2.cvtColor(img2show1, cv2.COLOR_RGB2BGR))
		# plt.show()
		# pass
	# print(target_joints)
	p.stepSimulation()
	# time.sleep(100)
	if createVideo:
		p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
	if not createVideo:
		time.sleep(timeStep)
	panda.bullet_client.submitProfileTiming()
panda.bullet_client.submitProfileTiming()
panda.bullet_client.stopStateLogging(logId)
	

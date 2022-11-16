import numpy as np 
import cv2 
import copy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def key_detector(camera_pos, peg_pos, hole_pos, hole_orn, img, depth, offset_peg):	

	img_size = 224 
	FOV = 45/180*np.pi 
	f = img_size/2/np.tan(FOV/2)
	intrinsic = np.array([[f, 0, 0],[0, f, 0], [0, 0, 1]])

	target_pos = np.array([0.7, 0.0, 0.05])
	up_direction = np.array([0, 0, 1]) 
	L = target_pos - camera_pos
	L = L/np.linalg.norm(L)
	s = np.cross(L,up_direction)
	s = s/np.linalg.norm(s)
	# u = np.cross(s, L)
	u = np.cross(L, s)
	R = np.vstack((s, u, -L))
	# R = R.dot(np.array([[-1,0,0],[0,-1,0],[0,0,1]]))
	T = -R.dot(camera_pos)

	rm = rotation_matrix(hole_orn,"yaw") 
	offset_hole = np.array([[0.022, 0.022, 0.007], 
					  [0.022, -0.022, 0.007],
					  [-0.022, 0.022, 0.007],
					  [-0.022, -0.022, 0.007]])  
	offset_hole = offset_hole.dot(rm)
	# import pdb; pdb.set_trace()

	p_im_list = []
	h_im_list = [] 
	for i in range(3):
		p_im = get_coor(peg_pos, offset_peg[i,:], intrinsic, R, T, img_size)
		# if visual_check('peg', p_im[:2], img):
		p_im_list.append(p_im[:2])

	for i in range(4):
		h_im = get_coor(hole_pos, offset_hole[i,:], intrinsic, R, T, img_size)
		if visual_check('hole', h_im[:2], img):
			h_im_list.append(h_im[:2])
		else:
			h_im_list.append([0., 0.])

	# valid(img, depth, p_im_list, h_im_list, peg_pos, offset_peg, offset_hole, target_pos, camera_pos)
	return p_im_list, h_im_list

def get_coor(obj_pos, offset, intrinsic, R, T, img_size):
	obj_pos_temp = copy.deepcopy(obj_pos)
	obj_pos_temp += offset
	p_im = intrinsic.dot(R.dot(obj_pos_temp.T)+T)
	p_im = p_im/p_im[2]
	p_im = np.array([img_size/2-p_im[0], img_size/2-p_im[1]])
	return p_im


def rotation_matrix(angle, input_type):
	if input_type == "yaw":
		rm = np.array([[np.cos(angle), np.sin(angle), 0],[-np.sin(angle), np.cos(angle), 0],[0,0,1]])
	elif input_type == "quat":
		# rm = R.from_quat(angle).as_matrix()
		eular_angle = R.from_quat(angle).as_euler('xyz', degrees=True)
		eular_angle *= -1
		rm = R.from_euler('xyz', eular_angle, degrees = True).as_matrix() 
		# import pdb; pdb.set_trace()
	return rm 


def visual_check(obj, coor, img):

	if coor[0] >= img.shape[0] or coor[1] >= img.shape[1] or coor[0] <0 or coor[0] <0:
		return False

	if obj == 'peg':
		if img[int(coor[1]), int(coor[0]), 1] > 100:
			return True 
		else:
			return False
	else:
		if img[int(coor[1]), int(coor[0]), 1] > 100 and img[int(coor[1]), int(coor[0]), 0] < 100:
			return False 
		else:
			return True  

# def visual_check(space_pos, img_pos, camera_pos, img, intrinsic, R, T):

# 	p2c = camera_pos - space_pos
# 	p2c_norm = p2c/np.linalg.norm(p2c)
# 	#forward 0.01 m  
# 	new_p = space_pos + p2c_norm*0.01 
# 	new_p_im = intrinsic.dot(R.dot(new_p.T)+T)
# 	new_p_im = new_p_im/new_p_im[2]
# 	return 




def show_keypoint(img, key_list1, key_list2):

	#peg 
	for key in key_list1:
		img = cv2.circle(img, (int(key[0]),int(key[1])), radius=4, color =(0,225,0) , thickness=-1)
		# cv2.imshow('img', img)
		# cv2.waitKey(0)

	for key in key_list2:
		img = cv2.circle(img, (int(key[0]),int(key[1])), radius=4, color =(255,0,0) , thickness=-1)

	return img


def valid(img, depth, key_peg_list1, key_hole_list1, peg_pos, offset_peg, offset_hole, target_pos, camera_pos):
	nearVal = 0.1
	farVal = 2.1
	depth = farVal * nearVal / (farVal - (farVal - nearVal) * depth)

	depth_peg_key = []
	depth_hole_key = []
	for i in range(4):
		depth_peg_key.append(depth[int(key_peg_list1[i][1]), int(key_peg_list1[i][0])]) 
		depth_hole_key.append(depth[int(key_hole_list1[i][1]), int(key_hole_list1[i][0])]) 

	depth_stack = np.dstack((depth, depth, depth))
	for i in range(4):
		peg_pos_temp = copy.deepcopy(peg_pos)
		peg_pos_temp = peg_pos_temp + offset_peg[i,:]
		v_pc = peg_pos_temp-camera_pos
		v_tc = target_pos - camera_pos
		dist = np.abs(v_tc.dot(v_pc.T)/np.linalg.norm(v_tc))
		
		print("depth detect", depth_peg_key[i], "depth est", dist, 'camera dis', np.linalg.norm(v_pc))

		img = cv2.circle(img, (int(key_peg_list1[i][0]),int(key_peg_list1[i][1])), radius=2, color =(0,225,0) , thickness=-1)

		depth_stack = cv2.circle(depth_stack, (int(key_peg_list1[i][0]),int(key_peg_list1[i][1])), radius=2, color =(0,225,0) , thickness=-1)
		cv2.imshow('img', img)
		# cv2.imshow('depth', depth_stack)
		cv2.waitKey(0)
		# plt.imshow(depth_stack)
		# plt.show()

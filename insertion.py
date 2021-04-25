import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
# import pybullet_robots.panda.panda_sim_grasp as panda_sim
import panda_env as panda_sim

#video requires ffmpeg available in path
createVideo=False
fps=240.
timeStep = 1./fps

if createVideo:
	p.connect(p.GUI, options="--minGraphicsUpdateTimeMs=0 --mp4=\"pybullet_grasp.mp4\" --mp4fps="+str(fps) )
else:
	p.connect(p.GUI)

# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=-300, cameraPitch=-30, cameraTargetPosition=[1.0,0.0,0.0])
p.setAdditionalSearchPath(pd.getDataPath())   #-22

p.setTimeStep(timeStep)
p.setGravity(0,0,-9.8)

panda = panda_sim.PandaSim(p,[0,0,0],fps)
panda.control_dt = timeStep


logId = panda.bullet_client.startStateLogging(panda.bullet_client.STATE_LOGGING_PROFILE_TIMINGS, "log.json")
panda.bullet_client.submitProfileTiming("start")

# print('peg posiiton', panda.peg_pos, panda.peg_orn)
time.sleep(1)
for i in range (100000):
	panda.bullet_client.submitProfileTiming("full_step")

	panda.grasp_step()
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
	

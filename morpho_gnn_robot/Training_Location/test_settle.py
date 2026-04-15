import sys
import numpy as np
import pybullet as p
sys.path.append("/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location")
from robot_env_bullet import RobotEnvBullet

URDF = "/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location/anymal_stripped.urdf"

env = RobotEnvBullet(URDF, render_mode=None)

# Patch the internal _load_robot to test start_pos = 0.45
old_load = env._load_robot
def custom_load():
    if env._robot_id is not None:
        p.removeBody(env._robot_id)
    start_pos    = [0.0, 0.0, 0.45] # Mid air!
    start_orient = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
    env._robot_id = p.loadURDF(env.urdf_path, start_pos, start_orient, useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION)
    
    num_joints = p.getNumJoints(env._robot_id)
    env._joint_idx = {}
    env.foot_indices = []
    env.heavy_indices = []
    
    for i in range(num_joints):
        info = p.getJointInfo(env._robot_id, i)
        name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8")
        if "_FOOT" in link_name: env.foot_indices.append(i)
        elif "HIP" in link_name or "THIGH" in link_name: env.heavy_indices.append(i)
        
        if name in env.joint_names:
            env._joint_idx[name] = i
            p.setJointMotorControl2(env._robot_id, i, p.VELOCITY_CONTROL, force=0)

    env._pybullet_indices = np.array([env._joint_idx[n] for n in env.joint_names], dtype=np.int32)
    from robot_env_bullet import NOMINAL_POSE_PER_JOINT
    env._nominal_pos = np.array([NOMINAL_POSE_PER_JOINT.get(n, 0.0) for n in env.joint_names])
    
    for name, idx in env._joint_idx.items():
        p.resetJointState(env._robot_id, int(idx), NOMINAL_POSE_PER_JOINT.get(name, 0.0))

env._load_robot = custom_load

old_reset = env.reset
def patched_reset(seed=None, options=None):
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 400.0)
    p.setPhysicsEngineParameter(numSolverIterations=50, numSubSteps=1)
    plane_id = p.loadURDF("plane.urdf")
    env._load_robot()
    # NO SETTLE LOOP!
    env._step_count = 0
    env._fell       = False
    env.prev_action = np.zeros(env.action_dim)
    env._prev_pos   = np.array(p.getBasePositionAndOrientation(env._robot_id)[0])
    env.action_history = [np.zeros(12, dtype=np.float32)] * 2
    return env._get_obs(), {}

env.reset = patched_reset

obs, info = env.reset()

contact_termination = False
contacts = p.getContactPoints(bodyA=env._robot_id)
for contact in contacts:
    link_idx = contact[3]
    if link_idx == -1: contact_termination = True
    elif link_idx not in env.foot_indices and link_idx in env.heavy_indices: contact_termination = True

base_pos, _ = p.getBasePositionAndOrientation(env._robot_id)
print(f"Skipped Settle from 0.45m: base_height={base_pos[2]:.4f}, contact_termination={contact_termination}")
env.close()

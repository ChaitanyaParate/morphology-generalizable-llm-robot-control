import sys
import time
import numpy as np
import pybullet as p
from robot_env_bullet import RobotEnvBullet

def test_stand():
    env = RobotEnvBullet(
        urdf_path="/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location/hexapod_anymal.urdf",
        render_mode=None
    )
    obs, info = env.reset()
    print("Initial base height:", info.get("base_height"))
    
    for i in range(100):
        # 0 action = nominal pose
        action = np.zeros(env.action_space.shape[0])
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"Terminated at step {i}, Reason: {info.get('term_reason')}, Height: {info.get('base_height'):.3f}")
            break
        if i % 10 == 0:
            print(f"Step {i}: Height = {info.get('base_height'):.3f}")
            
if __name__ == "__main__":
    test_stand()

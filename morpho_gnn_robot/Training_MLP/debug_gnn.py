import sys
sys.path.insert(0, '/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location')
from robot_env_bullet import RobotEnvBullet
env = RobotEnvBullet('anymal_stripped.urdf', max_episode_steps=200, render_mode=None)
obs, _ = env.reset()
for i in range(5):
    action = env.action_space.sample() * 0.1
    obs, reward, term, trunc, info = env.step(action)
    print(f"step {i}: term={term}, reason={info.get('term_reason', 'none')}, base_height={info.get('base_height', 0):.3f}")
    if term:
        break

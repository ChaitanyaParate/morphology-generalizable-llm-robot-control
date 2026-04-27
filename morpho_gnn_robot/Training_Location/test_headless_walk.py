import torch
import numpy as np
from torch_geometric.data import Batch
from train_gnn_ppo import RunningNorm
from gnn_actor_critic import GNNActorCritic
from urdf_to_graph import URDFGraphBuilder
from robot_env_bullet import RobotEnvBullet

def test_walk():
    device = torch.device('cpu')
    checkpoint = torch.load('checkpoints/multi/gnn_ppo_1961984.pt', map_location=device, weights_only=False)
    builder = URDFGraphBuilder('anymal_stripped.urdf', add_body_node=True)
    agent = GNNActorCritic(node_dim=builder.node_dim, edge_dim=builder.edge_dim, hidden_dim=48, num_joints=builder.action_dim).to(device)
    agent.load_state_dict(checkpoint['agent'])
    agent.eval()

    obs_norm = RunningNorm(shape=(30,))
    obs_norm.mean = checkpoint['obs_norm_mean']
    obs_norm.var = checkpoint['obs_norm_var']

    env = RobotEnvBullet('anymal_stripped.urdf', render_mode=None)
    obs, _ = env.reset()
    target_cmd = np.array([1.0, 0.0], dtype=np.float32)
    env.command = target_cmd
    
    for step in range(100):
        joint_pos = obs[:12]
        joint_vel = obs[12:24]
        body_lin_vel = obs[24:27]
        body_ang_vel = obs[27:30]
        body_quat = obs[30:34].astype(np.float32)
        body_grav = obs[34:37].astype(np.float32)
        obs[37:39] = target_cmd
        
        obs_n = obs_norm.normalize(obs[:30])
        joint_pos_n = obs_n[:12]
        joint_vel_n = obs_n[12:24]
        body_lin_vel_n = obs_n[24:27]
        body_ang_vel_n = obs_n[27:30]
        
        graph = builder.get_graph(joint_pos_n, joint_vel_n, body_quat, body_grav, body_lin_vel_n, body_ang_vel_n, target_cmd)
        batch = Batch.from_data_list([graph]).to(device)
        
        with torch.no_grad():
            joint_h = agent._joint_embeddings(agent._encode(batch)[0], batch)
            mean_action = agent.actor_head(joint_h).view(1, agent.num_joints)
        
        action_np = mean_action.squeeze(0).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
        env.command = target_cmd
        
        if step % 20 == 0:
            print(f"Step {step}: vx={info.get('forward_vel', body_lin_vel[0]):.3f}")

test_walk()

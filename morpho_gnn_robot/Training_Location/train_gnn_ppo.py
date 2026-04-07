"""
train_gnn_ppo.py

CleanRL-style PPO adapted for GNN graph observations.
Runs on Kaggle (T4/P100 GPU, headless, PyBullet DIRECT mode).

Key changes from CleanRL ppo_continuous_action.py:
  1. obs is a list of PyG Data objects, not a flat tensor
  2. Minibatch creation uses Batch.from_data_list()
  3. Graph is built from obs[:24] (joint pos/vel) at each rollout step

Usage:
  python train_gnn_ppo.py
  python train_gnn_ppo.py --seed 1 --total-timesteps 1000000

On Kaggle: set WANDB_API_KEY in Kaggle secrets and --track 1.
"""

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch

from urdf_to_graph      import URDFGraphBuilder
from robot_env_bullet   import RobotEnvBullet
from gnn_actor_critic   import GNNActorCritic


# -----------------------------------------------------------------------
# Running normalizer (Welford online algorithm)
# -----------------------------------------------------------------------
class RunningNorm:
    """
    Normalizes inputs to zero mean, unit variance using a running estimate.
    Used for both observations and rewards.

    WHY this matters: joint_pos in [-1.2, 1.2] and joint_vel in [-20, 20]
    are on different scales. Without normalization the GNN receives poorly
    scaled inputs and the critic cannot learn returns in the -5000 range.
    """
    def __init__(self, shape, clip: float = 10.0):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4
        self.clip  = clip

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0) if x.ndim > 1 else x
        batch_var  = np.var(x,  axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_n    = x.shape[0] if x.ndim > 1 else 1
        delta      = batch_mean - self.mean
        tot        = self.count + batch_n
        self.mean += delta * batch_n / tot
        self.var   = (self.var * self.count + batch_var * batch_n +
                      delta**2 * self.count * batch_n / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8),
                       -self.clip, self.clip).astype(np.float32)


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
@dataclass
class Config:
    # Environment
    urdf_path: str = (
        "/mnt/newvolume/Programming/Python/Deep_Learning/"
        "Relational_Bias_for_Morphological_Generalization/"
        "morpho_gnn_robot/morpho_ros2_ws/src/morpho_robot/urdf/anymal.urdf"
    )
    max_episode_steps: int = 1000

    # Training
    seed:             int   = 0
    total_timesteps:  int   = 5_000_000
    actor_learning_rate:  float = 3e-4
    critic_learning_rate: float = 1e-3  # faster value fitting to prevent actor-starvation from weak advantages
    weight_decay:     float = 1e-5    # tiny regularization to stabilize value fitting
    num_steps:        int   = 2048    # shorter rollouts to reduce Python graph-object overhead per update
    num_envs:         int   = 8       # Kaggle T4 default; use 12 if CPU headroom allows
    num_minibatches:  int   = 2       # larger minibatches reduce critic gradient variance
    update_epochs:    int   = 5
    gamma:            float = 0.995
    gae_lambda:       float = 0.95
    clip_coef:        float = 0.2
    ent_coef:         float = 0.0    # disable entropy push while policy entropy is already rising
    vf_coef:          float = 1.0    # critic is the bottleneck; give value fitting equal weight to policy loss
    max_grad_norm:    float = 0.5
    clip_vloss:       bool  = False    # keep unclipped value loss so critic can make large corrections
    log_std_min:      float = -2.5   # hard lower bound for policy log std (prevents variance collapse)
    log_std_max:      float = 0.0    # hard upper bound for policy log std (prevents late entropy drift)
    log_std_target:   float = -1.2   # slightly lower exploration target to improve late-stage exploitation
    log_std_reg_coef: float = 1e-3   # acts only on log_std; does not touch shared GNN weights
    resume_path: str = None

    # GNN
    hidden_dim:  int = 64

    # Logging
    track:       bool = False    # set True on Kaggle with wandb key
    run_name:    str  = ""
    save_every:  int  = 50_000   # checkpoint every N timesteps
    checkpoint_dir: str = "./checkpoints"

    @property
    def minibatch_size(self):
        return (self.num_steps * self.num_envs) // self.num_minibatches

    @property
    def batch_size(self):
        return self.num_steps * self.num_envs


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser()
    for f_name, f_val in cfg.__dict__.items():
        if f_name.startswith("_"):
            continue
        t = type(f_val) if f_val is not None else str
        if isinstance(f_val, bool):
            parser.add_argument(f"--{f_name.replace('_','-')}", type=int, default=int(f_val))
        else:
            parser.add_argument(f"--{f_name.replace('_','-')}", type=t, default=f_val)
    args = parser.parse_args()
    for f_name in cfg.__dict__:
        if f_name.startswith("_"):
            continue
        v = getattr(args, f_name.replace("-", "_"), None)
        if v is not None:
            if isinstance(getattr(cfg, f_name), bool):
                setattr(cfg, f_name, bool(v))
            else:
                setattr(cfg, f_name, v)
    if not cfg.run_name:
        cfg.run_name = f"gnn_ppo_seed{cfg.seed}_{int(time.time())}"
    return cfg


# -----------------------------------------------------------------------
# Rollout buffer (graphs stored as list, everything else as tensors)
# -----------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, action_dim: int, device: torch.device):
        self.num_steps  = num_steps
        self.num_envs   = num_envs
        self.action_dim = action_dim
        self.device     = device
        self.reset()

    def reset(self):
        self.graphs:   List[Data] = []
        self.actions   = torch.zeros(self.num_steps, self.num_envs, self.action_dim, device=self.device)
        self.logprobs  = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.rewards   = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.dones     = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        self.values    = torch.zeros(self.num_steps, self.num_envs, device=self.device)

    def store(self, step: int, env_idx: int, graph, action, logprob, reward, done, value):
        self.graphs.append(graph)
        self.actions[step, env_idx]  = action
        self.logprobs[step, env_idx] = logprob
        self.rewards[step, env_idx]  = reward
        self.dones[step, env_idx]    = done
        self.values[step, env_idx]   = value

    def compute_advantages(
        self, next_values: torch.Tensor, next_dones: torch.Tensor,
        gamma: float, gae_lambda: float
    ):
        """GAE advantage estimation."""
        advantages = torch.zeros(self.num_steps, self.num_envs, device=self.device)
        last_gae   = torch.zeros(self.num_envs, device=self.device)
        next_vals  = next_values.view(self.num_envs).to(self.device)
        next_dones = next_dones.view(self.num_envs).to(self.device)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                non_terminal = 1.0 - next_dones
                nv           = next_vals
            else:
                non_terminal = 1.0 - self.dones[t + 1, :]
                nv           = self.values[t + 1, :]

            delta     = self.rewards[t, :] + gamma * nv * non_terminal - self.values[t, :]
            last_gae  = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t, :] = last_gae

        returns = advantages + self.values
        return advantages, returns


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------
def train(cfg: Config):
    # ---- seeding ----
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- wandb ----
    if cfg.track:
        import wandb
        wandb.init(project="morpho_gnn_robot", name=cfg.run_name, config=cfg.__dict__)

    # ---- env + builder ----
    envs    = [
        RobotEnvBullet(cfg.urdf_path, max_episode_steps=cfg.max_episode_steps)
        for _ in range(cfg.num_envs)
    ]
    builder = URDFGraphBuilder(cfg.urdf_path, add_body_node=True)

    assert envs[0].joint_names == builder.joint_names, (
        "Joint ordering mismatch between env and builder. This will corrupt your training."
    )

    # Normalize only joint states (dims 0-30: pos, vel, lin_vel, ang_vel).
    # Orientation dims 30-37 (quaternion + gravity vector) are NOT normalized.
    # Quaternions are unit vectors on SO(3) -- applying (q-mean)/std destroys
    # the geometric meaning and gives the GNN garbage instead of orientation.
    # Gravity vector is already in [-1,1] by construction.
    # Joint states need normalization: pos in [-1.2,1.2], vel in [-20,20].
    OBS_NORM_DIM = 30   # normalize only first 30 dims
    obs_norm = RunningNorm(shape=(OBS_NORM_DIM,), clip=10.0)

    # ---- agent ----
    agent = GNNActorCritic(
        node_dim   = builder.node_dim,
        edge_dim   = builder.edge_dim,
        hidden_dim = cfg.hidden_dim,
        num_joints = builder.action_dim,
        log_std_min= cfg.log_std_min,
        log_std_max= cfg.log_std_max,
    ).to(device)

    # GNN encoder params (type_proj + GAT layers) update at critic speed
    # for value fitting to see consistent feature updates
    gnn_params = (
        list(agent.type_proj.parameters()) +
        list(agent.conv1.parameters()) +
        list(agent.norm1.parameters()) +
        list(agent.conv2.parameters()) +
        list(agent.norm2.parameters())
    )
    
    # Actor head (policy) + log_std variance parameter
    actor_params = list(agent.actor_head.parameters()) + [agent.log_std]
    
    # Critic head (value function) -- NO weight decay to allow unrestricted convergence
    critic_params = list(agent.critic_head.parameters())
    
    gnn_ids = {id(p) for p in gnn_params}
    actor_ids = {id(p) for p in actor_params}
    critic_ids = {id(p) for p in critic_params}
    shared_params = [
        p for p in agent.parameters()
        if id(p) not in gnn_ids and id(p) not in actor_ids and id(p) not in critic_ids
    ]

    optimizer = optim.Adam(
        [
            {"params": gnn_params, "lr": cfg.critic_learning_rate, "weight_decay": cfg.weight_decay},
            {"params": actor_params, "lr": cfg.actor_learning_rate, "weight_decay": cfg.weight_decay},
            {"params": critic_params, "lr": cfg.critic_learning_rate, "weight_decay": 0.0},
            {"params": shared_params, "lr": cfg.actor_learning_rate, "weight_decay": cfg.weight_decay},
        ],
        eps=1e-5,
    )
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    start_global_step = 0
    resumed_episode_rewards: List[float] = []

    if cfg.resume_path is not None:
        print(f"\nLoading checkpoint: {cfg.resume_path}")
        checkpoint = torch.load(cfg.resume_path, map_location=device, weights_only=False)

        # Resume across architecture changes by loading only matching keys.
        ckpt_agent = checkpoint.get("agent", {})
        model_state = agent.state_dict()
        filtered_state = {}
        skipped_missing = []
        skipped_shape = []

        for key, value in ckpt_agent.items():
            if key not in model_state:
                skipped_missing.append(key)
                continue
            if model_state[key].shape != value.shape:
                skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            filtered_state[key] = value

        load_result = agent.load_state_dict(filtered_state, strict=False)

        print(
            f"Loaded {len(filtered_state)}/{len(model_state)} model tensors "
            f"from checkpoint."
        )
        if skipped_missing:
            print(f"  Skipped {len(skipped_missing)} unknown checkpoint keys.")
        if skipped_shape:
            print(f"  Skipped {len(skipped_shape)} shape-mismatched keys:")
            for key, ckpt_shape, model_shape in skipped_shape[:10]:
                print(f"    - {key}: ckpt={ckpt_shape}, model={model_shape}")
            if len(skipped_shape) > 10:
                print(f"    ... and {len(skipped_shape) - 10} more")
        if load_result.missing_keys:
            print(f"  Model missing {len(load_result.missing_keys)} keys after load (expected for new layers).")

        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as exc:
                print(f"  Warning: could not load optimizer state ({exc}). Continuing with fresh optimizer state.")

        start_global_step = checkpoint.get("global_step", 0)
        resumed_episode_rewards = checkpoint.get("episode_rewards", [])
        if "obs_norm_mean" in checkpoint:
            obs_norm.mean  = checkpoint["obs_norm_mean"]
            obs_norm.var   = checkpoint["obs_norm_var"]
            obs_norm.count = checkpoint["obs_norm_count"]
        print(f"Resumed from step {start_global_step}")

    print(f"\nAgent parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print(
        f"Rollout steps/env: {cfg.num_steps} | "
        f"num_envs: {cfg.num_envs} | "
        f"batch size: {cfg.batch_size} | "
        f"minibatch size: {cfg.minibatch_size}"
    )
    print(f"Total updates: {cfg.total_timesteps // cfg.batch_size}\n")

    # ---- rollout buffer ----
    buffer = RolloutBuffer(cfg.num_steps, cfg.num_envs, builder.action_dim, device)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # ---- initial reset ----
    obs_list = []
    for i, env in enumerate(envs):
        obs, _ = env.reset(seed=cfg.seed + i)
        obs_list.append(obs)
    obs_array = np.stack(obs_list, axis=0)

    global_step      = start_global_step
    update           = 0
    episode_rewards: List[float] = list(resumed_episode_rewards)
    episode_lengths: List[int]   = []
    ep_rewards_live  = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_lengths_live  = np.zeros(cfg.num_envs, dtype=np.int32)

    start_time = time.time()

    # ================================================================
    # Main training loop
    # ================================================================
    target_timesteps = cfg.total_timesteps

    def _step_env(pair):
        env_i, act_i = pair
        return env_i.step(act_i)

    with ThreadPoolExecutor(max_workers=cfg.num_envs) as step_pool:
        while global_step < target_timesteps:
            update_start_time = time.time()

            # ---- learning rate annealing ----
            frac = 1.0 - (global_step - start_global_step) / (cfg.total_timesteps - start_global_step)
            frac = max(frac, 0.0)
            for base_lr, pg in zip(base_lrs, optimizer.param_groups):
                pg["lr"] = frac * base_lr

            # --------------------------------------------------------
            # Rollout collection
            # --------------------------------------------------------
            buffer.reset()
            last_step_dones = torch.zeros(cfg.num_envs, dtype=torch.float32, device=device)

            for step in range(cfg.num_steps):
                global_step += cfg.num_envs
                ep_lengths_live += 1

                # Normalize only joint states (dims 0-30); orientation dims are raw
                obs_norm.update(obs_array[:, :OBS_NORM_DIM])
                obs_n = obs_norm.normalize(obs_array[:, :OBS_NORM_DIM])

                graphs = []
                for env_idx in range(cfg.num_envs):
                    joint_pos = obs_n[env_idx, :12]
                    joint_vel = obs_n[env_idx, 12:24]
                    body_quat = obs_array[env_idx, 30:34].astype(np.float32)   # unit quaternion -- do not normalize
                    body_grav = obs_array[env_idx, 34:37].astype(np.float32)   # unit vector -- do not normalize
                    graphs.append(builder.get_graph(joint_pos, joint_vel, body_quat, body_grav))

                graph_batch = Batch.from_data_list(graphs).to(device)
                with torch.no_grad():
                    actions, logprobs, _, values = agent.get_action_and_value(graph_batch)

                actions_np = actions.cpu().numpy()
                step_results = list(
                    step_pool.map(_step_env, [(envs[i], actions_np[i]) for i in range(cfg.num_envs)])
                )

                for env_idx, (next_obs, reward, terminated, truncated, info) in enumerate(step_results):
                    done = terminated or truncated
                    last_step_dones[env_idx] = float(done)
                    ep_rewards_live[env_idx] += reward

                    buffer.store(
                        step,
                        env_idx,
                        graphs[env_idx],
                        actions[env_idx].cpu(),
                        logprobs[env_idx].cpu(),
                        float(reward),
                        float(done),
                        values[env_idx].cpu(),
                    )

                    if done:
                        episode_rewards.append(float(ep_rewards_live[env_idx]))
                        episode_lengths.append(int(ep_lengths_live[env_idx]))
                        ep_rewards_live[env_idx] = 0.0
                        ep_lengths_live[env_idx] = 0
                        next_obs, _ = envs[env_idx].reset()

                    obs_array[env_idx] = next_obs

            # next value for GAE bootstrap
            with torch.no_grad():
                obs_norm.update(obs_array[:, :OBS_NORM_DIM])
                obs_n = obs_norm.normalize(obs_array[:, :OBS_NORM_DIM])
                next_graphs = []
                for env_idx in range(cfg.num_envs):
                    joint_pos = obs_n[env_idx, :12]
                    joint_vel = obs_n[env_idx, 12:24]
                    body_quat = obs_array[env_idx, 30:34].astype(np.float32)
                    body_grav = obs_array[env_idx, 34:37].astype(np.float32)
                    next_graphs.append(builder.get_graph(joint_pos, joint_vel, body_quat, body_grav))
                next_batch = Batch.from_data_list(next_graphs).to(device)
                next_values = agent.get_value(next_batch).view(-1)

            b_next_dones = last_step_dones

            advantages, returns = buffer.compute_advantages(
                next_values, b_next_dones, cfg.gamma, cfg.gae_lambda
            )
            b_advantages = advantages.reshape(-1)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            b_returns    = returns.reshape(-1)
            b_actions    = buffer.actions.reshape(-1, builder.action_dim)
            b_logprobs   = buffer.logprobs.reshape(-1)
            b_values     = buffer.values.reshape(-1)

            # --------------------------------------------------------
            # PPO update
            # --------------------------------------------------------
            update += 1
            indices = np.arange(cfg.batch_size)

            pg_losses, vf_losses, ent_losses = [], [], []
            clip_fracs = []
            approx_kls = []

            for epoch in range(cfg.update_epochs):
                np.random.shuffle(indices)

                for start in range(0, cfg.batch_size, cfg.minibatch_size):
                    mb_idx = indices[start : start + cfg.minibatch_size]

                    # batch graphs for this minibatch
                    mb_batch    = Batch.from_data_list([buffer.graphs[i] for i in mb_idx]).to(device)
                    mb_actions  = b_actions[mb_idx].to(device)
                    mb_adv      = b_advantages[mb_idx].to(device)
                    mb_returns  = b_returns[mb_idx].to(device)
                    mb_logprobs = b_logprobs[mb_idx].to(device)

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        mb_batch, mb_actions
                    )
                    newvalue = newvalue.view(-1)

                    logratio  = newlogprob - mb_logprobs
                    ratio     = logratio.exp()

                    with torch.no_grad():
                        approx_kls.append(((ratio - 1.0) - logratio).mean().item())

                    # clip fraction diagnostic
                    with torch.no_grad():
                        clip_fracs.append(
                            ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                        )

                    # policy loss
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                    # value loss
                    if cfg.clip_vloss:
                        mb_vals = b_values[mb_idx].to(device)
                        v_clipped = mb_vals + (newvalue - mb_vals).clamp(
                            -cfg.clip_coef, cfg.clip_coef
                        )
                        vf_loss = torch.max(
                            (newvalue   - mb_returns) ** 2,
                            (v_clipped  - mb_returns) ** 2,
                        ).mean() * 0.5
                    else:
                        vf_loss = ((newvalue - mb_returns) ** 2).mean() * 0.5

                    ent_loss = entropy.mean()
                    std_reg_loss = ((agent.log_std - cfg.log_std_target) ** 2).mean()
                    loss = (
                        pg_loss
                        - cfg.ent_coef * ent_loss
                        + cfg.vf_coef * vf_loss
                        + cfg.log_std_reg_coef * std_reg_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    pg_losses.append(pg_loss.item())
                    vf_losses.append(vf_loss.item())
                    ent_losses.append(ent_loss.item())

            # --------------------------------------------------------
            # Logging
            # --------------------------------------------------------
            sps = int(global_step / (time.time() - start_time))
            update_dt = max(time.time() - update_start_time, 1e-8)
            sps_window = int(cfg.batch_size / update_dt)

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

            if episode_rewards:
                mean_ep_rew = np.mean(episode_rewards[-20:])
                mean_ep_len = np.mean(episode_lengths[-20:])
            else:
                mean_ep_rew = 0.0
                mean_ep_len = 0.0

            print(
                f"step={global_step:>8d} | "
                f"ep_rew={mean_ep_rew:>8.2f} | "
                f"ep_len={mean_ep_len:>6.0f} | "
                f"pg={np.mean(pg_losses):>7.4f} | "
                f"vf={np.mean(vf_losses):>7.4f} | "
                f"ent={np.mean(ent_losses):>6.4f} | "
                f"log_std={agent.log_std.mean().item():>6.3f} | "
                f"kl={np.mean(approx_kls):.5f} | "
                f"ev={explained_var:>6.3f} | "
                f"clip={np.mean(clip_fracs):.3f} | "
                f"lr_g={optimizer.param_groups[0]['lr']:.2e} | "
                f"lr_a={optimizer.param_groups[1]['lr']:.2e} | "
                f"lr_c={optimizer.param_groups[2]['lr']:.2e} | "
                f"sps={sps} | "
                f"sps_u={sps_window}"
            )

            if cfg.track:
                import wandb
                wandb.log({
                    "charts/ep_reward_mean":   mean_ep_rew,
                    "charts/ep_length_mean":   mean_ep_len,
                    "losses/policy_loss":      np.mean(pg_losses),
                    "losses/value_loss":       np.mean(vf_losses),
                    "losses/entropy":          np.mean(ent_losses),
                    "losses/log_std_mean":     agent.log_std.mean().item(),
                    "losses/approx_kl":        np.mean(approx_kls),
                    "losses/explained_variance": explained_var,
                    "charts/clip_frac":        np.mean(clip_fracs),
                    "charts/sps":              sps,
                    "charts/sps_window":       sps_window,
                    "charts/learning_rate_gnn":    optimizer.param_groups[0]["lr"],
                    "charts/learning_rate_actor":  optimizer.param_groups[1]["lr"],
                    "charts/learning_rate_critic": optimizer.param_groups[2]["lr"],
                }, step=global_step)

            # checkpoint
            if global_step % cfg.save_every < cfg.num_steps * cfg.num_envs:
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"gnn_ppo_{global_step}.pt")
                torch.save({
                    "global_step":   global_step,
                    "agent":         agent.state_dict(),
                    "optimizer":     optimizer.state_dict(),
                    "episode_rewards": episode_rewards,
                    "obs_norm_mean": obs_norm.mean,
                    "obs_norm_var":  obs_norm.var,
                    "obs_norm_count": obs_norm.count,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

    # ---- final save ----
    final_path = os.path.join(cfg.checkpoint_dir, "gnn_ppo_final.pt")
    torch.save({"global_step": global_step, "agent": agent.state_dict()}, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    for env in envs:
        env.close()
    if cfg.track:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

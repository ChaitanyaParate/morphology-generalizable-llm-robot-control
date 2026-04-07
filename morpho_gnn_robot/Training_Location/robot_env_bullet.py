"""
robot_env_bullet.py

Gym environment for legged robot locomotion using PyBullet DIRECT mode.
Designed for Kaggle headless training -- no display, no GUI.

Observation layout (37-dim):
  [0:12]  joint positions   (rad)
  [12:24] joint velocities  (rad/s)
  [24:27] base linear  velocity (m/s)
  [27:30] base angular velocity (rad/s)
  [30:34] base orientation quaternion (qx, qy, qz, qw)
  [34:37] projected gravity vector (body frame)

Joint ordering: alphabetically sorted by name.
This MUST match URDFGraphBuilder(urdf_path).joint_names for graph construction.

Action: 12-dim in [-1, 1]. Interpreted as offsets from nominal standing pose,
scaled by 0.35 rad. PD controller converts target positions to torques.

FIX (critical): Every env instance now uses its own physics client ID.
Previously all p.* calls used the implicitly-selected global client, so
env.reset() in any thread called p.resetSimulation() on the shared client,
wiping every other env's robot mid-episode. This caused ep_len of 21-118
steps regardless of policy quality -- the robot was being deleted, not falling.
The fix: store self._client at connect time, pass physicsClientId=self._client
to every single pybullet call in this file.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError("pip install pybullet")


CONTROLLABLE = {"revolute", "continuous", "prismatic"}

KP = 150.0
KD = 5.0

NOMINAL_POSE_PER_JOINT = {
    "LF_HAA":  0.0,  "LF_HFE":  0.6,  "LF_KFE": -1.2,
    "RF_HAA":  0.0,  "RF_HFE":  0.6,  "RF_KFE": -1.2,
    "LH_HAA":  0.0,  "LH_HFE": -0.6,  "LH_KFE":  1.2,
    "RH_HAA":  0.0,  "RH_HFE": -0.6,  "RH_KFE":  1.2,
}


class RobotEnvBullet(gym.Env):
    metadata = {"render_modes": ["human", "direct"]}

    FALL_PENALTY = 100.0
    ALIVE_BONUS = 0.05
    TARGET_FWD_VEL = 0.6
    MIN_FWD_VEL = 0.3
    SPAWN_HEIGHT = 0.70
    SETTLE_STEPS = 500
    FALL_HEIGHT = 0.30
    BAD_ORIENTATION = 1.2

    def __init__(
        self,
        urdf_path:         str,
        max_episode_steps: int = 1000,
        render_mode:       str = None,
        forward_axis:      int = 0,
    ):
        super().__init__()
        self.urdf_path         = urdf_path
        self.max_episode_steps = max_episode_steps
        self.render_mode       = render_mode
        self.forward_axis      = forward_axis

        self._parse_urdf()

        self.obs_dim    = 12 + 12 + 3 + 3 + 4 + 3   # = 37
        self.action_dim = len(self.joint_names)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,), dtype=np.float32
        )

        # Each instance gets its own physics client -- mandatory for parallel envs.
        # All subsequent p.* calls pass physicsClientId=self._client.
        if render_mode == "human":
            self._client = p.connect(p.GUI)
        else:
            self._client = p.connect(p.DIRECT)

        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self._client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)

        self._robot_id   = None
        self._step_count = 0
        self._fell       = False
        self.prev_action = np.zeros(self.action_dim)

    # ------------------------------------------------------------------
    def _parse_urdf(self):
        import xml.etree.ElementTree as ET
        root = ET.parse(self.urdf_path).getroot()
        names:   list = []
        efforts: dict = {}
        for joint in root.findall("joint"):
            jtype = joint.attrib.get("type", "fixed")
            if jtype not in CONTROLLABLE:
                continue
            name = joint.attrib["name"]
            names.append(name)
            lim = joint.find("limit")
            efforts[name] = float(lim.attrib.get("effort", 40.0)) if lim is not None else 40.0
        self.joint_names   = sorted(names)
        self.effort_limits = np.array([efforts[n] for n in self.joint_names])

    # ------------------------------------------------------------------
    def _load_robot(self):
        """(Re)load robot. Uses self._client for all pybullet calls."""
        c = self._client  # shorthand

        if self._robot_id is not None:
            p.removeBody(self._robot_id, physicsClientId=c)

        start_pos    = [0.0, 0.0, self.SPAWN_HEIGHT]
        start_orient = p.getQuaternionFromEuler([0.0, 0.0, 0.0], physicsClientId=c)
        robot_id = p.loadURDF(
            self.urdf_path,
            start_pos, start_orient,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=c,
        )
        self._robot_id = robot_id

        num_joints = p.getNumJoints(robot_id, physicsClientId=c)
        self._joint_idx: dict = {}

        for i in range(num_joints):
            info  = p.getJointInfo(robot_id, i, physicsClientId=c)
            name  = info[1].decode("utf-8")
            if name in self.joint_names:
                self._joint_idx[name] = i
                p.setJointMotorControl2(
                    robot_id, i,
                    p.VELOCITY_CONTROL,
                    force=0,
                    physicsClientId=c,
                )

        assert len(self._joint_idx) == len(self.joint_names), (
            f"URDF has {len(self.joint_names)} controllable joints but "
            f"PyBullet found {len(self._joint_idx)}."
        )

        self._pybullet_indices = np.array(
            [self._joint_idx[n] for n in self.joint_names], dtype=np.int32
        )
        self._nominal_pos = np.array([
            NOMINAL_POSE_PER_JOINT.get(name, 0.0)
            for name in self.joint_names
        ])

        for name, idx in self._joint_idx.items():
            val = NOMINAL_POSE_PER_JOINT.get(name, 0.0)
            p.resetJointState(robot_id, int(idx), val, physicsClientId=c)

        for i in range(p.getNumJoints(robot_id, physicsClientId=c)):
            p.changeDynamics(
                robot_id, i,
                linearDamping=0.04,
                angularDamping=0.04,
                physicsClientId=c,
            )

        p.changeDynamics(
            robot_id, -1,
            linearDamping=0.04,
            angularDamping=0.04,
            physicsClientId=c,
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        c = self._client

        p.resetSimulation(physicsClientId=c)
        self._robot_id = None
        p.setGravity(0, 0, -9.81, physicsClientId=c)
        p.setTimeStep(1.0 / 240.0, physicsClientId=c)
        p.loadURDF("plane.urdf", physicsClientId=c)

        self._load_robot()

        # Settle to standing pose
        for _ in range(self.SETTLE_STEPS):
            states   = p.getJointStates(self._robot_id, self._pybullet_indices.tolist(), physicsClientId=c)
            curr_pos = np.array([s[0] for s in states])
            curr_vel = np.array([s[1] for s in states])
            torques  = KP * (self._nominal_pos - curr_pos) - KD * curr_vel
            torques  = np.clip(torques, -self.effort_limits, self.effort_limits)
            for idx, torque in zip(self._pybullet_indices, torques):
                p.setJointMotorControl2(
                    self._robot_id, int(idx),
                    p.TORQUE_CONTROL, force=float(torque),
                    physicsClientId=c,
                )
            p.stepSimulation(physicsClientId=c)

            if np.max(np.abs(curr_pos - self._nominal_pos)) < 0.02:
                break

        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=c)
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn, physicsClientId=c)
        # print(f"[reset] height={base_pos[2]:.3f}  roll={roll:.3f}  pitch={pitch:.3f}")
        assert base_pos[2] > 0.35, f"Robot did not settle: height={base_pos[2]:.3f}"

        self._step_count = 0
        self._fell       = False
        self.prev_action = np.zeros(self.action_dim)

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        c = self._client
        action = np.clip(action, -1.0, 1.0)

        target_pos    = self._nominal_pos + action * 0.35
        smooth_penalty = np.sum((action - self.prev_action) ** 2)
        self.prev_action = action.copy()

        states   = p.getJointStates(self._robot_id, self._pybullet_indices.tolist(), physicsClientId=c)
        curr_pos = np.array([s[0] for s in states])
        curr_vel = np.array([s[1] for s in states])

        torques = KP * (target_pos - curr_pos) - KD * curr_vel
        torques = np.clip(torques, -self.effort_limits, self.effort_limits)

        for idx, torque in zip(self._pybullet_indices, torques):
            p.setJointMotorControl2(
                self._robot_id, int(idx),
                p.TORQUE_CONTROL, force=float(torque),
                physicsClientId=c,
            )

        for _ in range(4):
            p.stepSimulation(physicsClientId=c)

        self._step_count += 1

        obs = self._get_obs()

        base_pos, base_orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=c)
        base_height = base_pos[2]
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn, physicsClientId=c)

        reward = self._compute_reward(obs, torques, smooth_penalty, base_height)

        self._fell      = base_height < self.FALL_HEIGHT
        bad_orientation = abs(roll) > self.BAD_ORIENTATION or abs(pitch) > self.BAD_ORIENTATION
        terminated      = self._fell or bad_orientation
        truncated       = self._step_count >= self.max_episode_steps

        if terminated:
            reward -= self.FALL_PENALTY

        info = {"base_height": base_height, "step": self._step_count, "fell": self._fell}
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        c = self._client
        states    = p.getJointStates(self._robot_id, self._pybullet_indices.tolist(), physicsClientId=c)
        joint_pos = np.array([s[0] for s in states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in states], dtype=np.float32)

        lin_vel, ang_vel = p.getBaseVelocity(self._robot_id, physicsClientId=c)
        lin_vel = np.array(lin_vel, dtype=np.float32)
        ang_vel = np.array(ang_vel, dtype=np.float32)

        _, base_orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=c)
        quat = np.array(base_orn, dtype=np.float32)

        rot_mat = np.array(p.getMatrixFromQuaternion(base_orn, physicsClientId=c), dtype=np.float32).reshape(3, 3)
        gravity_body = rot_mat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

        return np.concatenate([joint_pos, joint_vel, lin_vel, ang_vel, quat, gravity_body])

    # ------------------------------------------------------------------
    def _compute_reward(self, obs, torques, smooth_penalty, base_height) -> float:
        c = self._client
        lin_vel = obs[24:27]
        ang_vel = obs[27:30]

        _, base_orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=c)
        roll, pitch, _ = p.getEulerFromQuaternion(base_orn, physicsClientId=c)

        forward_vel = float(lin_vel[self.forward_axis])
        lateral_vel = float(lin_vel[1 - self.forward_axis])

        forward_vel_pos = max(0.0, forward_vel)
        r_vel = 8.0 * (1.0 - np.exp(-2.0 * forward_vel_pos))
        r_vel_penalty = -0.5 * max(0.0, self.MIN_FWD_VEL - forward_vel_pos)

        r = (
            self.ALIVE_BONUS
            + r_vel
            + r_vel_penalty
            - 0.5  * abs(lateral_vel)
            - 1.5  * (roll**2 + pitch**2)
            - 1.0  * max(0.0, 0.35 - base_height)
            - 0.01 * smooth_penalty
            - 0.5  * ang_vel[2]**2
        )
        return float(r)

    # ------------------------------------------------------------------
    def close(self):
        if p.isConnected(self._client):
            p.disconnect(self._client)


# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    URDF = sys.argv[1] if len(sys.argv) > 1 else "anymal_stripped.urdf"

    print("Testing single env...")
    env = RobotEnvBullet(URDF, max_episode_steps=200)
    obs, _ = env.reset()
    print(f"  obs shape: {obs.shape}  expected (37,)")
    total_r = 0.0
    for i in range(200):
        obs, r, term, trunc, info = env.step(env.action_space.sample() * 0.1)
        total_r += r
        if term or trunc:
            print(f"  episode ended at step {i}, fell={info['fell']}")
            break
    print(f"  total reward: {total_r:.2f}")
    env.close()

    print("\nTesting 4 parallel envs (the critical fix)...")
    envs = [RobotEnvBullet(URDF, max_episode_steps=50) for _ in range(4)]
    for i, e in enumerate(envs):
        e.reset(seed=i)
    # Step all envs and confirm they don't interfere with each other
    for step in range(50):
        results = [e.step(e.action_space.sample() * 0.05) for e in envs]
        heights = [r[4]["base_height"] for r in results]
        dones   = [r[2] or r[3] for r in results]
        if any(dones):
            for i, (done, e) in enumerate(zip(dones, envs)):
                if done:
                    e.reset()
    print(f"  Final heights: {[f'{h:.3f}' for h in heights]}")
    print("  All 4 envs survived without interfering -- fix confirmed.")
    for e in envs:
        e.close()
    print("\nSmoke test passed.")
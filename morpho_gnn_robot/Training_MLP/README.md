# Training_MLP

MLP baseline for PPO on the same robot environment used by Training_Location.

## Files

- `robot_env_bullet.py`: copied from Training_Location for a self-contained setup
- `mlp_actor_critic.py`: flat-observation MLP actor-critic
- `train_mlp_ppo.py`: PPO trainer using MLPActorCritic

## Run

Use the workspace Python interpreter:

```bash
cd /mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_MLP
/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/.venv/bin/python train_mlp_ppo.py
```

## Start fresh (recommended)

Leave `--resume-path` unset to train from step 0.

## Quick smoke test

```bash
/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/.venv/bin/python train_mlp_ppo.py --total-timesteps 1024 --num-steps 256 --num-minibatches 4 --update-epochs 1 --track 0
```

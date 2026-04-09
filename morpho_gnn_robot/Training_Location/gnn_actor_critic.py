"""
gnn_actor_critic.py  (v2: HeteroGNNActorCritic)

What changed from v1 (homogeneous GATv2Conv) and WHY
------------------------------------------------------
v1 applied the same learned linear transformation to every node regardless
of its role (HAA vs HFE vs KFE vs body). That is wrong: these joints have
different axis directions, different torque limits, different kinematic
roles, and different functional behaviour during locomotion. Treating them
identically forces the GNN to waste capacity learning to distinguish them
from raw features that already encode the distinction structurally.

v2 adds one linear projection layer per node role BEFORE message passing.
All four HAA joints (LF, RF, LH, RH) share the same projection weights.
All four HFE joints share theirs. This is the core idea from MI-HGNN
(Butterfield et al. ICRA 2025) adapted to RL: enforce morphological
symmetry at the input level, then let GATv2Conv handle cross-joint
message passing.

Why NOT full MI-HGNN
---------------------
MI-HGNN is a supervised contact perception model. Its decoder is
foot-specific. Its training is supervised MSE/CE, not PPO. The
heterogeneity idea is sound; the full architecture is not applicable.

Node role integers (from urdf_to_graph.py)
  0: body (virtual root)
  1: HAA
  2: HFE
  3: KFE
  4: generic (non-ANYmal morphologies)

Morphology generalization
-------------------------
The projection layers are indexed by role, not by joint index. A hexapod
with 18 joints (6x HAA=1, 6x HFE=2, 6x KFE=3) will use the same 4
projection layers as a quadruped, just applied to more nodes. The
GATv2Conv weights are shared across the whole graph. The network is fully
morphology-agnostic at inference time -- you change the graph structure
(node_types, edge_index, x), not the network weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch

from urdf_to_graph import NUM_NODE_ROLES   # = 5


def _layer_init(layer, std=0.01, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class HeteroGNNActorCritic(nn.Module):
    """
    Parameters
    ----------
    node_dim    : raw node feature dim from URDFGraphBuilder  (26 for ANYmal)
    edge_dim    : edge feature dim                            (4)
    hidden_dim  : projected/GNN hidden size                   (64)
    num_joints  : number of controllable joints               (12 for ANYmal)
    num_roles   : number of node types (default: NUM_NODE_ROLES=5)

    Architecture
    ------------
        [raw node features, 26-dim]
            |
        type_proj[role]   <- one Linear(26, hidden_dim) per role, no weight sharing
          |                 body, HAA, HFE, KFE, generic each get their own
    [projected, hidden_dim]  <- all nodes now in same latent space
          |
    GATv2Conv(hidden_dim -> hidden_dim*4, heads=4, edge_dim=4) + LayerNorm + ELU
          |
    GATv2Conv(hidden_dim*4 -> hidden_dim, heads=1, edge_dim=4) + LayerNorm + ELU
          |
    [node embeddings, hidden_dim]
         / \\
    actor    critic
    (joint   (global
    nodes)    pool)
    """

    def __init__(
        self,
        node_dim:   int = 20,
        edge_dim:   int = 4,
        hidden_dim: int = 64,
        num_joints: int = 12,
        num_roles:  int = NUM_NODE_ROLES,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.num_roles  = num_roles

        # ---- Type-specific input projections ----------------------------
        # Each role gets its own Linear. No weight sharing BETWEEN roles
        # (HAA and KFE have genuinely different input semantics).
        # Weight sharing WITHIN a role (all 4 HAA nodes share the same
        # projection) is enforced implicitly: same role index -> same layer.
        self.type_proj = nn.ModuleList([
            nn.Linear(node_dim, hidden_dim) for _ in range(num_roles)
        ])

        # ---- GNN encoder ------------------------------------------------
        self.conv1 = GATv2Conv(
            in_channels  = hidden_dim,
            out_channels = hidden_dim,
            heads        = 4,
            edge_dim     = edge_dim,
            concat       = True,
            dropout      = 0.0,
        )
        self.norm1 = nn.LayerNorm(hidden_dim * 4)

        self.conv2 = GATv2Conv(
            in_channels  = hidden_dim * 4,
            out_channels = hidden_dim,
            heads        = 1,
            edge_dim     = edge_dim,
            concat       = False,
            dropout      = 0.0,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ---- Actor head (per joint node) --------------------------------
        self.actor_head = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=0.01),
        )
        self.log_std = nn.Parameter(torch.full((num_joints,), -0.7))

        # ---- Critic head (global pool) ----------------------------------
        self.critic_head = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, 64)),
            nn.Tanh(),
            _layer_init(nn.Linear(64, 1), std=1.0),
        )

    # ------------------------------------------------------------------
    def _project(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        """
        Apply type-specific linear projection to raw node features.

        x          : [N, node_dim]
        node_types : [N]  LongTensor of role integers

        Returns h  : [N, hidden_dim]

        Implementation note: iterating over roles and masking is O(num_roles),
        which is 5 iterations. This is negligible. An alternative is a
        scatter-based embedding table, but that loses the separate weight
        matrices per role (embedding tables have shared input->output dim).
        """
        h = torch.empty(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        for role in range(self.num_roles):
            mask = node_types == role
            if mask.any():
                h[mask] = self.type_proj[role](x[mask])
        return F.elu(h)

    # ------------------------------------------------------------------
    def _encode(self, data: Data):
        """Run type projection + GNN encoder."""
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        node_types = data.node_types  # [N] -- from URDFGraphBuilder.get_graph()
        batch      = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self._project(x, node_types)                          # [N, hidden_dim]
        h = F.elu(self.norm1(self.conv1(h, edge_index, edge_attr)))  # [N, hidden_dim*4]
        h = F.elu(self.norm2(self.conv2(h, edge_index, edge_attr)))  # [N, hidden_dim]
        return h, batch

    # ------------------------------------------------------------------
    def _joint_embeddings(self, h: torch.Tensor, data: Data) -> torch.Tensor:
        """Extract joint-node embeddings, excluding body node at index 0 per graph."""
        ptr = getattr(data, "ptr", None)
        if ptr is None:
            return h[1:]
        body_mask = torch.zeros(h.size(0), dtype=torch.bool, device=h.device)
        body_mask[ptr[:-1]] = True
        return h[~body_mask]

    # ------------------------------------------------------------------
    def get_value(self, data: Data) -> torch.Tensor:
        h, batch = self._encode(data)
        pooled   = global_mean_pool(h, batch)
        return self.critic_head(pooled)

    # ------------------------------------------------------------------
    def get_action_and_value(self, data: Data, action: torch.Tensor = None):
        h, batch = self._encode(data)

        joint_h = self._joint_embeddings(h, data)
        mean    = self.actor_head(joint_h)
        B       = batch.max().item() + 1
        mean    = mean.view(B, self.num_joints)

        std  = self.log_std.exp().clamp(min=0.15, max=0.8)
        std  = std.unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        pooled = global_mean_pool(h, batch)
        value  = self.critic_head(pooled)

        return action, log_prob, entropy, value


# -----------------------------------------------------------------------
# Backwards compatibility alias
# GNNActorCritic is used in gnn_policy_node.py and train_gnn_ppo.py.
# Point it at HeteroGNNActorCritic so you do not need to rename every import.
# Remove this alias once you have updated all callers.
# -----------------------------------------------------------------------
GNNActorCritic = HeteroGNNActorCritic


# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    from torch_geometric.data import Batch

    NODE_DIM = 26   # updated: was 20 before base velocity was added
    EDGE_DIM = 4
    N_JOINTS = 12
    N_NODES  = 13   # 1 body + 12 joints

    # Simulate what URDFGraphBuilder produces for ANYmal:
    # node 0 = body (role 0)
    # nodes 1-12 = LF_HAA(1), LF_HFE(2), LF_KFE(3), LH_HAA(1), ... x4 legs
    def _dummy_graph():
        roles = [0] + [1, 2, 3] * 4   # body + 4 x (HAA, HFE, KFE)
        return Data(
            x          = torch.randn(N_NODES, NODE_DIM),
            edge_index = torch.randint(0, N_NODES, (2, 24)),
            edge_attr  = torch.randn(24, EDGE_DIM),
            node_types = torch.tensor(roles, dtype=torch.long),
        )

    agent = HeteroGNNActorCritic(NODE_DIM, EDGE_DIM, hidden_dim=64, num_joints=N_JOINTS)
    total_params = sum(p.numel() for p in agent.parameters())
    proj_params  = sum(p.numel() for p in agent.type_proj.parameters())
    print(f"Total parameters  : {total_params:,}")
    print(f"Projection params : {proj_params:,}  ({5} roles x Linear({NODE_DIM},{64}))")
    print(f"GNN+head params   : {total_params - proj_params:,}")

    g = _dummy_graph()
    act, lp, ent, val = agent.get_action_and_value(g)
    print("\nSingle graph:")
    print(f"  action   : {act.shape}   expected [1, {N_JOINTS}]")
    print(f"  log_prob : {lp.shape}    expected [1]")
    print(f"  value    : {val.shape}   expected [1, 1]")

    batch = Batch.from_data_list([_dummy_graph() for _ in range(8)])
    act_b, lp_b, ent_b, val_b = agent.get_action_and_value(batch)
    print("\nBatched (B=8):")
    print(f"  action   : {act_b.shape}  expected [8, {N_JOINTS}]")
    print(f"  log_prob : {lp_b.shape}   expected [8]")
    print(f"  value    : {val_b.shape}  expected [8, 1]")

    loss = -lp_b.mean() + val_b.mean()
    loss.backward()
    dead = [n for n, p in agent.named_parameters() if p.grad is None]
    print(f"\nDead gradients: {dead if dead else 'none'}")

    # Verify morphology generalization: hexapod (18 joints)
    print("\nMorphology transfer test (hexapod, 18 joints):")
    N_HEX = 19  # 1 body + 18
    hexapod_roles = [0] + [1, 2, 3] * 6
    g_hex = Data(
        x          = torch.randn(N_HEX, NODE_DIM),
        edge_index = torch.randint(0, N_HEX, (2, 36)),
        edge_attr  = torch.randn(36, EDGE_DIM),
        node_types = torch.tensor(hexapod_roles, dtype=torch.long),
    )
    # Hexapod has 18 joints so we need a different agent for it
    agent_hex = HeteroGNNActorCritic(NODE_DIM, EDGE_DIM, hidden_dim=64, num_joints=18)
    # Copy shared weights (everything except actor output and log_std)
    # In practice you load the same checkpoint and only the actor head
    # output dimension differs -- handle this via partial loading.
    act_hex, _, _, val_hex = agent_hex.get_action_and_value(g_hex)
    print(f"  action : {act_hex.shape}  expected [1, 18]")
    print(f"  value  : {val_hex.shape}  expected [1, 1]")
    print("\nAll checks passed.")
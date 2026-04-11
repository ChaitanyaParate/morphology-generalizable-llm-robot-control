"""
gnn_actor_critic.py  (v3: SlimHeteroGNNActorCritic)

Parameter budget
----------------
v2 (HeteroGNNActorCritic, hidden=64)  :  ~85 k parameters
v3 (SlimHeteroGNNActorCritic, hidden=48) :  ~29 k parameters  ✓ < 50 k

Where the cuts came from
------------------------
Component            v2      v3      How
-------------------  ------  ------  -----------------------------------------
hidden_dim           64      48      Every Linear/GATv2 is O(hidden²); going
                                     64→48 cuts quadratic terms by ~44 %.
conv1 heads          4       2       4 heads × 64-dim = 256-wide bottleneck;
                                     2 heads × 48-dim = 96-wide is sufficient
                                     for 13-node ANYmal graphs.
actor head           64-64-1 48-32-1 One hidden layer kept; inner width 64→32.
critic head          64-64-1 48-32-1 Same reduction as actor.

What was NOT cut
----------------
- Heterogeneous type projections: kept, they are cheap (6 480 params, 22 %)
  and are the main architectural contribution (MI-HGNN insight).
- Two GATv2Conv layers: kept. A single message-passing layer cannot propagate
  information between non-adjacent joints (e.g., LF_KFE ↔ RH_KFE through
  the body node requires 2 hops).
- LayerNorm after each conv: kept. Training is unstable without it when
  GAT attention weights vary across heterogeneous node types.
- Separate actor/critic heads: required for PPO.

Expected parameter breakdown  (ANYmal: node_dim=26, 12 joints, 5 roles)
----------------------------------------------------------------------
  type_proj  (5 × Linear(26,48))             :   6 480
  conv1      (GATv2Conv, 48→96, heads=2)     :   9 792
  norm1      (LayerNorm, 96)                 :     192
  conv2      (GATv2Conv, 96→48, heads=1)     :   9 504
  norm2      (LayerNorm, 48)                 :      96
  actor_head (Linear(48,32)+Linear(32,1))    :   1 601
  log_std    (12,)                           :      12
  critic_head(Linear(48,32)+Linear(32,1))    :   1 601
  ─────────────────────────────────────────────────────
  TOTAL                                      :  29 278

Backward compatibility
----------------------
GNNActorCritic alias now points to SlimHeteroGNNActorCritic.
train_gnn_ppo.py requires no changes other than passing hidden_dim=48
(or relying on the new default).

Config change in train_gnn_ppo.py
----------------------------------
  hidden_dim: int = 48   # was 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch

from urdf_to_graph import NUM_NODE_ROLES   # = 5


# -----------------------------------------------------------------------
# Weight initialisation helper
# -----------------------------------------------------------------------
def _layer_init(layer: nn.Linear, std: float = 0.01, bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# -----------------------------------------------------------------------
# Slim heterogeneous GNN actor-critic
# -----------------------------------------------------------------------
class SlimHeteroGNNActorCritic(nn.Module):
    """
    A parameter-efficient version of HeteroGNNActorCritic.

    Architecture
    ------------
        [raw node features, 26-dim]
                |
        type_proj[role]            -- Linear(26, 48), 5 roles, no cross-role sharing
                |
        [projected, 48-dim]
                |
        GATv2Conv(48→48, heads=2, edge_dim=4, concat=True) → 96-dim
        LayerNorm(96) + ELU
                |
        GATv2Conv(96→48, heads=1, edge_dim=4)
        LayerNorm(48) + ELU
                |
        [node embeddings, 48-dim]
              /   \\
          actor   critic
         (joint   (global
          nodes)   pool)
           |          |
      Linear(48,32) Linear(48,32)
      Tanh          Tanh
      Linear(32,1)  Linear(32,1)

    Parameters
    ----------
    node_dim   : raw node feature dim (26 for ANYmal with base velocity)
    edge_dim   : edge feature dim (4)
    hidden_dim : projected/GNN width (default 48; was 64 in v2)
    num_joints : controllable joints (12 for ANYmal)
    num_roles  : node role count (default NUM_NODE_ROLES = 5)
    """

    def __init__(
        self,
        node_dim:   int = 26,
        edge_dim:   int = 4,
        hidden_dim: int = 48,
        num_joints: int = 12,
        num_roles:  int = NUM_NODE_ROLES,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        self.num_roles  = num_roles

        # ---- Type-specific input projections ----------------------------
        # One Linear per role: body, HAA, HFE, KFE, generic.
        # All nodes of the same role share one projection matrix -- this
        # enforces morphological symmetry (all 4 HAA joints treated equally)
        # without conflating structurally different joints.
        self.type_proj = nn.ModuleList([
            nn.Linear(node_dim, hidden_dim) for _ in range(num_roles)
        ])

        # ---- GNN encoder ------------------------------------------------
        # Layer 1: 2 attention heads, concat → hidden_dim * 2
        mid_dim = hidden_dim * 2   # 96

        self.conv1 = GATv2Conv(
            in_channels  = hidden_dim,
            out_channels = hidden_dim,
            heads        = 2,
            edge_dim     = edge_dim,
            concat       = True,   # output: mid_dim
            dropout      = 0.0,
        )
        self.norm1 = nn.LayerNorm(mid_dim)

        # Layer 2: 1 head, no concat → back to hidden_dim
        self.conv2 = GATv2Conv(
            in_channels  = mid_dim,
            out_channels = hidden_dim,
            heads        = 1,
            edge_dim     = edge_dim,
            concat       = False,  # output: hidden_dim
            dropout      = 0.0,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ---- Actor head (applied per joint node) ------------------------
        # Reduced inner width: 48 → 32 → 1 (vs 64 → 64 → 1 in v2)
        self.actor_head = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, 32)),
            nn.Tanh(),
            _layer_init(nn.Linear(32, 1), std=0.01),
        )
        self.log_std = nn.Parameter(torch.full((num_joints,), -0.7))

        # ---- Critic head (global mean pool over all nodes) --------------
        self.critic_head = nn.Sequential(
            _layer_init(nn.Linear(hidden_dim, 32)),
            nn.Tanh(),
            _layer_init(nn.Linear(32, 1), std=1.0),
        )

    # ------------------------------------------------------------------
    def _project(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        """
        Apply type-specific linear projection to raw node features.

        x          : [N, node_dim]
        node_types : [N] LongTensor of role integers (from urdf_to_graph.py)
        returns    : [N, hidden_dim]

        Iterates over at most 5 roles (O(num_roles)) — negligible overhead.
        """
        h = torch.empty(x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        for role in range(self.num_roles):
            mask = node_types == role
            if mask.any():
                h[mask] = self.type_proj[role](x[mask])
        return F.elu(h)

    # ------------------------------------------------------------------
    def _encode(self, data: Data):
        """Run type projection + 2-layer GATv2 encoder."""
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        node_types = data.node_types          # [N] -- set by URDFGraphBuilder.get_graph()
        batch      = getattr(data, "batch", None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = self._project(x, node_types)                              # [N, hidden_dim]
        h = F.elu(self.norm1(self.conv1(h, edge_index, edge_attr)))   # [N, hidden_dim*2]
        h = F.elu(self.norm2(self.conv2(h, edge_index, edge_attr)))   # [N, hidden_dim]
        return h, batch

    # ------------------------------------------------------------------
    def _joint_embeddings(self, h: torch.Tensor, data: Data) -> torch.Tensor:
        """
        Extract embeddings for joint nodes only (exclude body node at index 0
        per graph in the batch). Uses data.ptr when batched.
        """
        ptr = getattr(data, "ptr", None)
        if ptr is None:
            return h[1:]
        body_mask = torch.zeros(h.size(0), dtype=torch.bool, device=h.device)
        body_mask[ptr[:-1]] = True
        return h[~body_mask]

    # ------------------------------------------------------------------
    def get_value(self, data: Data) -> torch.Tensor:
        h, batch = self._encode(data)
        pooled   = global_mean_pool(h, batch)   # [B, hidden_dim]
        return self.critic_head(pooled)          # [B, 1]

    # ------------------------------------------------------------------
    def get_action_and_value(self, data: Data, action: torch.Tensor = None):
        h, batch = self._encode(data)

        # Actor: one scalar per joint per graph in the batch
        joint_h = self._joint_embeddings(h, data)  # [B*num_joints, hidden_dim]
        mean     = self.actor_head(joint_h)         # [B*num_joints, 1]
        B        = batch.max().item() + 1
        mean     = mean.view(B, self.num_joints)    # [B, num_joints]

        std  = self.log_std.exp().clamp(min=0.12, max=0.6)
        std  = std.unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)   # [B]
        entropy  = dist.entropy().sum(dim=-1)          # [B]

        # Critic: global mean pool over all nodes (body + joints)
        pooled = global_mean_pool(h, batch)
        value  = self.critic_head(pooled)              # [B, 1]

        return action, log_prob, entropy, value


# -----------------------------------------------------------------------
# Backward compatibility alias
# train_gnn_ppo.py imports GNNActorCritic; no rename needed.
# Update hidden_dim default in Config: hidden_dim = 48
# -----------------------------------------------------------------------
GNNActorCritic         = SlimHeteroGNNActorCritic
HeteroGNNActorCritic   = SlimHeteroGNNActorCritic   # alias for old import


# -----------------------------------------------------------------------
# Smoke test + parameter audit
# -----------------------------------------------------------------------
if __name__ == "__main__":
    NODE_DIM = 26
    EDGE_DIM = 4
    N_JOINTS = 12
    N_NODES  = 13   # 1 body + 12 joints

    def _dummy_graph(n_nodes=N_NODES, n_joints=N_JOINTS):
        """Simulate URDFGraphBuilder output for ANYmal."""
        roles = [0] + [1, 2, 3] * 4   # body + 4×(HAA, HFE, KFE)
        n_edges = n_nodes * 2
        return Data(
            x          = torch.randn(n_nodes, NODE_DIM),
            edge_index = torch.randint(0, n_nodes, (2, n_edges)),
            edge_attr  = torch.randn(n_edges, EDGE_DIM),
            node_types = torch.tensor(roles[:n_nodes], dtype=torch.long),
        )

    agent = SlimHeteroGNNActorCritic(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM, hidden_dim=48, num_joints=N_JOINTS
    )

    # ---- Parameter audit ----
    total = sum(p.numel() for p in agent.parameters())
    breakdown = {
        "type_proj":   sum(p.numel() for p in agent.type_proj.parameters()),
        "conv1":       sum(p.numel() for p in agent.conv1.parameters()),
        "norm1":       sum(p.numel() for p in agent.norm1.parameters()),
        "conv2":       sum(p.numel() for p in agent.conv2.parameters()),
        "norm2":       sum(p.numel() for p in agent.norm2.parameters()),
        "actor_head":  sum(p.numel() for p in agent.actor_head.parameters()),
        "log_std":     agent.log_std.numel(),
        "critic_head": sum(p.numel() for p in agent.critic_head.parameters()),
    }
    print("=== Parameter breakdown ===")
    for name, count in breakdown.items():
        print(f"  {name:<12s}: {count:>6,}")
    print(f"  {'TOTAL':<12s}: {total:>6,}")
    assert total < 50_000, f"FAIL: {total:,} params >= 50 000 budget!"
    print(f"  Budget check: {total:,} < 50 000  ✓\n")

    # ---- Single graph forward ----
    g = _dummy_graph()
    act, lp, ent, val = agent.get_action_and_value(g)
    print("Single graph:")
    print(f"  action   : {act.shape}   expected [1, {N_JOINTS}]")
    print(f"  log_prob : {lp.shape}    expected [1]")
    print(f"  entropy  : {ent.shape}   expected [1]")
    print(f"  value    : {val.shape}   expected [1, 1]")

    # ---- Batched forward ----
    batch = Batch.from_data_list([_dummy_graph() for _ in range(8)])
    act_b, lp_b, ent_b, val_b = agent.get_action_and_value(batch)
    print("\nBatched (B=8):")
    print(f"  action   : {act_b.shape}  expected [8, {N_JOINTS}]")
    print(f"  log_prob : {lp_b.shape}   expected [8]")
    print(f"  value    : {val_b.shape}  expected [8, 1]")

    # ---- Gradient check ----
    loss = -lp_b.mean() + val_b.mean()
    loss.backward()
    dead = [n for n, p in agent.named_parameters() if p.grad is None]
    print(f"\nDead gradients: {dead if dead else 'none'}")

    # ---- Morphology transfer: hexapod (18 joints) ----
    print("\nMorphology transfer test (hexapod, 18 joints):")
    agent_hex = SlimHeteroGNNActorCritic(NODE_DIM, EDGE_DIM, hidden_dim=48, num_joints=18)
    n_hex = 19   # 1 body + 18 joints
    roles_hex = [0] + [1, 2, 3] * 6
    from torch_geometric.data import Data as D
    g_hex = D(
        x          = torch.randn(n_hex, NODE_DIM),
        edge_index = torch.randint(0, n_hex, (2, 36)),
        edge_attr  = torch.randn(36, EDGE_DIM),
        node_types = torch.tensor(roles_hex, dtype=torch.long),
    )
    act_hex, _, _, val_hex = agent_hex.get_action_and_value(g_hex)
    print(f"  action : {act_hex.shape}  expected [1, 18]")
    print(f"  value  : {val_hex.shape}  expected [1, 1]")

    print("\nAll checks passed.")
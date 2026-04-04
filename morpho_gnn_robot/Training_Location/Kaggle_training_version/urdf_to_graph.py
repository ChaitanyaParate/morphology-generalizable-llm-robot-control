"""
urdf_to_graph.py  --  Joint-node graph builder for GNN RL training.

Design
------
Nodes  : controllable joints (revolute / continuous / prismatic)
         + 1 virtual body node at index 0 (enables cross-limb message passing)
Edges  : joint A -> joint B  iff  A.child_link == B.parent_link
         body node <-> every joint whose parent link is the URDF root link
         All edges are bidirectional (forward + reverse stored separately)

Node features
  Static  (URDF, computed once) : [type_onehot(4), axis(3), lower, upper, effort, velocity]  ->  11
  Runtime (injected each step)  : [joint_pos, joint_vel]                                      ->   2
  Total                         :                                                                  20

  NOTE: NODE_DIM is now 20 (was 13 before body orientation was added).
  gnn_actor_critic.py must be constructed with node_dim=20.

Edge features : [origin(3), direction(1)]  ->  4
  direction = +1.0 for parent->child, -1.0 for child->parent

Node roles (used by HeteroGNNActorCritic for type-specific projections)
  0 : body (virtual root node)
  1 : HAA  (hip abduction/adduction) -- rolls leg sideways
  2 : HFE  (hip flexion/extension)   -- swings leg fwd/back
  3 : KFE  (knee flexion/extension)  -- knee joint
  4 : generic joint (unknown suffix)

  Role is inferred from joint name suffix (last '_'-delimited token).
  For ANYmal: LF_HAA -> role 1, LH_HFE -> role 2, RF_KFE -> role 3.
  For a hexapod with different naming, unknown suffixes fall back to role 4.
  Add entries to JOINT_ROLE_MAP to cover your morphology.

Usage in RL training loop
-------------------------
    builder = URDFGraphBuilder("anymal.urdf")
    graph = builder.get_graph(pos, vel, body_quat, body_grav)
    # graph.node_types : LongTensor [num_nodes]  -- NEW field
    actions, value = policy(graph)
"""

import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
JOINT_TYPES    = ["revolute", "continuous", "prismatic", "fixed"]
JOINT_TYPE_MAP = {k: i for i, k in enumerate(JOINT_TYPES)}
CONTROLLABLE   = {"revolute", "continuous", "prismatic"}

STATIC_DIM      = 11   # type_onehot(4) + axis(3) + limits(4)
JOINT_RUNTIME   = 2    # joint_pos, joint_vel
BODY_EXTRA      = 7    # quat(4) + projected_gravity(3)
RUNTIME_DIM     = JOINT_RUNTIME + BODY_EXTRA   # 9
NODE_DIM        = STATIC_DIM + RUNTIME_DIM     # 20

EDGE_DIM        = 4

# -----------------------------------------------------------------------
# Node role mapping -- extend this for non-ANYmal morphologies
# -----------------------------------------------------------------------
NODE_ROLE_BODY    = 0
NODE_ROLE_HAA     = 1
NODE_ROLE_HFE     = 2
NODE_ROLE_KFE     = 3
NODE_ROLE_GENERIC = 4   # fallback for unknown suffix
NUM_NODE_ROLES    = 5

JOINT_ROLE_MAP: Dict[str, int] = {
    "HAA":    NODE_ROLE_HAA,
    "HFE":    NODE_ROLE_HFE,
    "KFE":    NODE_ROLE_KFE,
    # Add hexapod or other morphology joints here, e.g.:
    # "COXA":  NODE_ROLE_HAA,   # same role semantics for hip joint
    # "FEMUR": NODE_ROLE_HFE,
    # "TIBIA": NODE_ROLE_KFE,
}


def _joint_role(joint_name: str) -> int:
    """
    Infer node role from the joint name suffix.
    LF_HAA -> last token 'HAA' -> NODE_ROLE_HAA (1)
    Unknown suffix -> NODE_ROLE_GENERIC (4)
    """
    suffix = joint_name.split("_")[-1].upper()
    return JOINT_ROLE_MAP.get(suffix, NODE_ROLE_GENERIC)


# -----------------------------------------------------------------------
# URDF parsing helpers
# -----------------------------------------------------------------------
def _xyz(tag) -> List[float]:
    if tag is not None and "xyz" in tag.attrib:
        return [float(v) for v in tag.attrib["xyz"].split()]
    return [0.0, 0.0, 0.0]


def _limits(joint) -> List[float]:
    lim = joint.find("limit")
    if lim is None:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(lim.attrib.get("lower",    0.0)),
        float(lim.attrib.get("upper",    0.0)),
        float(lim.attrib.get("effort",   0.0)),
        float(lim.attrib.get("velocity", 0.0)),
    ]


# -----------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------
class URDFGraphBuilder:
    """
    Parse a URDF once; produce runtime-updated PyG graphs at every RL step.

    New in this version
    -------------------
    - get_graph() now includes data.node_types: LongTensor [num_nodes]
      containing the NODE_ROLE_* integer for each node.
      HeteroGNNActorCritic uses this to apply type-specific input projections.
    - node_roles property exposes the precomputed role list for inspection.
    """

    def __init__(self, urdf_path: str, add_body_node: bool = True):
        self.urdf_path     = urdf_path
        self.add_body_node = add_body_node
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self):
        root = ET.parse(self.urdf_path).getroot()

        all_joints: Dict[str, ET.Element] = {
            j.attrib["name"]: j for j in root.findall("joint")
        }

        child_links  = {j.find("child").attrib["link"]  for j in all_joints.values()}
        root_links   = {
            lnk.attrib["name"]
            for lnk in root.findall("link")
            if lnk.attrib["name"] not in child_links
        }

        parent_to_joints: Dict[str, List[str]] = {}
        for jname, j in all_joints.items():
            pl = j.find("parent").attrib["link"]
            parent_to_joints.setdefault(pl, []).append(jname)

        ctrl: List[str] = sorted([
            jname for jname, j in all_joints.items()
            if j.attrib.get("type", "fixed") in CONTROLLABLE
        ])

        self.joint_names  = ctrl
        self.num_joints   = len(ctrl)
        offset            = 1 if self.add_body_node else 0
        self.joint_to_idx = {name: i + offset for i, name in enumerate(ctrl)}

        # ---- node roles (NEW) --------------------------------------------
        joint_roles = [_joint_role(jname) for jname in ctrl]
        if self.add_body_node:
            all_roles = [NODE_ROLE_BODY] + joint_roles
        else:
            all_roles = joint_roles
        self._node_types = torch.tensor(all_roles, dtype=torch.long)

        # ---- static node features ----------------------------------------
        rows = []
        for jname in ctrl:
            j     = all_joints[jname]
            jt    = j.attrib.get("type", "fixed")
            jt_id = JOINT_TYPE_MAP.get(jt, 3)
            oh    = [0.0] * len(JOINT_TYPES)
            oh[jt_id] = 1.0
            rows.append(oh + _xyz(j.find("axis")) + _limits(j))

        joint_static = torch.tensor(rows, dtype=torch.float)

        if self.add_body_node:
            self._static_x = torch.cat(
                [torch.zeros(1, STATIC_DIM), joint_static], dim=0
            )
        else:
            self._static_x = joint_static

        cont_cols = slice(4, STATIC_DIM)
        col_mean  = self._static_x[:, cont_cols].mean(dim=0, keepdim=True)
        col_std   = self._static_x[:, cont_cols].std(dim=0, keepdim=True).clamp(min=1e-6)
        self._static_x[:, cont_cols] = (
            (self._static_x[:, cont_cols] - col_mean) / col_std
        )

        # ---- edges -------------------------------------------------------
        edges:  List[Tuple[int, int]] = []
        efeats: List[List[float]]     = []

        for jname in ctrl:
            j          = all_joints[jname]
            child_link = j.find("child").attrib["link"]
            i          = self.joint_to_idx[jname]

            for downstream in parent_to_joints.get(child_link, []):
                if downstream not in self.joint_to_idx:
                    continue
                k      = self.joint_to_idx[downstream]
                origin = _xyz(all_joints[downstream].find("origin"))
                edges.append((i, k));  efeats.append(origin + [1.0])
                edges.append((k, i));  efeats.append(origin + [-1.0])

        if self.add_body_node:
            body = 0
            for jname in ctrl:
                j  = all_joints[jname]
                pl = j.find("parent").attrib["link"]
                if pl in root_links:
                    k      = self.joint_to_idx[jname]
                    origin = _xyz(j.find("origin"))
                    edges.append((body, k));  efeats.append(origin + [1.0])
                    edges.append((k, body));  efeats.append(origin + [-1.0])

        if edges:
            self._edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            ea = torch.tensor(efeats, dtype=torch.float)
            ea_mean = ea[:, :3].mean(dim=0, keepdim=True)
            ea_std  = ea[:, :3].std(dim=0,  keepdim=True).clamp(min=1e-6)
            ea[:, :3] = (ea[:, :3] - ea_mean) / ea_std
            self._edge_attr = ea
        else:
            self._edge_index = torch.zeros((2, 0), dtype=torch.long)
            self._edge_attr  = torch.zeros((0, EDGE_DIM), dtype=torch.float)

        self.num_nodes = self.num_joints + offset
        self._print_summary()

    # ------------------------------------------------------------------
    def _print_summary(self):
        tag = f"body + {self.num_joints} joints" if self.add_body_node else f"{self.num_joints} joints"
        role_counts = {}
        for r in self._node_types.tolist():
            role_counts[r] = role_counts.get(r, 0) + 1
        role_names = {NODE_ROLE_BODY: "body", NODE_ROLE_HAA: "HAA",
                      NODE_ROLE_HFE: "HFE", NODE_ROLE_KFE: "KFE",
                      NODE_ROLE_GENERIC: "generic"}
        role_str = ", ".join(f"{role_names.get(k,'?')}x{v}" for k, v in sorted(role_counts.items()))
        print(f"\n[URDFGraphBuilder] {self.urdf_path.split('/')[-1]}")
        print(f"  Nodes            : {self.num_nodes}  ({tag})")
        print(f"  Node roles       : {role_str}")
        print(f"  Edges            : {self._edge_index.shape[1]}  (bidirectional)")
        print(f"  Node feature dim : {NODE_DIM}  (static {STATIC_DIM} + runtime {RUNTIME_DIM})")
        print(f"  Edge feature dim : {EDGE_DIM}")
        print(f"  Action dim       : {self.num_joints}  (one torque per controllable joint)")
        print(f"  Joint order      : {self.joint_names}")

    # ------------------------------------------------------------------
    def get_graph(
        self,
        joint_pos:   Optional[np.ndarray] = None,
        joint_vel:   Optional[np.ndarray] = None,
        body_quat:   Optional[np.ndarray] = None,
        body_grav:   Optional[np.ndarray] = None,
    ) -> Data:
        """
        Build a PyG Data for one RL timestep.
        NOW INCLUDES data.node_types for heterogeneous GNN.
        """
        pos  = joint_pos if joint_pos is not None else np.zeros(self.num_joints)
        vel  = joint_vel if joint_vel is not None else np.zeros(self.num_joints)
        quat = body_quat if body_quat is not None else np.array([0., 0., 0., 1.], dtype=np.float32)
        grav = body_grav if body_grav is not None else np.array([0., 0., -1.], dtype=np.float32)

        joint_runtime = np.zeros((self.num_joints, RUNTIME_DIM), dtype=np.float32)
        joint_runtime[:, 0] = pos
        joint_runtime[:, 1] = vel

        runtime_joints = torch.tensor(joint_runtime, dtype=torch.float)

        if self.add_body_node:
            body_runtime = np.zeros((1, RUNTIME_DIM), dtype=np.float32)
            body_runtime[0, 2:6] = quat
            body_runtime[0, 6:9] = grav
            runtime = torch.cat(
                [torch.tensor(body_runtime, dtype=torch.float), runtime_joints], dim=0
            )
        else:
            runtime = runtime_joints

        x = torch.cat([self._static_x, runtime], dim=1)  # [num_nodes, 20]

        return Data(
            x          = x,
            edge_index = self._edge_index.clone(),
            edge_attr  = self._edge_attr.clone(),
            node_types = self._node_types.clone(),   # NEW: [num_nodes] LongTensor
        )

    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return self.num_joints

    @property
    def node_dim(self) -> int:
        return NODE_DIM

    @property
    def edge_dim(self) -> int:
        return EDGE_DIM

    @property
    def node_roles(self) -> torch.Tensor:
        """Precomputed node role integers. Inspect to verify correctness."""
        return self._node_types

    def obs_to_arrays(
        self,
        pos_dict: Dict[str, float],
        vel_dict: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array([pos_dict.get(n, 0.0) for n in self.joint_names])
        vel = np.array([vel_dict.get(n, 0.0) for n in self.joint_names])
        return pos, vel


# -----------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    path = (
        sys.argv[1] if len(sys.argv) > 1 else
        "/kaggle/working/anymal_stripped.urdf"
    )

    rng = np.random.default_rng(42)
    b   = URDFGraphBuilder(path, add_body_node=True)

    print("\n  node_roles tensor:", b.node_roles.tolist())
    print("  expected: [0, 1,2,3, 1,2,3, 1,2,3, 1,2,3]  (body, then 4x HAA/HFE/KFE)")

    pos = rng.uniform(-0.5,  0.5, b.num_joints)
    vel = rng.uniform(-1.0,  1.0, b.num_joints)
    g   = b.get_graph(pos, vel)

    assert g.x.shape        == (b.num_nodes, NODE_DIM),  "node feature shape mismatch"
    assert g.edge_attr.shape[1] == EDGE_DIM,              "edge feature shape mismatch"
    assert g.node_types.shape   == (b.num_nodes,),        "node_types shape mismatch"
    assert g.node_types.dtype   == torch.long,            "node_types must be LongTensor"

    print("\n  graph.node_types:", g.node_types.tolist())
    print("\nAll tests passed.")

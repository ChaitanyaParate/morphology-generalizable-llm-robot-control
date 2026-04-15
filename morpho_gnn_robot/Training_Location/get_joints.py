import pybullet as p
p.connect(p.DIRECT)
r = p.loadURDF("/mnt/newvolume/Programming/Python/Deep_Learning/Relational_Bias_for_Morphological_Generalization/morpho_gnn_robot/Training_Location/anymal_stripped.urdf")
for i in range(p.getNumJoints(r)):
    info = p.getJointInfo(r, i)
    print(f"Index {i} - Link Name: {info[12].decode('utf-8')}")
p.disconnect()

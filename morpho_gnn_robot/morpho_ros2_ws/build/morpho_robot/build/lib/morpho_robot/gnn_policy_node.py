import rclpy
from rclpy.node import Node

class GNNPolicyNode(Node):
    def __init__(self):
        super().__init__('gnn_policy_node')
        self.get_logger().info('gnn_policy_node stub running')

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GNNPolicyNode())
    rclpy.shutdown()
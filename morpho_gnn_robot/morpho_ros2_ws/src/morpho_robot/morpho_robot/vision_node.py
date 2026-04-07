#!/usr/bin/env python3
# vision_node.py

from curses import window
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import json
import numpy as np
import argparse
#export QT_QPA_FONTDIR=/usr/share/fonts
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'
import cv2
from sensor_msgs.msg import CameraInfo
class VisionNode(Node):
    def __init__(self, args):
        super().__init__('vision_node')
        
        self.bridge = CvBridge()
        self.model = YOLO(args.yolo_model)  # downloads on first run (~6MB)
        
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.cam_info_callback,
            10
        )

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.depth_image = None
        self.pub = self.create_publisher(String, '/scene_graph', 10)
        
        # Approximate focal length for distance estimation (tune per camera)
        self.focal_length_px = 554.0  # for 640px wide, 60deg FOV
        
        self.get_logger().info('Vision node ready')
    
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def cam_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge failed: {e}')
            return

        results = self.model(cv_image, verbose=False)[0]
        cv2.imshow("raw_rgb", cv_image)

        annotated = results.plot()
        cv2.imshow("detections", annotated)
        cv2.waitKey(1)

        objects = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if confidence < 0.4:
                continue

            cx = (x1 + x2) / 2.0
            # Normalize x position: -1 (left) to +1 (right)

            if self.depth_image is not None and self.fx is not None:
                u = int(cx)
                v = int((y1 + y2) / 2.0)

                h, w = self.depth_image.shape

                if u < 0 or u >= w or v < 0 or v >= h:
                    continue

                window = self.depth_image[v-2:v+3, u-2:u+3]
                Z = np.nanmedian(window)

                if np.isnan(Z) or Z <= 0:
                    continue

                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
            else:
                continue

            if self.depth_image is not None:
                depth_vis = self.depth_image.copy()

                # Replace invalid values
                depth_vis[np.isnan(depth_vis)] = 0

                # Normalize to 0–255 for display
                depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)

                # Apply colormap (makes it readable)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                cv2.imshow("depth", depth_vis)
                
            cv2.circle(annotated, (u, v), 5, (0, 255, 0), -1)

            objects.append({
                'label': label,
                'confidence': round(confidence, 2),
                'position': {
                    'x': round(float(X), 3),
                    'y': round(float(Y), 3),
                    'z': round(float(Z), 3)
                },
                'bbox': [round(x1), round(y1), round(x2), round(y2)]
            })

        scene = {'objects': objects, 'timestamp': msg.header.stamp.sec}
        
        out = String()
        out.data = json.dumps(scene)
        self.pub.publish(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', default='yolov8n.pt')
    parser.add_argument('--conf', type=float, default=0.4)
    args = parser.parse_args()

    rclpy.init()
    node = VisionNode(args)
    rclpy.spin(node)

if __name__ == '__main__':
    main()
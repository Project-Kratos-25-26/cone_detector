#!/usr/bin/env python3
"""
ROS 2 Cone Detector Node
------------------------
ROS 2 node to test YOLO model on input from ZED Wrapper.
Publishes annotated images with detections for visualization in RViz2.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
from ultralytics import YOLO
import os

class ConeDetectorNode(Node):
    def __init__(self):
        super().__init__('cone_detector_node')

        # Parameters
        self.declare_parameter('model_path', '/home/neel/ros2_ws/src/cone_detector/cone_detector/models/best.pt')
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        self.declare_parameter('saturation_threshold', 0)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('cone_real_height', 0.3) # meters
        self.declare_parameter('focal_length', 700.0)   # pixels approximation

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.saturation_threshold = self.get_parameter('saturation_threshold').get_parameter_value().integer_value
        self.conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.cone_real_height = self.get_parameter('cone_real_height').get_parameter_value().double_value
        self.focal_length = self.get_parameter('focal_length').get_parameter_value().double_value

        # Resolve model path
        if not os.path.exists(self.model_path):
            self.get_logger().info(f"Model not found at {self.model_path}, checking alternatives...")
            script_dir = os.path.dirname(os.path.realpath(__file__))
            
            if os.path.isabs(self.model_path):
                candidate_rel = os.path.join(script_dir, 'models', os.path.basename(self.model_path))
                if os.path.exists(candidate_rel):
                    self.model_path = candidate_rel
                else:
                    candidate_flat = os.path.join(script_dir, os.path.basename(self.model_path))
                    if os.path.exists(candidate_flat):
                        self.model_path = candidate_flat
            else:
                candidate_path = os.path.join(script_dir, self.model_path)
                if os.path.exists(candidate_path):
                    self.model_path = candidate_path

        if not os.path.exists(self.model_path):
             self.get_logger().error(f"FATAL: Model not found at {self.model_path}. Current CWD: {os.getcwd()}")

        # Initialize YOLO
        self.get_logger().info(f"🚀 Loading YOLO model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            self.get_logger().error(f"❌ Error loading model: {e}")
            self.destroy_node()
            return

        # CV Bridge
        self.bridge = CvBridge()

        # Subscribers and Publishers
        self.get_logger().info(f"📷 Subscribing to {self.image_topic}...")
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10)
        
        # Publisher for annotated image
        self.result_pub = self.create_publisher(Image, 'cone_detector/annotated_image', 10)
        self.detection_pub = self.create_publisher(String, '/cone_detector/detections', 10)
        self.get_logger().info("📡 Publishing annotated images to 'cone_detector/annotated_image'")
        self.get_logger().info("📡 Publishing detections JSON to '/cone_detector/detections'")


    def get_color_label(self, hue):
        """
        Returns a string label based on Hue value (0-179).
        """
        # OpenCV Hue is 0-179
        if 0 <= hue < 10 or 170 <= hue <= 179:
            return "Red"
        elif 10 <= hue < 25:
            return "Orange"
        elif 25 <= hue < 35:
            return "Yellow"
        elif 35 <= hue < 85:
            return "Green"
        elif 85 <= hue < 130:
            return "Blue"
        elif 130 <= hue < 170:
            return "Purple"
        else:
            return "Unknown"

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # Inference
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
        
        annotated_frame = frame.copy()
        
        detections_list = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # Calculate centers & size for downstream consumption
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bbox_h = y2 - y1
                # Estimate depth: Z = (f * H_real) / H_px
                depth_m = (self.focal_length * self.cone_real_height) / bbox_h if bbox_h > 0 else 999.0

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Create Triangle Mask for color detection
                h, w = roi.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.array([[w // 2, 0], [0, h], [w, h]], np.int32)
                cv2.fillPoly(mask, [pts], 255)

                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                hue_values = hsv_roi[:, :, 0][mask > 0]
                sat_values = hsv_roi[:, :, 1][mask > 0]

                color_name = "Unknown"
                if len(hue_values) > 0:
                    vals, counts = np.unique(hue_values, return_counts=True)
                    mode_hue = vals[np.argmax(counts)]
                    avg_saturation = np.mean(sat_values)

                    # Color determination logic
                    if avg_saturation < self.saturation_threshold:
                        color = (128, 128, 128)
                        label = f"Dull (Sat:{int(avg_saturation)})"
                    else:
                        color_name = self.get_color_label(mode_hue)
                        label = f"{color_name} {conf:.2f} {depth_m:.1f}m"
                        
                        if color_name == "Yellow": color = (0, 255, 255)
                        elif color_name == "Blue": color = (255, 0, 0)
                        elif color_name == "Red": color = (0, 0, 255)
                        elif color_name == "Orange": color = (0, 165, 255)
                        elif color_name == "Green": color = (0, 255, 0)
                        elif color_name == "Purple": color = (128, 0, 128)
                        else: color = (255, 255, 255)

                    # Draw Visuals
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    tri_pts = pts + [x1, y1]
                    cv2.polylines(annotated_frame, [tri_pts], True, (255, 255, 255), 1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add to detections list for JSON publishing
                    detections_list.append({
                        "class": "cone",
                        "color": color_name.lower(),
                        "depth_m": depth_m,
                        "center": [cx, cy]
                    })

        # Publish detections JSON
        json_msg = {
            "detections": detections_list,
            "width": float(frame.shape[1])
        }
        self.detection_pub.publish(String(data=json.dumps(json_msg)))

        # Publish the annotated frame

        # Publish the annotated frame
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            out_msg.header = msg.header
            self.result_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

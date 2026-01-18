#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cone_msgs.msg import ConeDetection, ConeDetectionArray
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO
import numpy as np

class ConeDetectorNode(Node):
    def __init__(self):
        super().__init__('cone_detector_node')
        
        # --- Parameters ---
        self.declare_parameter('image_topic', '/zed/zed_node/left/image_rect_color')
        self.declare_parameter('model_path', '') 
        self.declare_parameter('conf_thresh', 0.4)
        self.declare_parameter('debug', True)

        self.image_topic = self.get_parameter('image_topic').value
        model_path_param = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('conf_thresh').value
        self.debug = self.get_parameter('debug').value

        # --- Model Resolution ---
        # USER REQUESTED HARDCODED PATH
        self.model_path = '/home/neel/ros2_ws/src/cone_detector/models/best.pt'

        self.get_logger().info(f"Loading Model: {self.model_path}")
        
        # Explicit file check to prevent silent failures
        if not os.path.exists(self.model_path):
             self.get_logger().error(f"❌ MODEL FILE NOT FOUND: {self.model_path}")
             # Check what IS there to help
             parent_dir = os.path.dirname(self.model_path)
             if os.path.exists(parent_dir):
                 files = os.listdir(parent_dir)
                 self.get_logger().error(f"   📂 Found in {parent_dir}: {files}")
             else:
                 self.get_logger().error(f"   📂 Directory does not exist: {parent_dir}")
             raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO: {e}")
            raise e

        # --- Setup ---
        self.bridge = CvBridge()
        
        # Publishers
        self.pub_annotated = self.create_publisher(Image, 'cone_detector/annotated_image', 10)
        self.pub_detections = self.create_publisher(ConeDetectionArray, '/cone_detector/detections', 10)
        
        # Subscribers
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        
        self.get_logger().info("Cone Detector (Simplified) Ready.")
        self.get_logger().info(f"Subscribed to: {self.image_topic}")

        # Metrics for debugging
        self.last_log_time = self.get_clock().now()
        self.frame_count = 0
        self.total_detections = 0

    def image_cb(self, msg):
        self.frame_count += 1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge conversion failed: {e}")
            return
        
        # DEBUG: Log raw image stats
        if self.frame_count % 10 == 0:
             self.get_logger().info(f"📷 RX IMAGE | {msg.width}x{msg.height} | Frame: {self.frame_count}")

        # Run Inference
        start_t = self.get_clock().now()
        results = self.model.predict(cv_image, conf=self.conf_thresh, verbose=False)[0]
        infer_dur = (self.get_clock().now() - start_t).nanoseconds / 1e6 # ms
        
        # Prepare Output Messages
        det_array = ConeDetectionArray()
        det_array.header = msg.header
        
        annotated_img = cv_image.copy()

        FOCAL_LENGTH = 700.0
        REAL_HEIGHT = 0.3

        detections_in_frame = 0
        
        if results.boxes:
            detections_in_frame = len(results.boxes)
            self.total_detections += detections_in_frame
            
            # DEBUG: Log EVERY detection frame heavily
            log_msg = f"🔍 DETECTED {detections_in_frame} CONES (Infer: {infer_dur:.1f}ms):"
            
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                
                if h <= 0: continue

                distance = (FOCAL_LENGTH * REAL_HEIGHT) / h
                
                # Append to log
                log_msg += f"\n    [{i}] Dist: {distance:5.2f}m | Conf: {conf:.2f} | Center: {cx:4.0f}x"

                # Populate Message
                det = ConeDetection()
                det.header = msg.header
                det.cx, det.cy = float(cx), float(cy)
                det.width, det.height = float(w), float(h)
                det.distance, det.confidence = float(distance), float(conf)
                det_array.detections.append(det)

                # Annotation
                label = f"{distance:.2f}m ({conf:.2f})"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.get_logger().info(log_msg)

        # Publish Messages
        self.pub_detections.publish(det_array)
        
        if infer_dur > 80.0:
             self.get_logger().warn(f"⚠️ SLOW FRAME: {infer_dur:.1f}ms")

        try:
            cv2.putText(annotated_img, f"Cones: {detections_in_frame} | Infer: {infer_dur:.0f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
            annotated_msg.header = msg.header
            self.pub_annotated.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

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

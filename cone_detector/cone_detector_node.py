#!/usr/bin/env python3
"""
ROS 2 Cone Detector Node
------------------------
Integrates YOLOv8 detection with robust post-processing logic (CLAHE, Aspect Ratio Filtering, Color Heuristics).

Author: [Your Name/Antigravity]
Date: 2025-01-13
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
import time

# ==============================================================================
# EASY CONFIGURATION SECTION
# Change these variables to tune the computer vision logic
# ==============================================================================
DEFAULT_SATURATION_THRESHOLD = 0      # Min saturation (0-255) to consider a cone "colored"
DEFAULT_BRIGHTNESS_THRESHOLD = 60     # Min brightness (0-255) to be considered valid
MIN_ASPECT_RATIO = 0.25               # Min width/height ratio (Cones are tall/thin)
MAX_ASPECT_RATIO = 0.9                # Max width/height ratio
USE_CLAHE_DEFAULT = True              # Enable Contrast Limited Adaptive Histogram Equalization by default
CONFIDENCE_THRESHOLD_DEFAULT = 0.4    # YOLO Confidence Threshold
CONE_REAL_HEIGHT_M = 0.3              # Real height of the cone in meters (for depth estimation)
FOCAL_LENGTH_PX = 700.0               # Camera focal length in pixels (approx)
DEBUG_MODE_DEFAULT = True             # Enable verbose logging by default
# ==============================================================================

class ConeDetectorNode(Node):
    def __init__(self):
        super().__init__('cone_detector_node')
        
        self.print_banner()
        self.get_logger().info("🔵 Initializing Cone Detector Node...")

        # ----------------------------------------------------------------------
        # 1. ROS PARAMETERS
        # ----------------------------------------------------------------------
        # We declare parameters so they can be changed at runtime without code edits
        self.declare_parameter('model_path', '/home/debraj/ros2_ws/src/cone_detector/cone_detector/models/best.pt')
        self.declare_parameter('image_topic', '/zed/zed_node/rgb/image_rect_color')
        
        # CV Params (initialized from defaults above)
        self.declare_parameter('use_clahe', USE_CLAHE_DEFAULT)
        self.declare_parameter('debug_mode', DEBUG_MODE_DEFAULT)
        self.declare_parameter('saturation_threshold', DEFAULT_SATURATION_THRESHOLD)
        self.declare_parameter('confidence_threshold', CONFIDENCE_THRESHOLD_DEFAULT)
        self.declare_parameter('min_aspect_ratio', MIN_ASPECT_RATIO)
        self.declare_parameter('max_aspect_ratio', MAX_ASPECT_RATIO)
        
        # Fetch values
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.use_clahe = self.get_parameter('use_clahe').get_parameter_value().bool_value
        self.debug_mode = self.get_parameter('debug_mode').get_parameter_value().bool_value
        self.sat_thresh = self.get_parameter('saturation_threshold').get_parameter_value().integer_value
        self.conf_thresh = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.min_ar = self.get_parameter('min_aspect_ratio').get_parameter_value().double_value
        self.max_ar = self.get_parameter('max_aspect_ratio').get_parameter_value().double_value

        # Log configuration
        if self.debug_mode:
            self.get_logger().info(f"🔧 CONFIG: CLAHE={self.use_clahe}, Conf={self.conf_thresh}, SatThresh={self.sat_thresh}")
            self.get_logger().info(f"🔧 CONFIG: Aspect Ratio Range [{self.min_ar} - {self.max_ar}]")

        # ----------------------------------------------------------------------
        # 2. MODEL LOADING
        # ----------------------------------------------------------------------
        # Robust model path resolution
        if not os.path.exists(self.model_path):
            self.get_logger().warn(f"⚠️ Model not found at absolute path: {self.model_path}. Searching relative...")
            script_dir = os.path.dirname(os.path.realpath(__file__))
            candidates = [
                os.path.join(script_dir, 'models', 'best.pt'),
                os.path.join(script_dir, 'best.pt')
            ]
            found = False
            for cand in candidates:
                if os.path.exists(cand):
                    self.model_path = cand
                    found = True
                    break
            if not found:
                self.get_logger().error(f"❌ FATAL: Could not find model file! Checked: {candidates}")
                self.destroy_node()
                return

        self.get_logger().info(f"🚀 Loading YOLO model from: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info("✅ YOLO model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"❌ CRITICAL: Failed to load YOLO model. Error: {e}")
            return

        # ----------------------------------------------------------------------
        # 3. SETUP COMMUNICATION
        # ----------------------------------------------------------------------
        self.bridge = CvBridge()

        # Subscriber
        self.get_logger().info(f"👂 Subscribing to topic: {self.image_topic}")
        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        # Publishers
        self.result_pub = self.create_publisher(Image, 'cone_detector/annotated_image', 10)
        self.detection_pub = self.create_publisher(String, '/cone_detector/detections', 10)
        
        self.get_logger().info("📢 Publishing annotated video to: cone_detector/annotated_image")
        self.get_logger().info("📢 Publishing JSON detections to: /cone_detector/detections")

        # ----------------------------------------------------------------------
        # 4. WATCHDOG & STATE
        # ----------------------------------------------------------------------
        self.last_image_time = self.get_clock().now()
        self.image_check_timer = self.create_timer(3.0, self.check_connection_status)
        self.first_image_received = False
        self.frame_count = 0 
        
        self.get_logger().info("🟢 Node setup complete. Waiting for images...")

    def print_banner(self):
        print(r"""
   ______                  ____       __            __            
  / ____/___  ____  ___   / __ \___  / /____  _____/ /_____  _____
 / /   / __ \/ __ \/ _ \ / / / / _ \/ __/ _ \/ ___/ __/ __ \/ ___/
/ /___/ /_/ / / / /  __// /_/ /  __/ /_/  __/ /__/ /_/ /_/ / /    
\____/\____/_/ /_/\___//_____/\___/\__/\___/\___/\__/\____/_/     
        """)

    def check_connection_status(self):
        """Warns if no images are received."""
        time_diff = (self.get_clock().now() - self.last_image_time).nanoseconds / 1e9
        if time_diff > 5.0 and not self.first_image_received:
            self.get_logger().warn(f"⏳ Still waiting for first image on {self.image_topic}... ({time_diff:.1f}s)")
        elif time_diff > 5.0:
            self.get_logger().error(f"⚠️ LOST CONNECTION: No images for {time_diff:.1f}s!")

    # ==========================================================================
    # CORE PROCESSING LOGIC
    # ==========================================================================

    def apply_clahe(self, image):
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the L-channel
        of a LAB image. This brings out details in shadows/dark areas.
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception as e:
            self.get_logger().error(f"Error in CLAHE: {e}")
            return image

    def get_color_label(self, hue):
        """Maps HSV Hue value (0-179) to a string Color Name."""
        if 0 <= hue < 10 or 170 <= hue <= 179: return "Red"
        elif 10 <= hue < 25: return "Orange"
        elif 25 <= hue < 35: return "Yellow"
        elif 35 <= hue < 85: return "Green"
        elif 85 <= hue < 130: return "Blue"
        elif 130 <= hue < 170: return "Purple"
        return "Unknown"

    def get_color_rgb(self, color_name):
        """Returns BGR tuple for visualization."""
        if color_name == "Yellow": return (0, 255, 255)
        elif color_name == "Blue": return (255, 0, 0)
        elif color_name == "Red": return (0, 0, 255)
        elif color_name == "Orange": return (0, 165, 255)
        elif color_name == "Green": return (0, 255, 0)
        elif color_name == "Purple": return (128, 0, 128)
        return (255, 255, 255)

    def image_callback(self, msg):
        """
        Main pipeline:
        1. Convert ROS Image -> OpenCV
        2. Preprocessing (CLAHE)
        3. Inference (YOLO)
        4. Filtering (Aspect Ratio, Masks, Color)
        5. Publishing (Annotated Image + JSON)
        """
        start_time = time.time()
        self.last_image_time = self.get_clock().now()

        # 1. Conversion
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"❌ Conversion Error: {e}")
            return

        if not self.first_image_received:
            self.get_logger().info(f"📸 First Image Received! Size: {frame.shape}")
            self.first_image_received = True

        self.frame_count += 1
        
        # 2. Preprocessing
        processing_frame = frame.copy()
        if self.use_clahe:
            processing_frame = self.apply_clahe(processing_frame)

        # 3. Inference
        try:
            # Verbose=False prevents YOLO from printing to stdout every frame
            results = self.model.predict(
                processing_frame, 
                conf=self.conf_thresh, 
                verbose=False
            )[0]
        except Exception as e:
            self.get_logger().error(f"❌ Inference Error: {e}")
            return

        # Prepare annotation
        annotated_frame = processing_frame.copy() # Draw on the processed (CLAHE) frame
        detections_list = []
        
        # 4. Processing Detections
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                
                # Boundary Checks
                h_img, w_img = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                w_box = x2 - x1
                h_box = y2 - y1
                
                if w_box <= 0 or h_box <= 0:
                    continue

                # --- A. Aspect Ratio Filter ---
                aspect_ratio = w_box / float(h_box)
                if aspect_ratio < self.min_ar or aspect_ratio > self.max_ar:
                    # Debug draw for rejected cones
                    if self.debug_mode:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 0), 1) # Black box for rejected
                    continue

                # --- B. ROI Extraction & Masking ---
                roi = processing_frame[y1:y2, x1:x2]
                if roi.size == 0: continue

                # Create Triangle Mask (Shrunk by 10-15% to check center color)
                margin_x = int(w_box * 0.15)
                margin_y = int(h_box * 0.10)
                
                # Check if box is too small for margins
                if margin_x < 1: margin_x = 1
                if margin_y < 1: margin_y = 1
                
                pts = np.array([
                    [w_box // 2, margin_y],         # Top
                    [margin_x, h_box - margin_y],   # Bottom Left
                    [w_box - margin_x, h_box - margin_y] # Bottom Right
                ], np.int32)

                mask = np.zeros((h_box, w_box), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)

                # --- C. Color Statistics ---
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # Select pixels inside the triangle mask
                hue_values = hsv_roi[:, :, 0][mask > 0]
                sat_values = hsv_roi[:, :, 1][mask > 0]
                val_values = hsv_roi[:, :, 2][mask > 0]
                
                if len(hue_values) == 0: continue

                avg_saturation = np.mean(sat_values)
                avg_brightness = np.mean(val_values)
                
                # Find dominant hue (mode)
                vals, counts = np.unique(hue_values, return_counts=True)
                mode_hue = vals[np.argmax(counts)]

                # --- D. Validity Check ---
                if avg_saturation < self.sat_thresh or avg_brightness < DEFAULT_BRIGHTNESS_THRESHOLD:
                    if self.debug_mode:
                         # Draw "Dim" label for debugging
                         cv2.putText(annotated_frame, "Dim", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)
                         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (100,100,100), 1)
                    continue
                
                # Valid Detection
                color_name = self.get_color_label(mode_hue)
                color_rgb = self.get_color_rgb(color_name)
                
                # Depth Estimation
                depth_m = (FOCAL_LENGTH_PX * CONE_REAL_HEIGHT_M) / h_box
                
                # --- E. Drawing & Output ---
                label = f"{color_name} {depth_m:.1f}m"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_rgb, 2)
                
                # Draw the triangle to show what we sampled
                pts_global = pts + [x1, y1]
                cv2.polylines(annotated_frame, [pts_global], True, (255, 255, 255), 1)
                
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)
                
                detections_list.append({
                    "class": "cone",
                    "color": color_name.lower(),
                    "confidence": float(conf),
                    "depth_m": float(depth_m),
                    "center": [int((x1+x2)/2), int((y1+y2)/2)],
                    "bbox": [x1, y1, x2, y2]
                })

        # 5. Publish Results
        # JSON
        json_msg = {"detections": detections_list, "timestamp": start_time}
        self.detection_pub.publish(String(data=json.dumps(json_msg)))
        
        # Image
        # Add status overlay
        status_text = f"CLAHE: {'ON' if self.use_clahe else 'OFF'} | FPS: {1.0/(time.time()-start_time + 1e-6):.1f}"
        cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            out_msg.header = msg.header
            self.result_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")

        # Periodic log
        if self.frame_count % 100 == 0:
            self.get_logger().info(f"✅ Processed {self.frame_count} frames. Recent detections: {len(detections_list)}")

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

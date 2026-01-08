#!/usr/bin/env python3
"""
Cone Detector – robust image decoding + bgra8 / bgr8 publishing
All debug output is now via ROS INFO (no Python logging)
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import numpy as np
import os
import json
from typing import List, Dict, Any, Tuple

# --------------------------------------------------------------------------- #
# Colour utilities
# --------------------------------------------------------------------------- #
def hsv_to_color_name(h: int, s: int, v: int) -> str:
    if v < 40: return "Black"
    if s < 30 and v > 200: return "White"
    if s < 30: return "Gray"
    if h <= 10 or h >= 170: return "Red"
    if 10 < h <= 25: return "Orange"
    if 25 < h <= 35: return "Yellow"
    if 35 < h <= 85: return "Green"
    if 85 < h <= 130: return "Blue"
    if 130 < h < 170: return "Magenta"
    return "Unknown"


def class_color_palette(n: int) -> List[Tuple[int, int, int]]:
    """BGR palette."""
    colors = []
    for i in range(max(1, n)):
        hue = int(179 * i / max(1, n))
        bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(map(int, bgr)))
    return colors


def draw_triangle_bbox(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       color: Tuple[int, int, int]):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    pts = np.array([(cx, y1), (x1, y2), (x2, y2)], np.int32)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 3, color, -1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Node
# --------------------------------------------------------------------------- #
class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')

        # ----- parameters ---------------------------------------------------- #
        self.declare_parameter('camera_topic', '/zed/zed_node/left/image_rect_color')
        self.declare_parameter('depth_topic', '/zed/zed_node/depth/depth_registered')
        self.declare_parameter('model_path',
                               '/home/neel/ros2_ws/src/cone_detector/cone_detector/models/best.pt')
        self.declare_parameter('confidence', 0.25)
        self.declare_parameter('annotated_topic', '/cone_detector/annotated')
        self.declare_parameter('detection_topic', '/cone_detector/detections')
        self.declare_parameter('image_encoding', '')          # e.g. bgr8, bgra8
        self.declare_parameter('annotated_encoding', 'bgr8')   # bgr8 = RViz, bgra8 = image_view

        self.camera_topic = self.get_parameter('camera_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.model_path = os.path.expanduser(self.get_parameter('model_path').value)
        self.conf = float(self.get_parameter('confidence').value)
        self.annotated_topic = self.get_parameter('annotated_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.force_input_enc = self.get_parameter('image_encoding').value.strip()
        self.annotated_enc = self.get_parameter('annotated_encoding').value.strip()

        # ----- CvBridge ------------------------------------------------------ #
        self.bridge = CvBridge()

        # ----- load YOLO ----------------------------------------------------- #
        if not os.path.isfile(self.model_path):
            self.get_logger().error(f'MODEL NOT FOUND: {self.model_path}')
            raise FileNotFoundError(self.model_path)
        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.names = getattr(self.model, 'names', {})
        self.palette = class_color_palette(len(self.names))  # BGR

        # ----- subscribers --------------------------------------------------- #
        self.create_subscription(RosImage, self.camera_topic,
                                 self.image_callback, 10)
        self.create_subscription(RosImage, self.depth_topic,
                                 self.depth_callback, 10)

        # ----- publishers ---------------------------------------------------- #
        self.pub_img = self.create_publisher(RosImage, self.annotated_topic, 10)
        self.pub_det = self.create_publisher(String, self.detection_topic, 10)

        # ----- runtime ------------------------------------------------------- #
        self.depth_image: np.ndarray | None = None
        self.frame_id: str = 'camera_frame'
        self._cone_counter = 0          # per-frame counter for info messages
        self.get_logger().info('ConeDetector ready.')

    # --------------------------------------------------------------------- #
    def depth_callback(self, msg: RosImage):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            self.depth_image = depth
            self.frame_id = msg.header.frame_id
        except CvBridgeError as e:
            self.get_logger().warn(f'Depth conversion error: {e}')

    # --------------------------------------------------------------------- #
    def image_callback(self, msg: RosImage):
        rgb: np.ndarray | None = None

        # ----- reset per-frame cone counter -------------------------------- #
        self._cone_counter = 0

        # ----- build list of encodings to try -------------------------------- #
        encodings = []
        if self.force_input_enc:
            encodings.append(self.force_input_enc)
        encodings += ['bgr8', 'rgb8', 'bgra8', 'mono8']

        for enc in encodings:
            try:
                cv_raw = self.bridge.imgmsg_to_cv2(msg)
                if cv_raw is None or cv_raw.size == 0:
                    continue

                # ---- convert to RGB -------------------------------------------------
                if enc in ('bgr8', 'bgra8'):
                    rgb = cv2.cvtColor(cv_raw, cv2.COLOR_BGR2RGB)
                elif enc == 'rgb8':
                    rgb = cv_raw.copy()
                elif enc == 'mono8':
                    rgb = cv2.cvtColor(cv_raw, cv2.COLOR_GRAY2RGB)
                else:
                    continue

                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    rgb = None
                    continue

                self.get_logger().info(f'Image decoded with encoding: {enc}')
                break
            except CvBridgeError as e:
                self.get_logger().info(f'CvBridge failed ({enc}): {e}')
            except Exception as e:
                self.get_logger().info(f'Unexpected error ({enc}): {e}')

        if rgb is None:
            self.get_logger().error('Could not decode RGB image – skipping frame')
            return

        self.frame_id = msg.header.frame_id

        # ------------------- YOLO inference --------------------------------- #
        results = self.model(rgb, conf=self.conf, verbose=False)
        annotated = rgb.copy()
        detections: List[Dict[str, Any]] = []

        for r in results:
            boxes = r.boxes
            if not boxes:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                name = self.names.get(cls, f'class_{cls}')

                # ----- depth ------------------------------------------------- #
                depth_val = -1.0
                raw_depth = None
                if (self.depth_image is not None
                        and 0 <= cy < self.depth_image.shape[0]
                        and 0 <= cx < self.depth_image.shape[1]):
                    raw_depth = float(self.depth_image[cy, cx])
                    if not (np.isnan(raw_depth) or np.isinf(raw_depth) or raw_depth <= 0):
                        depth_val = raw_depth

                # ----- colour (BGR palette → RGB for drawing) -------------- #
                bgr_col = self.palette[cls % len(self.palette)]
                rgb_col = tuple(reversed(bgr_col))

                # ----- draw triangle bbox ----------------------------------- #
                draw_triangle_bbox(annotated, x1, y1, x2, y2, rgb_col)

                # ----- cone colour from centre patch ------------------------ #
                PATCH = 20
                y1p = max(0, cy - PATCH)
                y2p = min(annotated.shape[0], cy + PATCH)
                x1p = max(0, cx - PATCH)
                x2p = min(annotated.shape[1], cx + PATCH)
                patch = annotated[y1p:y2p, x1p:x2p]

                cone_color = "Unknown"
                patch_mean_rgb = None
                hsv_vals = None

                if patch.size > 0:
                    # patch is **RGB**
                    mean_rgb = cv2.mean(patch)[:3]          # (R, G, B)
                    patch_mean_rgb = tuple(map(int, mean_rgb))

                    # Convert RGB → BGR for OpenCV
                    b, g, r = mean_rgb[2], mean_rgb[1], mean_rgb[0]
                    colour_uint8 = np.uint8([[[b, g, r]]])

                    try:
                        hsv = cv2.cvtColor(colour_uint8, cv2.COLOR_BGR2HSV)
                        h, s, v = hsv[0, 0]
                        hsv_vals = (int(h), int(s), int(v))
                        cone_color = hsv_to_color_name(*hsv_vals)
                    except Exception as exc:
                        self.get_logger().info(f"HSV conversion failed for cone {self._cone_counter}: {exc}")

                # ----------------------------------------------------------------- #
                # PER-CONE INFO (replaces logger.debug)
                # ----------------------------------------------------------------- #
                self.get_logger().info(
                    f"Cone {self._cone_counter:03d} | class={name} | conf={conf:.2f} | "
                    f"center=({cx},{cy}) | raw_depth={raw_depth} | depth={depth_val if depth_val > 0 else -1:.2f} m | "
                    f"patch_mean_RGB={patch_mean_rgb} | HSV={hsv_vals} | colour={cone_color}"
                )

                # ----- label ------------------------------------------------ #
                label = f'{name} {conf:.2f} {cone_color}'
                label += f' {depth_val:.2f}m' if depth_val > 0 else ' N/A'
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb_col, 2, cv2.LINE_AA)

                # ----- JSON detection --------------------------------------- #
                detections.append({
                    "class": name,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy],
                    "depth_m": depth_val if depth_val > 0 else None,
                    "color": cone_color
                })

                # increment counter
                self._cone_counter += 1

        # ------------------- publish annotated image ----------------------- #
        try:
            bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            enc = self.annotated_enc
            if enc == 'bgra8':
                h, w = bgr.shape[:2]
                alpha = np.full((h, w, 1), 255, dtype=np.uint8)
                img_out = np.dstack((bgr, alpha))
            else:
                img_out = bgr
                enc = 'bgr8'

            ann_msg = self.bridge.cv2_to_imgmsg(img_out, encoding=enc)
            ann_msg.header.stamp = msg.header.stamp
            ann_msg.header.frame_id = self.frame_id
            self.pub_img.publish(ann_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish annotated image: {e}')

        # ------------------- publish JSON detections ----------------------- #
        if detections:
            payload = {
                "frame_id": self.frame_id,
                "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "width": msg.width,
                "height": msg.height,
                "detections": detections
            }
            json_msg = String()
            json_msg.data = json.dumps(payload)
            self.pub_det.publish(json_msg)

            self.get_logger().info(f'Detected {len(detections)} cone(s)')


# --------------------------------------------------------------------------- #
def main():
    rclpy.init()
    try:
        node = ConeDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'FATAL: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
"""
=========================================================
Thermal Callback
Assign thermal information to each detected person.
=========================================================
"""

from __future__ import annotations

import cv2
import numpy as np

from core.detection import Detection

class ThermalCallback:
    def __init__(
        self,
        thermal_width: int = 320,
        thermal_height: int = 240
    ):
        self.thermal_width = thermal_width
        self.thermal_height = thermal_height

    def process(
        self,
        detections: list[Detection],
        thermal_frame: np.ndarray,
        rgb_width: int,
        rgb_height: int
    ) -> list[Detection]:
        if thermal_frame is None:
            return detections
        h, w = thermal_frame.shape
        scale_x = w / rgb_width
        scale_y = h / rgb_height

        for det in detections:
            if det.label != "person":
                continue
            x1 = int(det.bbox.x1 * scale_x)
            y1 = int(det.bbox.y1 * scale_y)
            x2 = int(np.ceil(det.bbox.x2 * scale_x))
            y2 = int(np.ceil(det.bbox.y2 * scale_y))
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            roi = thermal_frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            det.temperature = round(float(np.max(roi)), 1)
            # Cross-Modal Verification: Xác thực dấu hiệu sinh tồn nhiệt người
            if 30.0 <= det.temperature <= 39.0:
                det.thermal_verified = True
                if det.confidence is not None:
                    det.confidence = min(0.99, round(det.confidence + 0.12, 2))
        return detections

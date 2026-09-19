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
            x2 = int(det.bbox.x2 * scale_x)
            y2 = int(det.bbox.y2 * scale_y)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            roi = thermal_frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            det.temperature = float(np.max(roi))
        return detections

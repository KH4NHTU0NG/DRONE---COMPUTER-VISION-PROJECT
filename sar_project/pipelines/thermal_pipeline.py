"""
=========================================================
Thermal Processing Pipeline
=========================================================
"""

from __future__ import annotations

import cv2
import numpy as np

from camera.thermal_camera import ThermalCamera
from config import config


class ThermalPipeline:
    def __init__(self):
        self.enabled = config.thermal.enable
        # Chỉ khởi tạo phần cứng (mở bus I2C) khi cấu hình bật thermal và phần cứng khả dụng,
        # tránh crash trên máy chưa cắm cảm biến MLX90640 hoặc môi trường test.
        self.camera = None
        if self.enabled:
            try:
                self.camera = ThermalCamera()
            except Exception:
                self.camera = None
        self.frame = None
        self.heatmap = None

    def start(self):
        if self.camera is not None:
            self.camera.start()

    def stop(self):
        if self.camera is not None:
            self.camera.stop()

    def process(self):
        if self.camera is None:
            return None
        frame = self.camera.read()
        if frame is None:
            return None
        self.frame = frame
        return frame

    def resize(
        self,
        width: int = 320,
        height: int = 240
    ):
        if self.frame is None:
            return None
        return cv2.resize(
            self.frame,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

    def normalize(self):
        if self.frame is None:
            return None
        t_min = config.thermal.min_temperature
        t_max = config.thermal.max_temperature
        if t_max <= t_min:
            t_max = t_min + 1.0
        clipped = np.clip(self.frame, t_min, t_max)
        norm = ((clipped - t_min) / (t_max - t_min) * 255.0).astype(np.uint8)
        return norm

    def generate_heatmap(self):
        image = self.normalize()
        if image is None:
            return None
        image = cv2.resize(
            image,
            (320, 240),
            interpolation=cv2.INTER_CUBIC
        )
        self.heatmap = cv2.applyColorMap(
            image,
            cv2.COLORMAP_JET
        )
        return self.heatmap

    def max_temperature(self):
        if self.frame is None:
            return None
        return float(np.max(self.frame))

    def min_temperature(self):
        if self.frame is None:
            return None
        return float(np.min(self.frame))

    def average_temperature(self):
        if self.frame is None:
            return None
        return float(np.mean(self.frame))

    def get_heatmap(self):
        return self.heatmap

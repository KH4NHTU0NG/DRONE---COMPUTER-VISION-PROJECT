"""
=========================================================
Global Application Context
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import threading
import numpy as np

from core.detection import Detection


@dataclass(slots=True)
class ApplicationContext:
    # Latest RGB frame
    frame: np.ndarray | None = None
    # Detection results
    detections: List[Detection] = field(default_factory=list)
    # FPS
    fps: float = 0.0
    # Running state
    running: bool = False
    # Frame counter
    frame_count: int = 0
    # Thread lock
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame

    def set_detections(self, detections):
        with self.lock:
            self.detections = detections

    def get_detections(self):
        with self.lock:
            return self.detections

    def set_fps(self, fps):
        with self.lock:
            self.fps = fps

    def get_fps(self):
        with self.lock:
            return self.fps

    def increment_frame(self):
        with self.lock:
            self.frame_count += 1

    def reset(self):
        with self.lock:
            self.frame = None
            self.detections.clear()
            self.frame_count = 0
            self.fps = 0.0
            self.running = False

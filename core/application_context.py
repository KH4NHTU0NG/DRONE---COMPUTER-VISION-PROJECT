"""
=========================================================
Global Application Context & Asynchronous Dispatcher
Designed for Raspberry Pi 5 Multi-Core Throughput
=========================================================
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from core.detection import Detection


@dataclass(slots=True)
class ApplicationContext:
    # Ring queue for non-blocking stream / rendering decoupling
    render_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=3))
    # Latest telemetry state
    latest_frame: np.ndarray | None = None
    latest_detections: List[Detection] = field(default_factory=list)
    latest_fps: float = 0.0
    latest_thermal_heatmap: np.ndarray | None = None
    running: bool = True
    frame_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push_inference_result(
        self,
        frame: np.ndarray | None,
        detections: List[Detection],
        fps: float,
        thermal_heatmap: np.ndarray | None = None
    ) -> None:
        """Đẩy kết quả nhận diện vào queue không đồng bộ. Không bao giờ chặn pipeline chính."""
        with self.lock:
            self.frame_count += 1
            self.latest_detections = detections
            self.latest_fps = fps
            if frame is not None:
                self.latest_frame = frame
            if thermal_heatmap is not None:
                self.latest_thermal_heatmap = thermal_heatmap

        if frame is not None:
            # Drop frame cũ nếu render worker chưa kịp xử lý để giữ zero-latency
            try:
                self.render_queue.put_nowait((frame, detections, fps, thermal_heatmap))
            except queue.Full:
                try:
                    self.render_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.render_queue.put_nowait((frame, detections, fps, thermal_heatmap))
                except queue.Full:
                    pass

    def get_render_task(self, timeout: float = 0.1) -> Tuple[np.ndarray, List[Detection], float, np.ndarray | None] | None:
        try:
            return self.render_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def reset(self):
        with self.lock:
            self.latest_frame = None
            self.latest_detections.clear()
            self.latest_thermal_heatmap = None
            self.frame_count = 0
            self.latest_fps = 0.0
            self.running = False
        while not self.render_queue.empty():
            try:
                self.render_queue.get_nowait()
            except queue.Empty:
                break

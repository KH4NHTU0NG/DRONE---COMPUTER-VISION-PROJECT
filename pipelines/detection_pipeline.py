"""
=========================================================
SAR Detection Pipeline for macOS & Local Inference
Supports:
- iPhone Continuity Camera / AVFoundation / IP Video Stream
- Apple Silicon MPS (Metal Performance Shaders) / CPU YOLOv8
- Cocoa GUI Rendering on Main Thread (macOS AppKit Compliant)
- MJPEG Web Streaming (http://localhost:5001)
=========================================================
"""

import sys
import threading
import time
from typing import Optional
import cv2
import numpy as np

from camera.iphone_camera import IPhoneCamera
from callbacks.detection_callback import DetectionCallback
from callbacks.thermal_callback import ThermalCallback
from callbacks.fusion_callback import FusionCallback
from pipelines.thermal_pipeline import ThermalPipeline
from pipelines.tracker import DummyTracker
from ui.renderer import Renderer
from ui.display_manager import DisplayManager
from ui.stream_server import StreamServer
from core.application_context import ApplicationContext
from core.fps import FPSCounter
from core.logger import logger
from config import config


class DetectionPipeline:
    def __init__(self):
        self.context = ApplicationContext()
        self.detection_callback = DetectionCallback()
        self.thermal_callback = ThermalCallback()
        self.fusion_callback = FusionCallback()
        self.tracker = DummyTracker()
        self.thermal_pipeline = ThermalPipeline()
        self.renderer = Renderer()
        self.display = DisplayManager(
            window_name=config.display.window_name,
            width=config.camera.width,
            height=config.camera.height
        )
        self.fps = FPSCounter()
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_source: Optional[IPhoneCamera] = None
        self._running = False

        if config.display.fullscreen:
            self.display.toggle_fullscreen()

        port = getattr(config.performance, "stream_port", 5001)
        self.stream_server = StreamServer(port=port)
        self.stream_server.start()

    def _open_camera(self) -> bool:
        backend = config.camera.backend.lower()

        # 1. iPhone Continuity Camera / macOS AVFoundation / IP Stream
        if backend in ("iphone", "avfoundation", "mac", "stream", "ip_stream"):
            logger.info("Opening iPhone Continuity / macOS Camera...")
            self.camera_source = IPhoneCamera(
                device_id=config.camera.device_id,
                stream_url=config.camera.stream_url,
                width=config.camera.width,
                height=config.camera.height,
                fps=config.camera.fps,
                flip_method=config.camera.flip_method
            )
            self.camera_source.start()
            return True

        # 2. V4L2 USB Camera (nếu cắm trên Linux/Mac)
        elif backend == "v4l2":
            dev_id = getattr(config.camera, "device_id", 0)
            logger.info(f"Opening USB Camera at index {dev_id}...")
            cap = cv2.VideoCapture(dev_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
                cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
                self.cap = cap
                return True

        # 3. Chế độ mô phỏng / Không có camera phần cứng
        logger.warning("No hardware camera configured or opened. Running in offline/simulation mode.")
        return False

    def run(self):
        if config.thermal.enable:
            logger.info("Starting Thermal Pipeline...")
            self.thermal_pipeline.start()

        logger.info("Starting AI Detection Pipeline on macOS...")
        self._running = True
        self._open_camera()

        try:
            while self._running and self.context.running:
                frame = None

                # Lấy frame từ nguồn camera iPhone
                if self.camera_source is not None:
                    ret, frame = self.camera_source.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue
                elif self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue
                else:
                    # Giả lập khung hình test khi không có camera
                    frame = np.zeros((config.camera.height, config.camera.width, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "SIMULATION MODE - NO CAMERA",
                        (50, config.camera.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )
                    time.sleep(1.0 / max(1, config.camera.fps))

                h, w = frame.shape[:2]

                # 1. Suy luận AI với YOLOv8 (Apple Silicon MPS / CPU)
                detections = self.detection_callback.process(frame, w, h)

                # 2. Xử lý thân nhiệt (nếu bật cảm biến hoặc mock)
                thermal_frame = None
                if config.thermal.enable:
                    thermal_frame = self.thermal_pipeline.process()
                    detections = self.thermal_callback.process(
                        detections,
                        thermal_frame,
                        w,
                        h
                    )

                # 3. Tracking & Tính điểm ưu tiên SAR
                detections = self.tracker.update(detections)
                detections = self.fusion_callback.process(detections)
                fps = self.fps.update()

                # 4. Tạo heatmap nếu có dữ liệu nhiệt
                heatmap = None
                if thermal_frame is not None and config.thermal.enable:
                    heatmap = self.thermal_pipeline.generate_heatmap()

                # 5. Render đồ họa & Hiển thị cửa sổ TRỰC TIẾP TRÊN MAIN THREAD
                # (Bắt buộc theo chuẩn Apple AppKit / Cocoa UI trên macOS)
                rendered = self.renderer.draw(frame, detections, fps)
                self.display.show(rendered)

                # 6. Phát luồng Web MJPEG cho trình duyệt qua StreamServer
                self.stream_server.update(rendered, name="rgb")
                if heatmap is not None:
                    self.stream_server.update(heatmap, name="thermal")

                # 7. Kiểm tra sự kiện đóng cửa sổ (phím q hoặc ESC)
                if self.display.should_close():
                    logger.info("Window close requested by user. Shutting down...")
                    break

                # 8. Cập nhật kết quả vào ApplicationContext
                self.context.push_inference_result(frame, detections, fps, heatmap)

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down...")
        finally:
            self.stop()

    def stop(self):
        logger.info("Stopping SAR Pipeline on macOS...")
        self._running = False
        self.context.running = False

        if self.camera_source is not None:
            try:
                self.camera_source.stop()
            except Exception:
                pass
            self.camera_source = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        try:
            self.stream_server.stop()
        except Exception as e:
            logger.warning(f"Error stopping stream server: {e}")

        try:
            self.thermal_pipeline.stop()
        except Exception as e:
            logger.warning(f"Error stopping thermal pipeline: {e}")

        try:
            self.display.close()
        except Exception as e:
            logger.warning(f"Error closing display: {e}")

        logger.info("macOS pipeline stopped cleanly.")

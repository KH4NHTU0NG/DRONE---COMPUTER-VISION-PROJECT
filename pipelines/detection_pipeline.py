"""
=========================================================
SAR Detection Pipeline for NVIDIA Jetson Orin Nano
Hardware Accelerated with TensorRT FP16 & GStreamer (nvargus/NVMM)
Multi-Core Asynchronous Decoupled Ring Queue
=========================================================
"""

import sys
import threading
import time
import cv2
import numpy as np

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


def get_jetson_gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    framerate: int = 30,
    flip_method: int = 0
) -> str:
    """Tạo chuỗi GStreamer tối ưu cho camera CSI qua ISP Jetson Orin Nano (Zero-Copy NVMM)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){capture_width}, height=(int){capture_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1"
    )


def get_v4l2_pipeline(device_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> str:
    """Chuỗi GStreamer cho USB Webcam trên Jetson."""
    return (
        f"v4l2src device=/dev/video{device_id} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1"
    )


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
        self.cap = None
        self._running = False
        self._capture_thread = None

        if config.display.fullscreen:
            self.display.toggle_fullscreen()

        port = getattr(config.performance, "stream_port", 5000)
        self.stream_server = StreamServer(port=port)
        self.stream_server.start()

        # Khởi chạy luồng render & web stream độc lập
        self._worker_thread = threading.Thread(
            target=self._render_worker_loop,
            daemon=True
        )
        self._worker_thread.start()

    def _open_camera(self) -> cv2.VideoCapture | None:
        backend = config.camera.backend.lower()
        cap = None

        if backend == "nvargus":
            gst_pipeline = get_jetson_gstreamer_pipeline(
                sensor_id=config.camera.sensor_id,
                capture_width=config.camera.width,
                capture_height=config.camera.height,
                framerate=config.camera.fps,
                flip_method=config.camera.flip_method
            )
            logger.info(f"Opening Jetson CSI Camera via GStreamer: {gst_pipeline}")
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        elif backend == "v4l2":
            gst_pipeline = get_v4l2_pipeline(
                device_id=config.camera.device_id,
                width=config.camera.width,
                height=config.camera.height,
                fps=config.camera.fps
            )
            logger.info(f"Opening V4L2 USB Camera: {gst_pipeline}")
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                cap = cv2.VideoCapture(config.camera.device_id)

        if cap is not None and cap.isOpened():
            logger.info("Camera opened successfully.")
            return cap

        logger.warning("No hardware camera opened. Running in offline/simulation mode.")
        return None

    def run(self):
        logger.info("Starting Thermal Camera on Jetson...")
        self.thermal_pipeline.start()
        logger.info("Starting AI Detection Pipeline on Jetson Orin Nano...")
        self._running = True
        self.cap = self._open_camera()

        try:
            while self._running and self.context.running:
                frame = None
                if self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue
                else:
                    # Chế độ giả lập khung hình kiểm thử
                    frame = np.zeros((config.camera.height, config.camera.width, 3), dtype=np.uint8)
                    time.sleep(1.0 / max(1, config.camera.fps))

                h, w = frame.shape[:2]

                # 1. Suy luận AI với TensorRT FP16 trên GPU Orin Nano
                detections = self.detection_callback.process(frame, w, h)

                # 2. Thu nhận nhiệt độ & Cross-modal Verification
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
                if thermal_frame is not None:
                    heatmap = self.thermal_pipeline.generate_heatmap()

                # 5. Đẩy sang Ring Queue không đồng bộ (Không khóa luồng suy luận)
                self.context.push_inference_result(frame, detections, fps, heatmap)

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down...")
        finally:
            self.stop()

    def _render_worker_loop(self):
        """Luồng chuyên trách render đồ họa và phát video qua mạng, không chặn GPU AI."""
        while self.context.running:
            task = self.context.get_render_task(timeout=0.05)
            if task is None:
                continue

            frame, detections, fps, heatmap = task
            if frame is not None:
                rendered = self.renderer.draw(frame, detections, fps)
                self.display.show(rendered)
                self.stream_server.update(rendered, name="rgb")

            if heatmap is not None:
                self.stream_server.update(heatmap, name="thermal")

            if self.display.should_close():
                self.stop()

    def stop(self):
        logger.info("Stopping SAR Jetson Pipeline...")
        self._running = False
        self.context.running = False

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

        logger.info("Jetson pipeline stopped cleanly.")

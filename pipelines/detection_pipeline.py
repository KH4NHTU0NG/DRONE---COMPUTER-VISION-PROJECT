"""
=========================================================
SAR Detection Pipeline
Built on top of Hailo GStreamerDetectionApp
Optimized for Raspberry Pi 5 with Multi-Core Decoupled Pipeline
=========================================================
"""

from pathlib import Path
import os
import sys
import threading
import cv2

try:
    from hailo_apps.python.pipeline_apps.detection.detection_pipeline import (
        GStreamerDetectionApp,
    )
    from hailo_apps.python.core.common.buffer_utils import (
        get_caps_from_pad,
        get_numpy_from_buffer,
    )
except (ImportError, ModuleNotFoundError):
    class GStreamerDetectionApp:
        def __init__(self, *args, **kwargs):
            pass

        def get_pipeline_string(self):
            return ""

        def run(self):
            pass

    def get_caps_from_pad(pad):
        return None, 640, 640

    def get_numpy_from_buffer(buffer, fmt, width, height):
        return None

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


class _HeadlessGStreamerDetectionApp(GStreamerDetectionApp):
    def get_pipeline_string(self):
        pipeline_string = super().get_pipeline_string()
        return pipeline_string.replace("autovideosink", "fakesink")


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
        self.user_data = self.detection_callback
        if config.display.fullscreen:
            self.display.toggle_fullscreen()

        project_root = Path(__file__).resolve().parent.parent
        env_file = project_root / ".env"
        if env_file.exists():
            os.environ["HAILO_ENV_FILE"] = str(env_file)

        self.pipeline = _HeadlessGStreamerDetectionApp(
            self._app_callback,
            self.user_data
        )
        port = getattr(config.performance, "stream_port", 5000)
        self.stream_server = StreamServer(port=port)
        self.stream_server.start()

        # Luồng render & stream độc lập trên nhân CPU riêng, giải phóng luồng chính
        self._worker_thread = threading.Thread(
            target=self._render_worker_loop,
            daemon=True
        )
        self._worker_thread.start()

    def _app_callback(self, element, buffer, user_data):
        """Callback suy luận AI chạy ở tốc độ tối đa của NPU và Camera ISP."""
        if buffer is None:
            return 0
        pad = element.get_static_pad("src")
        fmt, width, height = get_caps_from_pad(pad)
        frame = None
        if fmt is not None:
            frame = get_numpy_from_buffer(buffer, fmt, width, height)

        # 1. Trích xuất bounding box từ Hailo NPU
        detections = self.detection_callback.process(buffer, width, height)

        # 2. Đọc nhiệt độ và kiểm chứng chéo (Thermal Cross-Verification)
        thermal_frame = self.thermal_pipeline.process()
        detections = self.thermal_callback.process(
            detections,
            thermal_frame,
            width,
            height
        )

        # 3. Theo dõi mục tiêu và tính toán thứ tự ưu tiên SAR
        detections = self.tracker.update(detections)
        detections = self.fusion_callback.process(detections)
        fps = self.fps.update()

        # 4. Chuẩn bị ảnh và đẩy vào Ring Queue không đồng bộ (Zero-Locking)
        bgr_frame = None
        if frame is not None:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if fmt == "RGB" else frame

        heatmap = None
        if thermal_frame is not None:
            heatmap = self.thermal_pipeline.generate_heatmap()

        self.context.push_inference_result(bgr_frame, detections, fps, heatmap)
        return 0

    def _render_worker_loop(self):
        """Luồng chuyên trách render đồ họa và phát video qua mạng, không chặn AI."""
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

    def run(self):
        logger.info("Starting Thermal Camera...")
        self.thermal_pipeline.start()
        logger.info("Starting Detection Pipeline...")
        try:
            self.pipeline.run()
        finally:
            self.stop()

    def stop(self):
        logger.info("Stopping SAR Application...")
        self.context.running = False
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
        try:
            if hasattr(self.pipeline, "pipeline") and self.pipeline.pipeline:
                import gi
                gi.require_version('Gst', '1.0')
                from gi.repository import Gst
                self.pipeline.pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            if hasattr(self.pipeline, "loop") and self.pipeline.loop:
                self.pipeline.loop.quit()
        except Exception:
            pass
        sys.exit(0)

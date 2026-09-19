"""
=========================================================
SAR Detection Pipeline
Built on top of Hailo GStreamerDetectionApp
=========================================================
"""

from pathlib import Path
import os

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import (
    GStreamerDetectionApp,
)
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)

from callbacks.detection_callback import DetectionCallback
from callbacks.thermal_callback import ThermalCallback
from callbacks.fusion_callback import FusionCallback
from pipelines.thermal_pipeline import ThermalPipeline
from pipelines.tracker import DummyTracker
from ui.renderer import Renderer
from ui.display_manager import DisplayManager
from ui.stream_server import StreamServer
from core.fps import FPSCounter
from core.logger import logger
from config import config

class _HeadlessGStreamerDetectionApp(GStreamerDetectionApp):
    def get_pipeline_string(self):
        pipeline_string = super().get_pipeline_string()
        return pipeline_string.replace("autovideosink", "fakesink")

class DetectionPipeline:
    def __init__(self):
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
        self.stream_server = StreamServer(port=5000)
        self.stream_server.start()

    def _app_callback(self, element, buffer, user_data):
        if buffer is None:
            return 0
        pad = element.get_static_pad("src")
        fmt, width, height = get_caps_from_pad(pad)
        frame = None
        if fmt is not None:
            frame = get_numpy_from_buffer(buffer, fmt, width, height)
        detections = self.detection_callback.process(buffer, width, height)
        thermal_frame = self.thermal_pipeline.process()
        detections = self.thermal_callback.process(
            detections,
            thermal_frame,
            width,
            height
        )
        detections = self.tracker.update(detections)
        detections = self.fusion_callback.process(detections)
        fps = self.fps.update()
        if frame is not None:
            frame = self.renderer.draw(frame, detections, fps)
            brg_frame = frame[:, :, ::-1]
            self.display.show(brg_frame)
            self.stream_server.update(brg_frame, name="rgb")
        if thermal_frame is not None:
            heatmap = self.thermal_pipeline.generate_heatmap()
            if heatmap is not None:
                self.stream_server.update(heatmap, name="thermal")
        if self.display.should_close():
            self.stop()
        return 0

    def run(self):
        logger.info("Starting Thermal Camera")
        self.thermal_pipeline.start()
        logger.info("Starting Detection Pipeline")
        try:
            self.pipeline.run()
        finally:
            self.stop()

    def stop(self):
        logger.info("Stopping...")
        self.thermal_pipeline.stop()
        self.display.close()
        os._exit(0)

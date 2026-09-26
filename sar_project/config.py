"""
=========================================================
Global configuration for Search and Rescue (SAR) Project
Platform : Raspberry Pi 5 + Hailo AI HAT+
=========================================================
"""

from dataclasses import dataclass, field
from pathlib import Path

# =========================================================
# PATH CONFIGURATION
# =========================================================

@dataclass(slots=True)
class PathConfig:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    @property
    def models(self) -> Path:
        return self.root / "models"
    @property
    def logs(self) -> Path:
        return self.root / "logs"
    @property
    def results(self) -> Path:
        return self.root / "results"
    @property
    def yolo_model(self) -> Path:
        return self.models / "yolov8n.hef"
    @property
    def label_file(self) -> Path:
        return self.models / "coco.txt"

# =========================================================
# RGB CAMERA CONFIGURATION
# =========================================================

@dataclass(slots=True)
class CameraConfig:
    width: int = 640
    height: int = 640
    fps: int = 30
    format: str = "RGB888"
    auto_focus: bool = True
    hflip: bool = False
    vflip: bool = False
    rotation: int = 0

# =========================================================
# THERMAL CAMERA CONFIGURATION
# =========================================================

@dataclass(slots=True)
class ThermalConfig:
    enable: bool = True
    sensor: str = "MLX90640"
    refresh_rate: int = 16
    i2c_frequency: int = 100000
    min_temperature: float = 20.0
    max_temperature: float = 45.0
    interpolation: int = 10

# =========================================================
# DETECTION CONFIGURATION
# =========================================================

@dataclass(slots=True)
class DetectionConfig:
    model_input_size: int = 640
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    person_class_id: int = 0
    max_detections: int = 100

# =========================================================
# TRACKER CONFIGURATION
# =========================================================

@dataclass(slots=True)
class TrackerConfig:
    enable: bool = False
    max_lost: int = 30
    iou_threshold: float = 0.3

# =========================================================
# DISPLAY CONFIGURATION
# =========================================================

@dataclass(slots=True)
class DisplayConfig:
    window_name: str = "SAR Human Detection"
    show_fps: bool = True
    show_confidence: bool = True
    show_id: bool = False
    show_temperature: bool = True
    fullscreen: bool = False

# =========================================================
# PERFORMANCE CONFIGURATION
# =========================================================

@dataclass(slots=True)
class PerformanceConfig:
    camera_queue_size: int = 3
    detection_queue_size: int = 3
    render_queue_size: int = 3
    worker_threads: int = 2
    stream_port: int = 5000

# =========================================================
# APPLICATION CONFIGURATION
# =========================================================

@dataclass(slots=True)
class AppConfig:
    path: PathConfig = field(default_factory=PathConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

# =========================================================
# GLOBAL CONFIG OBJECT
# =========================================================

config = AppConfig()

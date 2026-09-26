"""
=========================================================
Global configuration for Search and Rescue (SAR) Project
Platform : NVIDIA Jetson Orin Nano (4GB / 8GB)
Accelerator: NVIDIA Ampere GPU + Tensor Cores (TensorRT FP16)
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
    def tensorrt_model(self) -> Path:
        return self.models / "yolov8n.engine"
    @property
    def onnx_model(self) -> Path:
        return self.models / "yolov8n.onnx"
    @property
    def pt_model(self) -> Path:
        return self.models / "yolov8n.pt"
    @property
    def label_file(self) -> Path:
        return self.models / "coco.txt"

# =========================================================
# JETSON CAMERA CONFIGURATION (CSI via nvarguscamerasrc / USB via v4l2)
# =========================================================

@dataclass(slots=True)
class CameraConfig:
    backend: str = "nvargus"      # "nvargus" (Jetson CSI), "v4l2" (USB), or "demo"
    sensor_id: int = 0            # Jetson CSI camera sensor ID (0 or 1)
    device_id: int = 0            # V4L2 USB camera index (/dev/video0)
    width: int = 1280
    height: int = 720
    fps: int = 30                 # 30 or 60 fps
    format: str = "BGR"
    flip_method: int = 0          # 0: none, 2: 180 deg (useful on drone gimbal)

# =========================================================
# THERMAL CAMERA CONFIGURATION (MLX90640 via Jetson I2C-1)
# =========================================================

@dataclass(slots=True)
class ThermalConfig:
    enable: bool = True
    sensor: str = "MLX90640"
    i2c_bus: int = 1              # I2C-1 on Jetson 40-pin header (Pins 3 & 5)
    refresh_rate: int = 16        # 16 Hz
    i2c_frequency: int = 400000   # 400 kHz Fast-Mode on Jetson Tegra I2C
    min_temperature: float = 20.0
    max_temperature: float = 45.0
    interpolation: int = 10

# =========================================================
# DETECTION & TENSORRT CONFIGURATION
# =========================================================

@dataclass(slots=True)
class DetectionConfig:
    model_input_size: int = 640
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    person_class_id: int = 0
    max_detections: int = 100
    precision: str = "fp16"       # "fp16" for Tensor Cores on Jetson Orin Nano
    device: str = "cuda:0"

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
    window_name: str = "SAR Human Detection (Jetson Orin Nano)"
    show_fps: bool = True
    show_confidence: bool = True
    show_id: bool = False
    show_temperature: bool = True
    fullscreen: bool = False

# =========================================================
# JETSON PERFORMANCE CONFIGURATION
# =========================================================

@dataclass(slots=True)
class PerformanceConfig:
    camera_queue_size: int = 3
    detection_queue_size: int = 3
    render_queue_size: int = 3
    worker_threads: int = 4
    stream_port: int = 5000
    nvpmodel_mode: int = 0        # 0 = MAXN (15W power mode on Orin Nano)
    jetson_clocks: bool = True    # Lock GPU/EMC to max clock frequencies

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

config = AppConfig()

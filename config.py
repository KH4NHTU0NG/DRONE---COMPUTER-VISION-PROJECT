"""
=========================================================
Global configuration for Search and Rescue (SAR) Project
Platform   : macOS (Apple Silicon M-series GPU / Intel CPU)
Camera     : iPhone Connected via Apple Continuity Camera (AVFoundation / IP)
Accelerator: Apple Metal Performance Shaders (MPS) / CoreML / CPU
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
    def pt_model(self) -> Path:
        return self.models / "yolov8n.pt"

    @property
    def onnx_model(self) -> Path:
        return self.models / "yolov8n.onnx"

    @property
    def tensorrt_model(self) -> Path:
        return self.models / "yolov8n.engine"

    @property
    def label_file(self) -> Path:
        return self.models / "coco.txt"


# =========================================================
# CAMERA CONFIGURATION (iPhone Continuity Camera / Mac AVFoundation / IP Stream)
# =========================================================

@dataclass(slots=True)
class CameraConfig:
    backend: str = "iphone"       # "iphone" (Continuity Camera), "ip_stream", "avfoundation", or "demo"
    device_id: int = 1            # 1: iPhone Continuity Camera (hoặc -1 để tự động dò tìm)
    stream_url: str = ""          # URL luồng video nếu dùng App IP Camera (ví dụ: http://192.168.1.50:8080/video)
    width: int = 1280
    height: int = 720
    fps: int = 30
    format: str = "BGR"
    flip_method: int = 0          # 0: none, 1: mirror, 2: vflip, 3: 180-deg


# =========================================================
# THERMAL CAMERA CONFIGURATION (Tắt mặc định trên Mac hoặc chạy giả lập)
# =========================================================

@dataclass(slots=True)
class ThermalConfig:
    enable: bool = False          # False khi chạy local Mac không có chân I2C cứng
    mock: bool = False            # True nếu muốn tạo dữ liệu nhiệt giả lập để test luồng cứu hộ
    sensor: str = "MLX90640"
    refresh_rate: int = 16
    min_temperature: float = 20.0
    max_temperature: float = 45.0
    interpolation: int = 10


# =========================================================
# DETECTION CONFIGURATION (YOLOv8 + PyTorch MPS / Apple Silicon GPU)
# =========================================================

@dataclass(slots=True)
class DetectionConfig:
    model_input_size: int = 640
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    person_class_id: int = 0
    max_detections: int = 100
    precision: str = "fp32"       # "fp32" hoặc "fp16" (FP32 tối ưu độ chính xác trên MPS)
    device: str = "mps"           # "mps" cho Apple Silicon GPU, "cpu" cho Intel Mac


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
    window_name: str = "SAR Human Detection (macOS - iPhone Continuity Camera)"
    show_fps: bool = True
    show_confidence: bool = True
    show_id: bool = False
    show_temperature: bool = False
    fullscreen: bool = False


# =========================================================
# PERFORMANCE & THREADING CONFIGURATION
# =========================================================

@dataclass(slots=True)
class PerformanceConfig:
    camera_queue_size: int = 3
    detection_queue_size: int = 3
    render_queue_size: int = 3
    worker_threads: int = 4
    stream_port: int = 5001


# =========================================================
# ROOT APPLICATION CONFIGURATION
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

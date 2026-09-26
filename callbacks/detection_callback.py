"""
=========================================================
Detection Callback
YOLOv8 Real-Time Human Detector for macOS & Local Inference
Accelerated with Apple Silicon MPS (Metal Performance Shaders) / CPU
=========================================================
"""

from typing import List, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    torch = None
    HAS_TORCH = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except (ImportError, ModuleNotFoundError):
    YOLO = None
    HAS_YOLO = False

from core.detection import Detection, BoundingBox
from core.logger import logger
from config import config


class DetectionCallback:
    def __init__(self):
        self._detections: List[Detection] = []
        self.model = None
        self.device = self._resolve_device()

        if HAS_YOLO:
            # Ưu tiên pt_model -> onnx_model -> yolov8n.pt
            model_path = config.path.pt_model
            if not model_path.exists():
                if config.path.onnx_model.exists():
                    model_path = config.path.onnx_model
                else:
                    model_path = "yolov8n.pt"

            try:
                logger.info(f"Loading YOLO detection model for macOS: {model_path} (Device: {self.device})")
                self.model = YOLO(str(model_path))
                logger.info(f"Model loaded successfully. Inference backend: {self.device}")
            except Exception as e:
                logger.warning(f"Could not load YOLO model ({e}). Running in fallback mode.")
                self.model = None

    def _resolve_device(self) -> str:
        req_device = getattr(config.detection, "device", "mps").lower()
        if req_device in ("mps", "metal"):
            if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            logger.info("Apple Silicon MPS not available. Falling back to CPU.")
            return "cpu"
        elif req_device.startswith("cuda"):
            if HAS_TORCH and torch.cuda.is_available():
                return req_device
            logger.info("CUDA not available on macOS. Falling back to MPS/CPU.")
            if HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return "cpu"

    def process(self, frame: np.ndarray, frame_width: int = 0, frame_height: int = 0) -> List[Detection]:
        self._detections.clear()
        if frame is None:
            return self._detections

        h, w = frame.shape[:2]
        if frame_width <= 0:
            frame_width = w
        if frame_height <= 0:
            frame_height = h

        if self.model is None:
            return self._detections

        try:
            predict_args = {
                "source": frame,
                "conf": config.detection.confidence_threshold,
                "iou": config.detection.nms_threshold,
                "classes": [config.detection.person_class_id],
                "device": self.device,
                "verbose": False
            }
            if config.detection.precision.lower() == "fp16" and self.device != "cpu":
                predict_args["half"] = True

            results = self.model.predict(**predict_args)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None

                    x1 = max(0.0, float(xyxy[0]))
                    y1 = max(0.0, float(xyxy[1]))
                    x2 = min(float(frame_width), float(xyxy[2]))
                    y2 = min(float(frame_height), float(xyxy[3]))

                    det = Detection(
                        class_id=cls_id,
                        label="person",
                        confidence=conf,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        track_id=track_id
                    )
                    self._detections.append(det)
        except Exception as e:
            logger.warning(f"Inference error on macOS ({self.device}): {e}")

        return self._detections

    def get_persons(self) -> List[Detection]:
        return [
            det
            for det in self._detections
            if det.label == "person"
        ]

    def clear(self):
        self._detections.clear()

    @property
    def detections(self) -> List[Detection]:
        return self._detections

    @property
    def person_count(self) -> int:
        return len(self.get_persons())

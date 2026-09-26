"""
=========================================================
Jetson Detection Callback
TensorRT / CUDA YOLOv8 Detector for NVIDIA Jetson Orin Nano
=========================================================
"""

from typing import List
import numpy as np

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

        if HAS_YOLO:
            model_path = config.path.tensorrt_model
            # Ưu tiên load TensorRT .engine, sau đó đến .onnx, cuối cùng là .pt
            if not model_path.exists():
                if config.path.onnx_model.exists():
                    model_path = config.path.onnx_model
                elif config.path.pt_model.exists():
                    model_path = config.path.pt_model
                else:
                    model_path = "yolov8n.pt"

            try:
                logger.info(f"Loading detection model for Jetson Orin Nano: {model_path}")
                self.model = YOLO(str(model_path))
                logger.info("Model loaded successfully with TensorRT/CUDA acceleration.")
            except Exception as e:
                logger.warning(f"Could not load YOLO model ({e}). Using mock/fallback mode.")
                self.model = None

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
            is_half = (config.detection.precision.lower() == "fp16")
            results = self.model.predict(
                source=frame,
                conf=config.detection.confidence_threshold,
                iou=config.detection.nms_threshold,
                classes=[config.detection.person_class_id],
                device=config.detection.device,
                half=is_half,
                verbose=False
            )

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
            logger.warning(f"Inference error on Jetson: {e}")

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

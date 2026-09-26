"""
=========================================================
Detection Callback
Convert Hailo Detection -> Detection Object
=========================================================
"""

from typing import List

try:
    import hailo
    from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class
except (ImportError, ModuleNotFoundError):
    hailo = None

    class app_callback_class:
        pass

from core.detection import Detection, BoundingBox
from config import config


class DetectionCallback(app_callback_class):

    def __init__(self):
        super().__init__()
        self._detections: List[Detection] = []

    def process(self, buffer, frame_width: int, frame_height: int) -> List[Detection]:
        self._detections.clear()
        roi = hailo.get_roi_from_buffer(buffer)
        hailo_detections = roi.get_objects_typed(
            hailo.HAILO_DETECTION
        )

        for det in hailo_detections:
            label = det.get_label()
            confidence = det.get_confidence()
            if confidence < config.detection.confidence_threshold:
                continue
            bbox = det.get_bbox()
            track_id = None
            unique_ids = det.get_objects_typed(
                hailo.HAILO_UNIQUE_ID
            )

            if len(unique_ids) == 1:
                track_id = unique_ids[0].get_id()
            x1 = max(0.0, float(bbox.xmin()) * frame_width)
            y1 = max(0.0, float(bbox.ymin()) * frame_height)
            x2 = min(float(frame_width), float(bbox.xmax()) * frame_width)
            y2 = min(float(frame_height), float(bbox.ymax()) * frame_height)

            detection = Detection(
                class_id=det.get_class_id(),
                label=label,
                confidence=confidence,
                bbox=BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                ),
                track_id=track_id
            )
            self._detections.append(detection)
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

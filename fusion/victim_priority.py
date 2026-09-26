"""
=========================================================
Victim Priority Scoring
Search And Rescue (SAR) Medical & Tactical Priority Engine
=========================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from core.detection import Detection


@dataclass(slots=True)
class PriorityWeights:
    temperature: float = 0.35
    confidence: float = 0.25
    distance: float = 0.25
    tracking: float = 0.15


class VictimPriority:
    def __init__(self, focal_length_px: float = 500.0, human_height_m: float = 1.7):
        self.weights = PriorityWeights()
        self.focal_length_px = focal_length_px
        self.human_height_m = human_height_m

    def estimate_distance(self, bbox_height: float) -> float | None:
        """Ước lượng khoảng cách từ chiều cao bounding box (pinhole model)."""
        if bbox_height <= 1.0:
            return None
        distance = (self.focal_length_px * self.human_height_m) / bbox_height
        return round(float(min(max(distance, 1.0), 100.0)), 1)

    def calculate(
        self,
        detection: Detection
    ) -> float:
        # Tự động ước lượng khoảng cách nếu chưa có cảm biến khoảng cách / telemetry
        if detection.distance is None and detection.bbox is not None:
            detection.distance = self.estimate_distance(detection.bbox.height)

        temperature = self._temperature_score(
            detection.temperature
        )
        confidence = (detection.confidence or 0.0) * 100
        distance = self._distance_score(
            detection.distance
        )
        tracking = self._tracking_score(
            detection.track_id
        )
        score = (
            self.weights.temperature * temperature +
            self.weights.confidence * confidence +
            self.weights.distance * distance +
            self.weights.tracking * tracking
        )
        detection.priority = round(float(score), 2)
        return detection.priority

    def calculate_all(
        self,
        detections: list[Detection]
    ) -> list[Detection]:
        persons = []
        for det in detections:
            if det.label == "person":
                self.calculate(det)
                persons.append(det)
        persons.sort(
            key=lambda x: x.priority if x.priority is not None else 0.0,
            reverse=True
        )
        return persons

    @staticmethod
    def _temperature_score(
        temperature: float | None
    ) -> float:
        """
        Đánh giá thân nhiệt chuẩn y tế SAR:
        - 32°C - 38°C: Dải thân nhiệt da người bình thường -> Điểm cao (95-100)
        - 38°C - 41°C: Sốt cao / kiệt sức do nhiệt -> Điểm cao (90)
        - 28°C - 32°C: Nguy cơ hạ thân nhiệt (Hypothermia) khẩn cấp trong SAR -> Điểm cao (90)
        - 22°C - 28°C: Nhiễm lạnh nặng -> Điểm 75
        - > 42°C: Nhiệt độ quá cao (động cơ, ngọn lửa, đá hấp thụ nhiệt mặt trời) -> Điểm 40
        - < 22°C: Gần nhiệt độ môi trường lạnh -> Điểm 30
        - None: Chưa có dữ liệu nhiệt -> Điểm trung tính 50
        """
        if temperature is None:
            return 50.0
        if 32.0 <= temperature <= 38.0:
            return 100.0
        if 38.0 < temperature <= 41.0:
            return 90.0
        if 28.0 <= temperature < 32.0:
            return 90.0
        if 22.0 <= temperature < 28.0:
            return 75.0
        if temperature > 42.0:
            return 40.0
        return 30.0

    @staticmethod
    def _distance_score(
        distance: float | None
    ) -> float:
        if distance is None:
            return 50.0
        if distance < 3.0:
            return 100.0
        if distance < 6.0:
            return 85.0
        if distance < 12.0:
            return 70.0
        if distance < 25.0:
            return 50.0
        return 30.0

    @staticmethod
    def _tracking_score(
        track_id: int | None
    ) -> float:
        if track_id is None:
            return 50.0
        return 100.0

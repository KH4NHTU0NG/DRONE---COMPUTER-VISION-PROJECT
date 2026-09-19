"""
=========================================================
Victim Priority Scoring
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
    def __init__(self):
        self.weights = PriorityWeights()

    def calculate(
        self,
        detection: Detection
    ) -> float:
        temperature = self._temperature_score(
            detection.temperature
        )
        confidence = detection.confidence * 100
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
        detection.priority = round(score, 2)
        return detection.priority

    def calculate_all(
        self,
        detections: list[Detection]
    ) -> list[Detection]:
        for det in detections:
            if det.label == "person":
                self.calculate(det)
        detections.sort(
            key=lambda x: x.priority if x.priority else 0,
            reverse=True
        )
        return detections

    @staticmethod
    def _temperature_score(
        temperature: float | None
    ) -> float:
        if temperature is None:
            return 0
        if temperature >= 39:
            return 100
        if temperature >= 37:
            return 85
        if temperature >= 35:
            return 70
        if temperature >= 30:
            return 50
        return 20

    @staticmethod
    def _distance_score(
        distance: float | None
    ) -> float:
        if distance is None:
            return 50
        if distance < 2:
            return 100
        if distance < 5:
            return 85
        if distance < 10:
            return 70
        if distance < 20:
            return 40
        return 20

    @staticmethod
    def _tracking_score(
        track_id
    ) -> float:
        if track_id is None:
            return 50
        return 100

"""
=========================================================
Obstacle Estimation
=========================================================
"""

from __future__ import annotations

from core.detection import Detection

class ObstacleEstimator:
    def __init__(self):
        self.safe_threshold = 70.0

    def estimate(
        self,
        detection: Detection
    ) -> float:
        width = detection.bbox.width
        height = detection.bbox.height
        area = width * height
        # Bounding box lớn → khả năng ít bị che
        if area > 120000:
            score = 100
        elif area > 80000:
            score = 85
        elif area > 40000:
            score = 70
        elif area > 20000:
            score = 50
        else:
            score = 25
        detection.obstacle_score = score
        detection.safe = score >= self.safe_threshold
        return score

    def process(
        self,
        detections: list[Detection]
    ) -> list[Detection]:
        for det in detections:
            if det.label != "person":
                continue
            self.estimate(det)
        return detections

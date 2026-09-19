"""
=========================================================
Fusion Callback
RGB + Thermal + Priority Fusion
=========================================================
"""

from __future__ import annotations

from typing import List

from core.detection import Detection
from fusion.victim_priority import VictimPriority

class FusionCallback:
    def __init__(self):
        self.priority = VictimPriority()

    def process(
        self,
        detections: List[Detection]
    ) -> List[Detection]:
        if not detections:
            return []
        # Chỉ giữ lại người
        persons = [
            det
            for det in detections
            if det.label == "person"
        ]
        # Tính Priority Score
        persons = self.priority.calculate_all(
            persons
        )
        # Đánh số thứ tự ưu tiên
        for index, det in enumerate(persons):

            det.priority_rank = index + 1
        return persons

    def highest_priority(
        self,
        detections: List[Detection]
    ) -> Detection | None:
        if len(detections) == 0:
            return None
        detections = self.process(
            detections
        )
        return detections[0]

    def get_priority_list(
        self,
        detections: List[Detection]
    ):
        detections = self.process(
            detections
        )
        return [
            {
                "track_id": d.track_id,
                "temperature": d.temperature,
                "confidence": d.confidence,
                "priority": d.priority,
                "rank": d.priority_rank
            }
            for d in detections
        ]

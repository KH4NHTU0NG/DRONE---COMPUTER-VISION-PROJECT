"""
=========================================================
Fusion Callback
RGB + Thermal + Priority Fusion
=========================================================
"""

from __future__ import annotations

from typing import List, Dict, Any

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

        # Tính Priority Score và sắp xếp giảm dần cho person
        persons = self.priority.calculate_all(detections)

        # Đánh số thứ tự ưu tiên (1 = ưu tiên cao nhất)
        for index, det in enumerate(persons):
            det.priority_rank = index + 1

        return persons

    def highest_priority(
        self,
        detections: List[Detection]
    ) -> Detection | None:
        if not detections:
            return None
        ranked_persons = self.process(detections)
        if not ranked_persons:
            return None
        return ranked_persons[0]

    def get_priority_list(
        self,
        detections: List[Detection]
    ) -> List[Dict[str, Any]]:
        ranked_persons = self.process(detections)
        return [
            {
                "rank": d.priority_rank,
                "track_id": d.track_id,
                "label": d.label,
                "temperature": d.temperature,
                "distance": d.distance,
                "confidence": round(d.confidence, 2) if d.confidence else None,
                "priority": d.priority
            }
            for d in ranked_persons
        ]

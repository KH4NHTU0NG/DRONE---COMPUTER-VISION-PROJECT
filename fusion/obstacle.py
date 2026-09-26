"""
=========================================================
Obstacle & Occlusion Estimation
=========================================================
"""

from __future__ import annotations

from core.detection import Detection


class ObstacleEstimator:
    def __init__(self, safe_threshold: float = 70.0):
        self.safe_threshold = safe_threshold

    def estimate(
        self,
        detection: Detection,
        frame_width: int = 640,
        frame_height: int = 640
    ) -> float:
        if detection.bbox is None:
            detection.obstacle_score = 50.0
            detection.safe = False
            return 50.0

        width = detection.bbox.width
        height = detection.bbox.height
        total_frame_area = max(1.0, float(frame_width * frame_height))
        norm_area = (width * height) / total_frame_area

        # Tỷ lệ khung người chuẩn (Human aspect ratio h/w xấp xỉ 1.8 - 3.2)
        aspect_ratio = height / max(1.0, width)
        
        # Điểm hình học: nếu tỷ lệ chiều cao/rộng hợp lý -> ít bị che khuất ngang
        geometry_score = 100.0 if 1.8 <= aspect_ratio <= 3.5 else 60.0

        # Kết hợp diện tích tương đối và hình học
        if norm_area > 0.15:
            area_score = 100.0
        elif norm_area > 0.08:
            area_score = 85.0
        elif norm_area > 0.03:
            area_score = 70.0
        else:
            area_score = 50.0

        score = round(0.6 * area_score + 0.4 * geometry_score, 1)
        detection.obstacle_score = score
        detection.safe = score >= self.safe_threshold
        return score

    def process(
        self,
        detections: list[Detection],
        frame_width: int = 640,
        frame_height: int = 640
    ) -> list[Detection]:
        for det in detections:
            if det.label != "person":
                continue
            self.estimate(det, frame_width, frame_height)
        return detections

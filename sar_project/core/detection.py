from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2
        )

@dataclass(slots=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: BoundingBox
    track_id: int | None = None
    temperature: float | None = None
    distance: float | None = None
    priority: float | None = None
    priority_rank: int | None = None
    obstacle_score: float | None = None
    safe: bool | None = None

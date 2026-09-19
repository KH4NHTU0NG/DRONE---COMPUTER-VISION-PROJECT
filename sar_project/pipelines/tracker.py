"""
=========================================================
Object Tracker
=========================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from core.detection import Detection

class BaseTracker(ABC):
    @abstractmethod
    def update(self, detections: List[Detection]) -> List[Detection]:
        raise NotImplementedError

class DummyTracker(BaseTracker):
    def update(
        self,
        detections: List[Detection]
    ) -> List[Detection]:
        return detections

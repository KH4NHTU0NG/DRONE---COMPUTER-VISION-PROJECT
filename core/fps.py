"""
=========================================================
FPS Counter
=========================================================
"""

from __future__ import annotations

import time

class FPSCounter:
    def __init__(self):
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._frame_count = 0
        self._fps = 0.0

    def update(self) -> float:
        self._frame_count += 1
        current_time = time.perf_counter()
        elapsed = current_time - self._last_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_time = current_time
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    def reset(self):
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._frame_count = 0
        self._fps = 0.0

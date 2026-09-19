"""
=========================================================
MLX90640 Thermal Camera
Đọc trên luồng nền, không chặn vòng lặp detect chính.
=========================================================
"""

from __future__ import annotations

import threading
import time
import numpy as np
import board
import busio
import adafruit_mlx90640

from core.logger import logger


class ThermalCamera:
    def __init__(self):
        self._i2c = busio.I2C(
            board.SCL,
            board.SDA,
            frequency=800000
        )
        self._camera = adafruit_mlx90640.MLX90640(
            self._i2c
        )
        self._camera.refresh_rate = (
            adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        )
        self._raw_frame = np.zeros(
            (24 * 32,),
            dtype=np.float32
        )
        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _read_loop(self):
        while self._running:
            try:
                self._camera.getFrame(self._raw_frame)
                image = np.reshape(self._raw_frame, (24, 32)).copy()
                with self._lock:
                    self._latest_frame = image
            except Exception as exc:
                logger.warning(f"Thermal read error: {exc}")
                time.sleep(0.5)

    def read(self):
        if not self._running:
            return None
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_temperature_range(self):
        frame = self.read()
        if frame is None:
            return None
        return (float(np.min(frame)), float(np.max(frame)))

    def get_max_temperature(self):
        frame = self.read()
        if frame is None:
            return None
        return float(np.max(frame))

    def get_average_temperature(self):
        frame = self.read()
        if frame is None:
            return None
        return float(np.mean(frame))

    def is_running(self):
        return self._running

    def close(self):
        self.stop()

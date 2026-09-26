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
try:
    import board
    import busio
    import adafruit_mlx90640
    HAS_THERMAL_HARDWARE = True
except (ImportError, NotImplementedError):
    HAS_THERMAL_HARDWARE = False
    board = None
    busio = None
    adafruit_mlx90640 = None

from config import config
from core.logger import logger

_REFRESH_RATE_MAP = {}
if HAS_THERMAL_HARDWARE:
    _REFRESH_RATE_MAP = {
        1: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ,
        2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
        4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
        8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
        16: adafruit_mlx90640.RefreshRate.REFRESH_16_HZ,
        32: adafruit_mlx90640.RefreshRate.REFRESH_32_HZ,
    }


class ThermalCamera:
    def __init__(self):
        if not HAS_THERMAL_HARDWARE:
            raise RuntimeError("MLX90640 / board hardware libraries are not supported on this platform.")
        freq = getattr(config.thermal, "i2c_frequency", 100000)
        self._i2c = busio.I2C(
            board.SCL,
            board.SDA,
            frequency=freq
        )
        self._camera = adafruit_mlx90640.MLX90640(
            self._i2c
        )
        rate = _REFRESH_RATE_MAP.get(
            config.thermal.refresh_rate,
            adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
        )
        self._camera.refresh_rate = rate
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

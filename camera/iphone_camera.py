"""
=========================================================
iPhone Camera Capture Module for macOS
Supports:
1. Native Apple Continuity Camera (AVFoundation / USB / Wi-Fi)
2. IP WebCam / RTSP / HTTP Video Stream (Camo, DroidCam, etc.)
3. Built-in Mac Webcam fallback
4. Dedicated Background Capture Thread (Zero-Lag Buffer)
=========================================================
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from core.logger import logger
from config import config


def find_iphone_camera_index(max_indices: int = 4) -> int:
    """
    Tự động dò tìm camera iPhone kết nối qua macOS Continuity Camera.
    1. Kiểm tra system_profiler để xác định có iPhone kết nối hay không.
    2. Thử mở các camera index theo thứ tự ưu tiên (1, 0, 2, 3) để tìm nguồn camera hợp lệ.
    """
    # Thử quét system_profiler để kiểm tra tên iPhone
    has_iphone = False
    try:
        sp_out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            stderr=subprocess.DEVNULL,
            timeout=3.0
        ).decode("utf-8", errors="ignore")
        if "iPhone" in sp_out:
            has_iphone = True
            logger.info("Apple Continuity Camera (iPhone) detected in macOS system profiler.")
    except Exception:
        pass

    # Nếu có iPhone, thông thường index 1 là Continuity Camera trên MacBook
    candidate_indices = [1, 0] if has_iphone else [0, 1]
    for idx in range(2, max_indices):
        candidate_indices.append(idx)

    for idx in candidate_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                logger.info(f"Detected working camera at index {idx} ({w}x{h}).")
                return idx

    logger.warning("No working hardware camera found. Defaulting to index 0.")
    return 0


class IPhoneCamera:
    """
    Quản lý luồng thu nhận hình ảnh từ iPhone trên macOS.
    Sử dụng luồng nền đọc frame liên tục với hàng đợi 1 khung hình (Zero-lag),
    đảm bảo luồng suy luận AI luôn nhận khung hình thời gian thực mới nhất.
    """
    def __init__(
        self,
        device_id: Optional[int] = None,
        stream_url: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_method: int = 0
    ):
        self.stream_url = stream_url or getattr(config.camera, "stream_url", "")
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method

        if self.stream_url:
            self.source = self.stream_url
            self.is_stream_url = True
        else:
            configured_id = device_id if device_id is not None else getattr(config.camera, "device_id", -1)
            if configured_id is None or configured_id < 0:
                self.source = find_iphone_camera_index()
            else:
                self.source = configured_id
            self.is_stream_url = False

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._reconnect_attempts = 0

    def start(self) -> bool:
        if self._running:
            return True

        if not self._open_capture():
            logger.warning(f"Failed initial connection to camera source {self.source}. Will retry in background.")

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="IPhoneCameraCaptureThread",
            daemon=True
        )
        self._thread.start()
        return True

    def _open_capture(self) -> bool:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        logger.info(f"Opening camera source: {self.source} (StreamURL: {self.is_stream_url})")

        if self.is_stream_url:
            cap = cv2.VideoCapture(self.source)
        else:
            # Trên macOS, ưu tiên backend AVFoundation
            cap = cv2.VideoCapture(int(self.source), cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                cap = cv2.VideoCapture(int(self.source))

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            # Tối ưu kích thước đệm để giảm độ trễ
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                self._cap = cap
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"Connected to camera successfully: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
                return True

        logger.error(f"Could not open camera source {self.source}.")
        return False

    def _capture_loop(self):
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(1.0)
                logger.info(f"Attempting to reconnect camera source {self.source}...")
                self._open_capture()
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Camera read failed or iPhone disconnected. Reconnecting...")
                time.sleep(0.5)
                self._open_capture()
                continue

            # Xử lý lật/xoay hình ảnh nếu cần
            if self.flip_method == 1:
                frame = cv2.flip(frame, 1)  # Lật ngang (Mirror)
            elif self.flip_method == 2:
                frame = cv2.flip(frame, 0)  # Lật dọc
            elif self.flip_method == 3:
                frame = cv2.flip(frame, -1) # Lật cả 2 trục (180 độ)

            with self._lock:
                self._latest_frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._running:
            return False, None

        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        logger.info("IPhoneCamera stopped.")

    def close(self):
        self.stop()

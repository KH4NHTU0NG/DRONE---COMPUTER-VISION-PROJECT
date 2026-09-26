"""
Unit tests for iPhone Camera Module on macOS
"""

import time
import pytest
import numpy as np

from camera.iphone_camera import IPhoneCamera, find_iphone_camera_index


def test_find_iphone_camera_index():
    idx = find_iphone_camera_index()
    assert isinstance(idx, int)
    assert idx >= 0


def test_iphone_camera_initialization():
    cam = IPhoneCamera(device_id=1, width=640, height=480, fps=15)
    assert cam.width == 640
    assert cam.height == 480
    assert cam.fps == 15
    assert cam.source == 1
    assert not cam._running


def test_iphone_camera_start_and_read():
    idx = find_iphone_camera_index()
    cam = IPhoneCamera(device_id=idx, width=640, height=480, fps=15)
    started = cam.start()
    assert started is True
    assert cam._running is True

    # Cho luồng background đọc ít nhất 1 frame
    time.sleep(1.0)
    ret, frame = cam.read()
    if ret:
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3

    cam.stop()
    assert cam._running is False

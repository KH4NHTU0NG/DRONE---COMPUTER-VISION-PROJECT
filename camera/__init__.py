"""
=========================================================
Camera Package
Exports:
- IPhoneCamera: macOS camera manager (Continuity Camera & IP Webcam)
- ThermalCamera: MLX90640 thermal sensor manager
=========================================================
"""

from camera.iphone_camera import IPhoneCamera, find_iphone_camera_index
from camera.thermal_camera import ThermalCamera

__all__ = ["IPhoneCamera", "find_iphone_camera_index", "ThermalCamera"]

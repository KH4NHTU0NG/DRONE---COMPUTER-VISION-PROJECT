"""
=========================================================
Display Manager
Tự động chạy chế độ "headless" (không mở cửa sổ) nếu không
có màn hình. Video vẫn xem qua MJPEG stream.
=========================================================
"""

from __future__ import annotations

import os
import sys

import cv2

from core.logger import logger


class DisplayManager:
    def __init__(
        self,
        window_name: str = "SAR",
        width: int = 1280,
        height: int = 720
    ):
        self.window_name = window_name
        self._target_width = width
        self._target_height = height
        self._resized = False
        self._fullscreen = False

        if sys.platform == "darwin":
            self.headless = os.environ.get("HEADLESS", "0").lower() in ("1", "true", "yes")
            if threading.current_thread() is not threading.main_thread():
                self.headless = True
        else:
            self.headless = not os.environ.get("DISPLAY")

        if self.headless:
            logger.info(
                "Chạy chế độ headless (không mở cửa sổ GUI trực tiếp). "
                "Xem video qua MJPEG stream (http://localhost:5001)."
            )
        else:
            try:
                cv2.namedWindow(
                    self.window_name,
                    cv2.WINDOW_NORMAL
                )
            except Exception:
                self.headless = True

    def show(self, frame):
        if self.headless or frame is None:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            cv2.imshow(
                self.window_name,
                frame
            )
            if not self._resized:
                cv2.resizeWindow(
                    self.window_name,
                    self._target_width,
                    self._target_height
                )
                self._resized = True
        except Exception as e:
            logger.warning(f"Display error: {e}")

    def wait_key(self, delay: int = 1):
        if self.headless:
            return 0xFF
        if threading.current_thread() is not threading.main_thread():
            return 0xFF
        try:
            return cv2.waitKey(delay) & 0xFF
        except Exception:
            return 0xFF

    def should_close(self):
        if self.headless:
            return False
        key = self.wait_key()
        if key == ord("q"):
            return True
        if key == 27:
            return True
        return False

    def toggle_fullscreen(self):
        if self.headless:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            self._fullscreen = not self._fullscreen
            if self._fullscreen:
                cv2.setWindowProperty(
                    self.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )
            else:
                cv2.setWindowProperty(
                    self.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )
        except Exception:
            pass

    def resize(
        self,
        width: int,
        height: int
    ):
        if self.headless:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            cv2.resizeWindow(
                self.window_name,
                width,
                height
            )
        except Exception:
            pass

    def destroy(self):
        if self.headless:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            cv2.destroyWindow(
                self.window_name
            )
        except Exception:
            pass

    def close(self):
        self.destroy()

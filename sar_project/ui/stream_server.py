"""
=========================================================
MJPEG Stream Server
Xem video trực tiếp qua trình duyệt: http://<IP-Pi>:5000
=========================================================
"""

from __future__ import annotations

import threading
import time

import cv2
from flask import Flask, Response

from core.logger import logger

STREAM_FPS = 15
STREAM_INTERVAL = 1.0 / STREAM_FPS

class _Channel:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

    def update(self, frame):
        with self.lock:
            self.frame = frame
        self.new_frame_event.set()

    def generate(self):
        last_sent = 0.0
        while True:
            self.new_frame_event.wait(timeout=1.0)
            now = time.monotonic()
            if now - last_sent < STREAM_INTERVAL:
                time.sleep(STREAM_INTERVAL - (now - last_sent))
            with self.lock:
                if self.frame is None:
                    continue
                frame = self.frame
                self.new_frame_event.clear()
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            last_sent = time.monotonic()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
class StreamServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self._channels: dict[str, _Channel] = {}
        self._app = Flask(__name__)
        self._register_routes()
        self._thread = None

    def _get_channel(self, name: str) -> _Channel:
        if name not in self._channels:
            self._channels[name] = _Channel()
        return self._channels[name]

    def _register_routes(self):
        @self._app.route("/")
        def index():
            names = list(self._channels.keys()) or ["rgb"]
            boxes = "".join(
                f"<div style='flex:1;min-width:280px;'>"
                f"<h3 style='color:#fff;font-family:sans-serif;margin:8px;'>{n}</h3>"
                f"<img src='/stream/{n}' style='width:100%;height:auto;display:block;'>"
                f"</div>"
                for n in names
            )
            return (
                "<html><body style='margin:0;background:#000;'>"
                f"<div style='display:flex;flex-wrap:wrap;'>{boxes}</div>"
                "</body></html>"
            )

        @self._app.route("/stream/<name>")
        def stream(name):
            channel = self._get_channel(name)
            return Response(
                channel.generate(),
                mimetype="multipart/x-mixed-replace; boundary=frame"
            )

    def update(self, frame, name: str = "rgb"):
        self._get_channel(name).update(frame)

    def start(self):
        self._thread = threading.Thread(
            target=self._app.run,
            kwargs={
                "host": self.host,
                "port": self.port,
                "debug": False,
                "use_reloader": False,
                "threaded": True
            },
            daemon=True
        )
        self._thread.start()
        logger.info(f"Stream server started: http://<IP-Pi>:{self.port}")

"""
=========================================================
MJPEG Stream Server
Xem video trực tiếp qua trình duyệt: http://<IP-Pi>:5000
Hỗ trợ đa kết nối client an toàn luồng (Multi-client Thread Safe).
=========================================================
"""

from __future__ import annotations

import logging
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
        self.frame_id = 0
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def update(self, frame):
        with self.condition:
            self.frame = frame
            self.frame_id += 1
            self.condition.notify_all()

    def generate(self):
        last_frame_id = -1
        last_sent = 0.0
        while True:
            with self.condition:
                while self.frame_id == last_frame_id or self.frame is None:
                    if not self.condition.wait(timeout=1.0):
                        break
                frame = self.frame
                last_frame_id = self.frame_id

            if frame is None:
                continue

            now = time.monotonic()
            if now - last_sent < STREAM_INTERVAL:
                time.sleep(STREAM_INTERVAL - (now - last_sent))

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue

            last_sent = time.monotonic()
            try:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            except (GeneratorExit, BrokenPipeError, ConnectionResetError):
                break


from werkzeug.serving import make_server


class StreamServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        self.host = host
        self.port = port
        self._channels: dict[str, _Channel] = {}
        self._app = Flask(__name__)
        # Giảm log ồn ào của Werkzeug HTTP server
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        self._register_routes()
        self._server = None
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
                f"<div style='flex:1;min-width:320px;margin:8px;background:#111;padding:8px;border-radius:6px;'>"
                f"<h3 style='color:#00ffcc;font-family:sans-serif;margin:4px 0 8px 0;text-transform:uppercase;'>Stream: {n}</h3>"
                f"<img src='/stream/{n}' style='width:100%;height:auto;display:block;border-radius:4px;'>"
                f"</div>"
                for n in names
            )
            return (
                "<!DOCTYPE html><html><head><title>SAR Live Monitoring</title></head>"
                "<body style='margin:0;background:#050505;font-family:sans-serif;color:#eee;padding:16px;'>"
                "<h1 style='margin-bottom:12px;'>Search & Rescue Live Monitoring</h1>"
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
        import socket
        # Kiểm tra trước xem port có khả dụng không để tránh Werkzeug gọi sys.exit(1)
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            test_sock.bind((self.host, self.port))
            test_sock.close()
        except OSError as e:
            logger.warning(f"Port {self.port} is already in use ({e}). StreamServer will be disabled.")
            return

        try:
            self._server = make_server(self.host, self.port, self._app, threaded=True)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            logger.info(f"Stream server started: http://<IP-Pi>:{self.port}")
        except (Exception, SystemExit) as e:
            self._server = None
            logger.warning(f"Could not start stream server on {self.host}:{self.port} ({e})")

    def stop(self):
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:
                pass
            self._server = None

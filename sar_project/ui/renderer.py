"""
=========================================================
Renderer
Draw all information on the RGB frame.
=========================================================
"""
import cv2
from typing import List
from config import config
from core.detection import Detection


def _put_label(frame, text, x, y, color, font_scale=1.5, thickness=2):
    """Vẽ chữ có nền đen phía sau + chống răng cưa, đỡ bị nhoè/lẫn nền."""
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA
    )


class Renderer:
    def draw(
        self,
        frame,
        detections: List[Detection],
        fps: float = 0.0
    ):
        for det in detections:
            x1 = int(det.bbox.x1)
            y1 = int(det.bbox.y1)
            x2 = int(det.bbox.x2)
            y2 = int(det.bbox.y2)
            color = (0, 255, 0)
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color, 2,
                lineType=cv2.LINE_AA
            )
            label = det.label.capitalize()
            if config.display.show_confidence:
                label += f" {det.confidence*100:.0f}%"
            if config.display.show_id and det.track_id is not None:
                label += f" ID:{det.track_id}"
            _put_label(frame, label, x1, y1 - 10, color)

            if det.temperature is not None:
                _put_label(
                    frame,
                    f"{det.temperature:.1f} C",
                    x1, y2 + 25,
                    (0, 0, 255)
                )
        if config.display.show_fps:
            _put_label(frame, f"FPS: {fps:.2f}", 20, 40, (255, 0, 0), font_scale=1.0)
        return frame

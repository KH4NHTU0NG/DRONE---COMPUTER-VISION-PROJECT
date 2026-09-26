"""
=========================================================
Renderer
Draw annotations, bounding boxes, thermal & priority overlays.
Expects and operates on BGR format.
=========================================================
"""

from __future__ import annotations

import cv2
from typing import List
from config import config
from core.detection import Detection


def _put_label(frame, text: str, x: int, y: int, color=(0, 255, 0), font_scale: float = 0.6, thickness: int = 1):
    """Vẽ nhãn kèm hộp nền mờ đen phía sau để chống lóa và dễ đọc trên mọi địa hình."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Đảm bảo nhãn không tràn ra ngoài viền trên
    y_text = max(y, text_h + 6)
    
    # Hộp nền đen
    bg_x1 = max(0, x)
    bg_y1 = max(0, y_text - text_h - 4)
    bg_x2 = min(frame.shape[1], x + text_w + 6)
    bg_y2 = min(frame.shape[0], y_text + baseline + 2)
    
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        frame,
        text,
        (bg_x1 + 3, y_text),
        font,
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
        if frame is None:
            return None

        h, w = frame.shape[:2]

        for det in detections:
            x1 = max(0, min(int(det.bbox.x1), w - 1))
            y1 = max(0, min(int(det.bbox.y1), h - 1))
            x2 = max(0, min(int(det.bbox.x2), w - 1))
            y2 = max(0, min(int(det.bbox.y2), h - 1))

            # Chọn màu theo rank cứu hộ (Rank 1 = Đỏ cảnh báo, các rank sau = Vàng/Xanh)
            if det.priority_rank == 1:
                bbox_color = (0, 0, 255)      # Đỏ (BGR)
            elif det.priority_rank and det.priority_rank <= 3:
                bbox_color = (0, 165, 255)    # Cam (BGR)
            else:
                bbox_color = (0, 255, 0)      # Xanh lá (BGR)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                bbox_color, 2,
                lineType=cv2.LINE_AA
            )

            # Nhãn chính: Tên + Độ tin cậy + ID
            label_parts = [det.label.capitalize()]
            if config.display.show_confidence and det.confidence is not None:
                label_parts.append(f"{det.confidence * 100:.0f}%")
            if config.display.show_id and det.track_id is not None:
                label_parts.append(f"ID:{det.track_id}")
            if det.priority_rank is not None:
                label_parts.append(f"Rank#{det.priority_rank}")

            main_label = " ".join(label_parts)
            _put_label(frame, main_label, x1, y1 - 8, color=bbox_color, font_scale=0.6, thickness=1)

            # Thông tin phụ trợ phía dưới: Nhiệt độ + Khoảng cách + Priority Score
            sub_parts = []
            if config.display.show_temperature and det.temperature is not None:
                sub_parts.append(f"T:{det.temperature:.1f}C")
            if det.distance is not None:
                sub_parts.append(f"D:{det.distance:.1f}m")
            if det.priority is not None:
                sub_parts.append(f"P:{det.priority:.1f}")

            if sub_parts:
                sub_label = " | ".join(sub_parts)
                _put_label(frame, sub_label, x1, y2 + 18, color=(255, 255, 255), font_scale=0.5, thickness=1)

        # Hiển thị FPS và số lượng nạn nhân
        if config.display.show_fps:
            status_text = f"FPS: {fps:.1f} | Victims: {len(detections)}"
            _put_label(frame, status_text, 16, 32, color=(0, 255, 255), font_scale=0.7, thickness=2)

        return frame

"""
=========================================================
Unit Tests for SAR System
Validating priority scoring, thermal mapping, fusion, and streaming.
=========================================================
"""

import threading
import time
import numpy as np
import pytest

from core.detection import Detection, BoundingBox
from fusion.victim_priority import VictimPriority
from fusion.obstacle import ObstacleEstimator
from callbacks.fusion_callback import FusionCallback
from callbacks.thermal_callback import ThermalCallback
from pipelines.thermal_pipeline import ThermalPipeline
from ui.renderer import Renderer
from ui.stream_server import _Channel


def test_victim_priority_scoring_and_hypothermia():
    engine = VictimPriority()

    # 1. Người bình thường (36.5°C)
    det_normal = Detection(
        class_id=0,
        label="person",
        confidence=0.9,
        bbox=BoundingBox(x1=100, y1=100, x2=200, y2=400), # height = 300px
        temperature=36.5,
        track_id=1
    )
    score_normal = engine.calculate(det_normal)
    assert det_normal.distance is not None
    assert score_normal > 80.0

    # 2. Nạn nhân hạ thân nhiệt khẩn cấp (30.0°C) - SAR cần ưu tiên cao
    det_hypo = Detection(
        class_id=0,
        label="person",
        confidence=0.9,
        bbox=BoundingBox(x1=100, y1=100, x2=200, y2=400),
        temperature=30.0,
        track_id=2
    )
    score_hypo = engine.calculate(det_hypo)
    assert score_hypo >= 80.0, "Hạ thân nhiệt trong SAR phải được ưu tiên khẩn cấp"

    # 3. Vật thể quá nhiệt (55.0°C - động cơ xe hoặc tảng đá phơi nắng)
    det_hot = Detection(
        class_id=0,
        label="person",
        confidence=0.9,
        bbox=BoundingBox(x1=100, y1=100, x2=200, y2=400),
        temperature=55.0,
        track_id=3
    )
    score_hot = engine.calculate(det_hot)
    assert score_hot < score_normal, "Vật thể quá nóng (>42°C) không được ưu tiên hơn thân nhiệt người"


def test_fusion_callback_empty_and_non_person_safety():
    fusion = FusionCallback()

    # Edge case 1: Dữ liệu rỗng
    assert fusion.highest_priority([]) is None
    assert fusion.process([]) == []

    # Edge case 2: Không có class person (chỉ có chair, backpack)
    non_persons = [
        Detection(class_id=1, label="chair", confidence=0.8, bbox=BoundingBox(10, 10, 50, 50)),
        Detection(class_id=2, label="backpack", confidence=0.75, bbox=BoundingBox(60, 60, 90, 90))
    ]
    # Trước khi fix: dòng này gây IndexError: list index out of range!
    top = fusion.highest_priority(non_persons)
    assert top is None

    # Normal case: Có 2 người
    mixed = non_persons + [
        Detection(class_id=0, label="person", confidence=0.85, bbox=BoundingBox(100, 100, 200, 350), temperature=35.5),
        Detection(class_id=0, label="person", confidence=0.95, bbox=BoundingBox(50, 50, 200, 450), temperature=36.8)
    ]
    ranked = fusion.process(mixed)
    assert len(ranked) == 2
    assert ranked[0].priority_rank == 1
    assert ranked[1].priority_rank == 2
    assert ranked[0].priority >= ranked[1].priority


def test_thermal_callback_small_bbox():
    callback = ThermalCallback()
    
    # Giả lập frame nhiệt 24x32
    thermal_data = np.full((24, 32), 25.0, dtype=np.float32)
    thermal_data[5, 5] = 37.2 # Điểm nhiệt người
    
    # Đối tượng cực nhỏ ở xa trên frame RGB 640x640 (rộng 4px, cao 8px)
    # Tọa độ tương ứng: x: 100 -> 100*(32/640) = 5.0
    det_small = Detection(
        class_id=0,
        label="person",
        confidence=0.8,
        bbox=BoundingBox(x1=100.0, y1=133.0, x2=104.0, y2=141.0)
    )
    
    callback.process([det_small], thermal_data, rgb_width=640, rgb_height=640)
    assert det_small.temperature is not None
    assert det_small.temperature >= 25.0


def test_thermal_pipeline_normalization_clamping():
    pipeline = ThermalPipeline()
    # Giả lập frame chênh lệch nhiệt cực nhỏ (nhiễu 23.1 - 23.4 °C)
    raw = np.linspace(23.1, 23.4, 24 * 32, dtype=np.float32).reshape((24, 32))
    pipeline.frame = raw
    
    norm = pipeline.normalize()
    assert norm is not None
    assert norm.dtype == np.uint8
    # Với khoảng kẹp 20.0 - 45.0, giá trị phải nằm quanh mức ~30-40, không bị kéo dãn 0-255 bùng nổ nhiễu
    assert np.max(norm) < 100
    assert np.min(norm) > 10


def test_obstacle_estimator_normalized():
    estimator = ObstacleEstimator(safe_threshold=70.0)
    det = Detection(
        class_id=0,
        label="person",
        confidence=0.9,
        bbox=BoundingBox(x1=200, y1=100, x2=320, y2=450) # width=120, height=350, ratio=2.9
    )
    score = estimator.estimate(det, frame_width=640, frame_height=640)
    assert score >= 70.0
    assert det.safe is True


def test_stream_server_multi_client():
    channel = _Channel()
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

    gen1 = channel.generate()
    gen2 = channel.generate()

    channel.update(frame1)
    chunk1 = next(gen1)
    chunk2 = next(gen2)

    assert b"--frame\r\n" in chunk1
    assert b"--frame\r\n" in chunk2

    channel.update(frame2)
    chunk1_next = next(gen1)
    chunk2_next = next(gen2)

    assert b"--frame\r\n" in chunk1_next
    assert b"--frame\r\n" in chunk2_next


def test_renderer_bgr():
    renderer = Renderer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = Detection(
        class_id=0,
        label="person",
        confidence=0.88,
        bbox=BoundingBox(50, 50, 150, 300),
        temperature=36.6,
        distance=4.5,
        priority=89.2,
        priority_rank=1
    )
    rendered = renderer.draw(frame, [det], fps=28.5)
    assert rendered is not None
    assert rendered.shape == (480, 640, 3)


def test_fps_counter():
    from core.fps import FPSCounter
    counter = FPSCounter()
    assert counter.fps == 0.0
    counter.update()
    # Sau reset
    counter.reset()
    assert counter.fps == 0.0


def test_detection_callback_utilities():
    from callbacks.detection_callback import DetectionCallback
    cb = DetectionCallback()
    cb._detections = [
        Detection(class_id=0, label="person", confidence=0.9, bbox=BoundingBox(10, 10, 50, 100)),
        Detection(class_id=2, label="car", confidence=0.8, bbox=BoundingBox(100, 100, 200, 200)),
        Detection(class_id=0, label="person", confidence=0.95, bbox=BoundingBox(60, 60, 120, 180)),
    ]
    assert cb.person_count == 2
    persons = cb.get_persons()
    assert len(persons) == 2
    assert all(p.label == "person" for p in persons)
    cb.clear()
    assert cb.person_count == 0


def test_thermal_camera_fallback():
    from camera.thermal_camera import ThermalCamera, HAS_THERMAL_HARDWARE
    if not HAS_THERMAL_HARDWARE:
        with pytest.raises(RuntimeError):
            ThermalCamera()


def test_application_context_non_blocking_queue():
    from core.application_context import ApplicationContext
    ctx = ApplicationContext()
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_dets = [Detection(class_id=0, label="person", confidence=0.9, bbox=BoundingBox(0, 0, 5, 5))]
    
    # Đẩy 5 frame liên tục vào queue kích thước tối đa 3 -> không được gây deadlock hay blocking
    for i in range(5):
        ctx.push_inference_result(dummy_frame, dummy_dets, fps=30.0 + i)
    
    assert ctx.frame_count == 5
    assert ctx.latest_fps == 34.0
    task = ctx.get_render_task(timeout=0.01)
    assert task is not None
    frame, dets, fps, heatmap = task
    assert frame.shape == (10, 10, 3)
    assert len(dets) == 1
    ctx.reset()
    assert ctx.frame_count == 0


def test_detection_pipeline_initialization():
    from config import config
    from pipelines.detection_pipeline import DetectionPipeline
    config.performance.stream_port = 58291
    pipeline = DetectionPipeline()
    assert pipeline.context is not None
    assert pipeline.stream_server is not None
    pipeline.context.running = False
    pipeline.stream_server.stop()

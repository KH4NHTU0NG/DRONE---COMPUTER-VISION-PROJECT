# SAR Project — Search And Rescue System (macOS & iPhone Camera Edition)

Hệ thống phát hiện nạn nhân cứu hộ thời gian thực chạy trực tiếp (local) trên **máy Mac** (tối ưu hóa cho Apple Silicon M1/M2/M3/M4 qua **Metal Performance Shaders - MPS** hoặc Intel CPU), kết nối nguồn hình ảnh trực tiếp từ **Camera iPhone** (thông qua tính năng Apple Continuity Camera hoặc IP Webcam Stream).

---

## 1. Tính Năng Nổi Bật Trên Nhánh macOS + iPhone

- **Kết nối Camera iPhone liền mạch (Apple Continuity Camera):**
  - Tự động nhận diện iPhone được kết nối với máy Mac (cả qua Wi-Fi lẫn cáp USB Lightning/Type-C) thông qua backend `AVFoundation`.
  - Hỗ trợ độ phân giải linh hoạt (720p, 1080p, 4K) với luồng thu hình đa luồng (Multi-threaded Zero-Lag Ring Buffer) không gây giật lag.
  - Tự động fallback sang Webcam tích hợp của MacBook nếu iPhone tạm thời mất kết nối.
  - Hỗ trợ luồng RTSP / HTTP Video Stream từ các app như Camo, DroidCam hoặc IP Webcam.
- **Tăng tốc suy luận AI trên Apple Silicon GPU (MPS):**
  - Tận dụng lõi GPU Apple Metal thông qua PyTorch MPS (`device="mps"`), mang lại tốc độ suy luận mô hình YOLOv8 mượt mà (30 - 60+ FPS) ngay trên máy tính xách tay.
- **Xem trực tiếp trên giao diện Desktop & Web Browser:**
  - Hiển thị trực tiếp qua cửa sổ OpenCV Quartz/Cocoa trên màn hình Mac.
  - Tích hợp máy chủ phát luồng Web MJPEG độc lập tại cổng `5001` (`http://localhost:5001`), tránh xung đột với dịch vụ AirPlay Receiver của macOS.
- **Triage Scoring & Tracking:**
  - Phân tích bounding box, độ tin cậy và theo dõi nạn nhân thời gian thực.
  - Khung kiến trúc sẵn sàng kết hợp cảm biến nhiệt khi kết nối ngoại vi.

---

## 2. Hướng Dẫn Kết Nối Camera iPhone Với Máy Mac

### Cách 1: Sử dụng Apple Continuity Camera (Khuyến nghị - Nhanh nhất)
1. Đảm bảo iPhone và máy Mac:
   - Đăng nhập cùng một tài khoản Apple ID.
   - Bật cả Wi-Fi và Bluetooth trên cả 2 thiết bị (hoặc cắm dây cáp USB từ iPhone vào máy Mac để có độ trễ thấp nhất).
   - Trên iPhone: Vào **Cài đặt (Settings) > Cài đặt chung (General) > AirPlay & Handoff > Bật "Camera thông suốt" (Continuity Camera)**.
2. Đặt iPhone gần máy Mac (hoặc gắn lên giá đỡ màn hình).
3. Hệ thống sẽ tự động quét và kết nối với camera iPhone (thường là index `1`).

### Cách 2: Sử dụng Ứng dụng IP Webcam / RTSP
Nếu muốn dùng app truyền hình ảnh qua mạng nội bộ:
1. Mở app IP Camera trên iPhone (ví dụ: *Live-Reporter*, *DroidCam*, *IP Webcam*).
2. Lấy địa chỉ IP và cổng hiển thị trên app (ví dụ: `http://192.168.1.50:8080/video`).
3. Mở [config.py](file:///Users/trankhanhtuong/Desktop/drone/config.py) và cấu hình:
   ```python
   config.camera.stream_url = "http://192.168.1.50:8080/video"
   ```

---

## 3. Cài Đặt Môi Trường Trên Máy Mac

### Bước 1: Kích hoạt môi trường Python (Python 3.10+)
```bash
cd ~/Desktop/drone
# Nếu sử dụng venv:
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 2: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Khởi Chạy Ứng Dụng

Chạy ứng dụng chính:
```bash
python3 App.py
```

- **Cửa sổ hiển thị:** Cửa sổ đồ họa trực quan sẽ mở trên màn hình macOS, hiển thị khung hình từ iPhone kèm bounding box nhận diện người và chỉ số FPS thời gian thực.
- **Xem qua trình duyệt Web:** Mở trình duyệt bất kỳ (Safari, Chrome) và truy cập:
  ```text
  http://localhost:5001
  ```
- **Thoát ứng dụng:** Nhấn phím `q` hoặc phím `ESC` trên cửa sổ video, hoặc nhấn `Ctrl + C` trên Terminal.

---

## 5. Chạy Bộ Kiểm Thử Tự Động (Unit Tests)

Chạy toàn bộ 15 bài kiểm thử tự động (bao gồm pipeline, iPhone camera module, ring queue và thuật toán SAR):
```bash
PYTHONPATH=. pytest tests/ -v
```

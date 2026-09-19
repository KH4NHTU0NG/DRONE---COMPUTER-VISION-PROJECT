# SAR Project — Search And Rescue Detection System

Hệ thống phát hiện người (nạn nhân) thời gian thực trên drone/thiết bị di động, sử dụng Raspberry Pi 5 và Hailo AI HAT+. Kết hợp camera RGB (nhận diện người bằng AI) và camera nhiệt MLX90640 (đo nhiệt độ), tính điểm ưu tiên cứu hộ cho từng đối tượng phát hiện được.

## Tính năng chính

- Phát hiện người thời gian thực bằng model YOLOv8 chạy trên NPU Hailo (tăng tốc phần cứng)
- Đo nhiệt độ đối tượng qua cảm biến MLX90640, kết hợp (fusion) với kết quả detect
- Tính điểm ưu tiên cứu hộ dựa trên nhiệt độ, khoảng cách, độ che khuất
- Xem video trực tiếp qua trình duyệt (MJPEG stream), không cần màn hình cắm vào thiết bị
- Tự động chạy chế độ headless khi không có màn hình (phù hợp khi gắn lên drone)

## Phần cứng yêu cầu

| Thiết bị | Ghi chú |
|---|---|
| Raspberry Pi 5 | Board chính |
| Hailo AI HAT+ | Kiểm tra đúng kiến trúc chip (Hailo-8 hoặc Hailo-8L) bằng lệnh `hailortcli fw-control identify` trước khi cài đặt |
| Camera Module (CSI) | Camera RGB chính |
| Cảm biến nhiệt MLX90640 | Gắn qua I2C (SDA/SCL/VCC/GND) |
| Thẻ microSD (32GB+) | Raspberry Pi OS 64-bit bản **Trixie** trở lên |

## Cài đặt hệ thống

### 1. Flash Raspberry Pi OS (Trixie)

Dùng [Raspberry Pi Imager](https://www.raspberrypi.com/software/), chọn Raspberry Pi OS (64-bit) bản Trixie, bật sẵn SSH khi cấu hình.

> Hailo yêu cầu Debian Trixie — Bookworm không hỗ trợ các bản HailoRT/TAPPAS mới. Không nâng cấp trực tiếp từ Bookworm, hãy flash lại thẻ SD.

### 2. Cài driver + HailoRT + TAPPAS

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install dkms -y
sudo apt install hailo-all -y
sudo reboot
```

Kiểm tra:
```bash
hailortcli fw-control identify
```

### 3. Cài framework `hailo-apps`

```bash
git clone https://github.com/hailo-ai/hailo-apps.git
cd hailo-apps
sudo ./install.sh
```

Mỗi lần dùng, activate venv:
```bash
source ~/hailo-apps/setup_env.sh
```

### 4. Bật I2C (cho cảm biến nhiệt)

```bash
sudo raspi-config
```
→ `Interface Options` → `I2C` → `Enable`

### 5. Sửa cấu hình I2C baudrate (bắt buộc, né lỗi phần cứng clock-stretching)

```bash
sudo nano /boot/firmware/config.txt
```
Thêm dòng:
```
dtparam=i2c_arm=on,i2c_arm_baudrate=10000
```
Reboot lại: `sudo reboot`

### 6. Cài dependencies của project

```bash
cd ~/sar_project
pip install -r requirements.txt
```

## Cách chạy

```bash
source ~/hailo-apps/setup_env.sh
cd ~/sar_project
python3 App.py --input rpi
```

Tham số hữu ích:
- `--input rpi` — dùng camera Pi thật (mặc định nếu không có sẽ dùng video demo)
- `--frame-rate 60` — tăng tốc độ khung hình (tuỳ camera hỗ trợ)
- `--list-models` — xem danh sách model khả dụng cho đúng kiến trúc chip

## Xem video trực tiếp

Mở trình duyệt trên máy khác cùng mạng LAN:
```
http://<IP-Pi>:5000
```

Lấy IP của Pi bằng: `hostname -I`

Trang sẽ hiện 2 luồng: camera RGB (có khung nhận diện) và camera nhiệt (heatmap màu), nếu cảm biến nhiệt đang bật (`config.thermal.enable = True`).

## Cấu trúc thư mục

```
sar_project/
├── App.py                     # Điểm chạy chính
├── config.py                  # Cấu hình camera, display, thermal, model...
├── requirements.txt
├── test_thermal.py            # Test độc lập cảm biến nhiệt
├── callbacks/
│   ├── detection_callback.py  # Xử lý kết quả detect từ Hailo
│   ├── thermal_callback.py    # Gắn nhiệt độ vào detection
│   └── fusion_callback.py     # Tổng hợp, tính priority
├── camera/
│   └── thermal_camera.py      # Driver MLX90640
├── core/
│   ├── detection.py           # Định nghĩa Detection, BoundingBox
│   ├── fps.py
│   └── logger.py
├── fusion/
│   ├── victim_priority.py     # Tính điểm ưu tiên cứu hộ
│   └── obstacle.py            # Ước lượng vật cản (chưa nối vào pipeline)
├── models/
│   ├── yolov8n.hef
│   └── coco.txt
├── pipelines/
│   ├── detection_pipeline.py  # Pipeline chính (GStreamer + Hailo)
│   ├── thermal_pipeline.py
│   └── tracker.py
└── ui/
    ├── renderer.py             # Vẽ bounding box, nhãn lên khung hình
    ├── display_manager.py      # Cửa sổ hiển thị (tự tắt nếu headless)
    └── stream_server.py        # Server MJPEG stream qua trình duyệt
```

## Cấu hình thường chỉnh

Trong `config.py`:

| Cấu hình | Mặc định | Ghi chú |
|---|---|---|
| `camera.width/height` | 1280x720 | Giảm xuống 640x480 nếu cần FPS cao hơn |
| `camera.fps` | 30 | Tăng cần camera hỗ trợ |
| `thermal.enable` | `False` | Bật `True` khi đã gắn cảm biến nhiệt thật |
| `display.show_confidence` | — | Hiện % độ tin cậy trên nhãn |
| `display.show_id` | — | Hiện track ID trên nhãn |

## Xử lý sự cố thường gặp

| Vấn đề | Nguyên nhân / cách xử lý |
|---|---|
| `ModuleNotFoundError: hailo_apps` | Chưa activate venv — chạy `source ~/hailo-apps/setup_env.sh` |
| `HEF format is not compatible with device` | Sai kiến trúc chip — kiểm tra bằng `hailortcli fw-control identify`, không ép `--arch` sai |
| Camera nhiệt báo `Too many retries` | Lỗi phần cứng I2C clock-stretching của Raspberry Pi — đã xử lý bằng cách giảm `i2c_arm_baudrate=10000` trong `/boot/firmware/config.txt` |
| FPS thấp bất thường | Kiểm tra `thermal.enable` — nếu đọc cảm biến nhiệt đồng bộ (blocking) sẽ kéo chậm cả pipeline; code hiện đọc nhiệt trên luồng nền riêng |
| Không mở được cửa sổ video (`qt.qpa.xcb`) | Chạy qua SSH thuần không có màn hình — dùng VNC, hoặc bỏ qua, chỉ xem qua MJPEG stream (tự động headless) |
| Cổng 5000 đã được dùng | Có tiến trình `App.py` cũ chưa tắt hẳn — chạy `pkill -f "App.py"` |

## Hướng phát triển tiếp theo

- Nối `fusion/obstacle.py` (đã viết, chưa dùng) vào pipeline chính để tính độ an toàn/che khuất
- Thêm nguồn đo khoảng cách thật (độ cao bay + kích thước bbox, hoặc cảm biến lidar/tof)
- Tích hợp MAVLink để lấy toạ độ GPS + độ cao từ flight controller, gắn vào mỗi detection
- Lưu ảnh + toạ độ + priority score vào thư mục kết quả khi phát hiện người
- Thay `DummyTracker` bằng thuật toán tracking thật (SORT/DeepSORT) để giữ track_id ổn định
- Viết unit test cho `VictimPriority`, `ObstacleEstimator`

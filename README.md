# SAR Project — Search And Rescue System (NVIDIA Jetson Orin Nano Edition)

He thong phat hien nan nhan va cuu ho thoi gian thuc tren Drone/Thiet bi di dong, toi uu hoa phan cung chuyen biet cho **NVIDIA Jetson Orin Nano (4GB / 8GB)**. Tan dung kien truc **NVIDIA Ampere GPU (512 - 1024 CUDA Cores, 32 Tensor Cores)** voi bo tang toc **TensorRT FP16**, ket hop camera CSI toc do cao qua ISP phan cung va cam bien nhiet hong ngoai **MLX90640**.

---

## 1. Tinh Nang Chinh Tren Jetson Orin Nano

- **Tang toc AI TensorRT FP16:** Chay mo hinh YOLOv8 tren Tensor Cores cua Jetson Orin Nano voi thong luong cuc cao (60 - 100+ FPS).
- **Zero-Copy GStreamer Hardware Acceleration:** Thu nhan luong camera CSI truc tiep qua `nvarguscamerasrc` va bo nho phan cung `NVMM`, khong tieu ton tai nguyen CPU.
- **Cross-Modal Thermal Verification:** Tu dong kiem chung cheo dau hieu sinh ton nhiet nguoi (30°C - 39°C) qua cam bien MLX90640, nang cao do nhay phat hien nguoi tu xa hoac bi che khuat.
- **Tinh diem uu tien cuu nan (Triage Scoring):** Danh gia tinh trang ha than nhiet (Hypothermia), khoang cach quang hoc va do an toan vat can.
- **Kien truc da nhan bat dong bo (Multi-Core Ring Queue):** Tach biet luong AI va luong Render/Web Stream, duy tri video muot ma khong bao gio bi nghen mang.

---

## 2. Phan Cung Yeu Cau

| Thiet bi | Thong so ky thuat | Ghi chu |
|---|---|---|
| **Board chinh** | NVIDIA Jetson Orin Nano Developer Kit (4GB hoac 8GB) | Ampere GPU, 6-core ARM Cortex-A78AE |
| **Nguon dien** | 9V - 20V DC (khuyen nghi 19V / 45W hoac pin drone 5V/3A) | Bat buoc de chay che do MAXN 15W |
| **Camera RGB** | Camera CSI (IMX219 / IMX477) hoac USB Webcam | Ket noi cong CAM0/CAM1 qua cap ribbon |
| **Cam bien nhiet** | MLX90640 Thermal Sensor (32 x 24) | Giao tiep qua bus I2C-1 (Pin 3 & Pin 5) |
| **Luu tru** | M.2 NVMe SSD (128GB+) | JetPack 5.1.2 hoac JetPack 6.0 LTS |

---

## 3. So Do Dau Day Cam Bien Nhiet MLX90640 (40-Pin Header)

| Chan MLX90640 | Chan 40-Pin Jetson Orin Nano | Ten Tin Hieu |
|---|---|---|
| **VIN / VCC** | **Pin 1** hoac **Pin 17** | 3.3V Power |
| **GND** | **Pin 6**, **Pin 9** hoac **Pin 14** | Ground |
| **SDA** | **Pin 3** | I2C1_SDA (Bus 1) |
| **SCL** | **Pin 5** | I2C1_SCL (Bus 1) |

> Cam bien MLX90640 tren Jetson chay o tan so Fast-Mode **400 kHz** rat on dinh, khong bi loi clock-stretching.

---

## 4. Cai Dat Moi Truong Tren Jetson

### Buoc 1: Kich hoat che do cong suat toi da (MAXN 15W) & Khoa xung
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Buoc 2: Cai dat dependencies
```bash
cd ~/drone
pip3 install -r requirements.txt
```

### Buoc 3: Xuat mo hinh sang TensorRT Engine FP16
```bash
yolo export model=yolov8n.pt format=engine half=True device=0
mkdir -p models
mv yolov8n.engine models/
```

### Buoc 4: Kiem tra cam bien nhiet MLX90640
```bash
sudo i2cdetect -y -r 1
python3 test_thermal.py
```

---

## 5. Khoi Chay Ung Dung

### 1. Chay voi Camera CSI (Khuyen nghi cho Drone)
```bash
python3 App.py
```

### 2. Chay voi Webcam USB
Chinh `camera.backend = "v4l2"` trong `config.py`, roi chay:
```bash
python3 App.py
```

---

## 6. Xem Video & Du Lieu Cuu Ho Truc Tiep

Mo trinh duyet tren may tinh/dien thoai cung mang:
```text
http://<IP-Jetson>:5000
```

---

## 7. Chay Bo Kiem Thu Tu Dong (Unit Tests)

```bash
PYTHONPATH=. pytest tests/test_sar.py -v
```

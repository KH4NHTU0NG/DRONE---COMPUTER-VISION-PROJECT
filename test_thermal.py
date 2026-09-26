"""
=========================================================
Standalone MLX90640 Thermal Camera Test Script
Kiểm tra độc lập cảm biến nhiệt MLX90640 trên bus I2C
=========================================================
"""

import time
import sys
from config import config
from camera.thermal_camera import ThermalCamera, HAS_THERMAL_HARDWARE


def main():
    print("=" * 60)
    print("SAR System — MLX90640 Thermal Sensor Diagnostic Tool")
    print("=" * 60)

    if not HAS_THERMAL_HARDWARE:
        print("[LỖI] Thư viện adafruit_mlx90640 hoặc board không khả dụng trên máy này.")
        print("Vui lòng chạy script này trực tiếp trên NVIDIA Jetson Orin Nano.")
        sys.exit(1)

    print(f"[*] Cấu hình I2C Clock: {config.thermal.i2c_frequency} Hz")
    print(f"[*] Cấu hình Refresh Rate: {config.thermal.refresh_rate} Hz")
    print("[*] Đang khởi tạo kết nối cảm biến MLX90640 qua I2C...")

    try:
        cam = ThermalCamera()
        cam.start()
        print("[+] Kết nối I2C thành công! Đang đọc 10 khung hình thử nghiệm...\n")
        time.sleep(1.0)

        for i in range(1, 11):
            frame = cam.read()
            if frame is not None:
                t_min = float(frame.min())
                t_max = float(frame.max())
                t_mean = float(frame.mean())
                print(f"Khung #{i:02d} | T_Min: {t_min:.1f}°C | T_Max: {t_max:.1f}°C | T_Mean: {t_mean:.1f}°C")
            else:
                print(f"Khung #{i:02d} | [Đang chờ dữ liệu từ cảm biến...]")
            time.sleep(0.5)

        print("\n[+] Hoàn thành kiểm tra cảm biến nhiệt MLX90640 thành công.")
    except Exception as exc:
        print(f"\n[LỖI PHẦN CỨNG] Không thể giao tiếp với MLX90640: {exc}")
        print("Khuyến nghị:")
        print("1. Kiểm tra kết nối chân SDA, SCL, 3.3V, GND.")
        print("2. Chạy 'i2cdetect -y -r 1' xem địa chỉ 0x33 có xuất hiện không.")
        print("3. Đảm bảo chân Pin 3 (SDA), Pin 5 (SCL) trên 40-pin header của Jetson.")
    finally:
        try:
            cam.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Chụp ảnh vân tay từ SEN0348 và lưu thành file BMP (+ hiển thị).
Tương đương getQuarterFingerImage.ino trên Arduino.

Cách chạy:
    python3 get_finger_image.py                     # Ảnh 1/4 (80x80)
    python3 get_finger_image.py --full               # Ảnh đầy đủ (160x160)
    python3 get_finger_image.py --output my_fp.bmp   # Đổi tên file
    python3 get_finger_image.py --show               # Hiển thị ảnh sau khi chụp

Yêu cầu (tùy chọn, để hiển thị ảnh):
    pip3 install Pillow
"""

import sys
import os
import struct
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from DFRobot_ID809 import (
    DFRobot_ID809, LEDMode, LEDColor,
    ERR_SUCCESS
)


def build_bmp_header(width, height):
    """
    Tạo BMP header cho ảnh grayscale 8-bit (tương đương bmpHeader.h).
    
    BMP format:
        - File header: 14 bytes
        - Info header: 40 bytes
        - Color palette: 256 * 4 = 1024 bytes (grayscale)
        - Pixel data: width * height bytes
    Total header = 14 + 40 + 1024 = 1078 bytes
    """
    palette_size = 256 * 4  # 1024
    header_size = 14 + 40 + palette_size  # 1078
    image_size = width * height
    file_size = header_size + image_size

    bmp = bytearray()

    # === File Header (14 bytes) ===
    bmp += b'BM'                                    # Signature
    bmp += struct.pack('<I', file_size)             # File size
    bmp += struct.pack('<HH', 0, 0)                 # Reserved
    bmp += struct.pack('<I', header_size)            # Pixel data offset

    # === Info Header (40 bytes) ===
    bmp += struct.pack('<I', 40)                    # Info header size
    bmp += struct.pack('<i', width)                 # Width
    bmp += struct.pack('<i', height)                # Height (positive = bottom-up)
    bmp += struct.pack('<HH', 1, 8)                 # Planes=1, Bits per pixel=8
    bmp += struct.pack('<I', 0)                     # Compression (none)
    bmp += struct.pack('<I', 0)                     # Image size (can be 0 for uncompressed)
    bmp += struct.pack('<i', 0)                     # X pixels per meter
    bmp += struct.pack('<i', 0)                     # Y pixels per meter
    bmp += struct.pack('<I', 0)                     # Colors used
    bmp += struct.pack('<I', 0)                     # Important colors

    # === Color Palette (1024 bytes) - Grayscale ===
    for i in range(256):
        bmp += struct.pack('BBBB', i, i, i, 0)     # B, G, R, Reserved

    return bytes(bmp)


def save_bmp(filename, image_data, width, height):
    """Lưu dữ liệu ảnh grayscale thành file BMP."""
    header = build_bmp_header(width, height)

    # BMP lưu pixel từ dưới lên (bottom-up), cần flip theo hàng
    flipped = bytearray()
    for row in range(height - 1, -1, -1):
        row_start = row * width
        flipped.extend(image_data[row_start:row_start + width])

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(flipped)

    print(f"  Đã lưu: {filename} ({os.path.getsize(filename)} bytes)")


def show_image(filename):
    """Hiển thị ảnh BMP (cần thư viện Pillow)."""
    try:
        from PIL import Image
        img = Image.open(filename)
        img.show()
        print("  Đã mở ảnh để hiển thị.")
    except ImportError:
        print("  ⚠ Cần cài Pillow để hiển thị: pip3 install Pillow")
    except Exception as e:
        print(f"  ⚠ Không hiển thị được: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Chụp ảnh vân tay từ SEN0348 và lưu BMP"
    )
    parser.add_argument("--port", "-p", default="/dev/ttyTHS1",
                        help="Cổng serial (mặc định: /dev/ttyTHS1)")
    parser.add_argument("--baudrate", "-b", type=int, default=115200)
    parser.add_argument("--full", "-f", action="store_true",
                        help="Chụp ảnh đầy đủ 160x160 (mặc định: 80x80)")
    parser.add_argument("--output", "-o", default="finger.bmp",
                        help="Tên file BMP đầu ra (mặc định: finger.bmp)")
    parser.add_argument("--show", "-s", action="store_true",
                        help="Hiển thị ảnh sau khi chụp")
    parser.add_argument("--loop", "-l", action="store_true",
                        help="Chụp liên tục (Ctrl+C để dừng)")
    parser.add_argument("--debug", "-d", action="store_true")

    args = parser.parse_args()

    # Xác định kích thước ảnh
    if args.full:
        width, height = 160, 160
        img_type = "Full (160x160)"
    else:
        width, height = 80, 80
        img_type = "Quarter (80x80)"

    print(f"=== Chụp ảnh vân tay SEN0348 ===")
    print(f"  Loại ảnh: {img_type}")
    print(f"  File đầu ra: {args.output}")
    print()

    # Kết nối module
    fp = DFRobot_ID809(port=args.port, baudrate=args.baudrate, debug=args.debug)
    if not fp.begin():
        print("Không kết nối được module!")
        return

    print("Module sẵn sàng!\n")

    try:
        shot_count = 0
        while True:
            # LED xanh dương = đang chờ
            fp.ctrl_led(LEDMode.BREATHING, LEDColor.BLUE, 0)
            print("Đặt ngón tay lên cảm biến...")

            # Chờ ngón tay
            while not fp.detect_finger():
                time.sleep(0.05)

            print("Đang chụp ảnh...")
            fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.CYAN, 0)

            # Chụp ảnh
            if args.full:
                image_data = fp.get_finger_image()
            else:
                image_data = fp.get_quarter_finger_image()

            if image_data is None:
                fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
                print(f"  ✗ Chụp thất bại: {fp.get_error_description()}")
                time.sleep(1)
                if not args.loop:
                    break
                continue

            # Tạo tên file
            if args.loop:
                shot_count += 1
                name, ext = os.path.splitext(args.output)
                filename = f"{name}_{shot_count:03d}{ext}"
            else:
                filename = args.output

            # Lưu BMP
            fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.GREEN, 0)
            save_bmp(filename, image_data, width, height)
            print(f"  ✓ Chụp thành công! ({width}x{height}, {len(image_data)} bytes)")

            # Hiển thị
            if args.show:
                show_image(filename)

            if not args.loop:
                break

            # Chờ nhấc tay
            print("  Nhấc ngón tay ra để chụp tiếp...\n")
            while fp.detect_finger():
                time.sleep(0.1)
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nĐã dừng.")

    time.sleep(1)
    fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)
    fp.close()
    print("Done!")


if __name__ == "__main__":
    main()

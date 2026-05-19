"""
Thu thập ảnh vân tay từ SEN0348 → lưu thành dataset BMP cho fine-tuning.

Cấu trúc thư mục output (tương thích `FingerprintLoader.load_sensor_dataset`):
    sensor_dataset/
      person_001/
        sample_0.bmp
        sample_1.bmp
        ...
      person_002/
        ...

Cách dùng:
    python3 collect_sensor_data.py --person 001 --samples 10
    python3 collect_sensor_data.py --person 002 --samples 5 --port /dev/ttyTHS1
"""
import argparse
import os
import sys
import time
import cv2
import numpy as np

from config import SENSOR_UART_PORT, SENSOR_BAUD_RATE, SENSOR_IMG_SIZE
from DFRobot_ID809 import DFRobot_ID809, LEDMode, LEDColor


def capture_image(fp):
    data = fp.get_quarter_finger_image()
    if data is None:
        return None
    w, h = SENSOR_IMG_SIZE
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    if len(arr) != w * h:
        return None
    return arr.reshape((h, w))


def wait_finger(fp, timeout_s=15):
    fp.ctrl_led(LEDMode.BREATHING, LEDColor.BLUE, 0)
    print("  Đặt ngón tay lên cảm biến...")
    start = time.time()
    while not fp.detect_finger():
        time.sleep(0.05)
        if time.time() - start > timeout_s:
            return False
    return True


def wait_release(fp):
    print("  Nhấc ngón tay ra...")
    while fp.detect_finger():
        time.sleep(0.1)
    time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(description="Thu thập ảnh SEN0348 cho fine-tuning")
    parser.add_argument('--person', required=True, help='ID người dùng (vd: 001)')
    parser.add_argument('--samples', type=int, default=10, help='Số ảnh cần thu (mặc định 10)')
    parser.add_argument('--out', default='sensor_dataset', help='Thư mục output')
    parser.add_argument('--port', default=SENSOR_UART_PORT)
    parser.add_argument('--baud', type=int, default=SENSOR_BAUD_RATE)
    args = parser.parse_args()

    person_dir = os.path.join(args.out, f"person_{args.person}")
    os.makedirs(person_dir, exist_ok=True)
    existing = len([f for f in os.listdir(person_dir) if f.endswith('.bmp')])
    print(f"Thư mục: {person_dir} (đã có {existing} ảnh)")

    fp = DFRobot_ID809(port=args.port, baudrate=args.baud)
    if not fp.begin():
        print("Không kết nối được SEN0348!")
        sys.exit(1)
    print(f"SEN0348 OK, capacity={fp.FINGERPRINT_CAPACITY}")

    captured = 0
    try:
        while captured < args.samples:
            print(f"\n--- Mẫu {captured+1}/{args.samples} ---")
            if not wait_finger(fp):
                print("  Hết thời gian chờ, thử lại...")
                fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
                continue

            fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.CYAN, 0)
            img = capture_image(fp)
            if img is None:
                print(f"  Chụp lỗi: {fp.get_error_description()}")
                fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
                wait_release(fp)
                continue

            idx = existing + captured
            path = os.path.join(person_dir, f"sample_{idx}.bmp")
            cv2.imwrite(path, img)
            print(f"  Đã lưu: {path}")
            fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.GREEN, 2)
            captured += 1
            wait_release(fp)
    except KeyboardInterrupt:
        print("\nDừng bởi người dùng.")
    finally:
        fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)
        fp.close()
        print(f"\nHoàn tất. Đã thu {captured} ảnh cho person_{args.person}.")


if __name__ == "__main__":
    main()

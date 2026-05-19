#!/usr/bin/env python3
"""
Ví dụ: Nhận diện vân tay 1:N trên SEN0348 (Jetson Nano)

Quét vân tay và so sánh với toàn bộ thư viện đã đăng ký.

Cách chạy:
    python3 fingerprint_matching.py
"""

import sys
import time
sys.path.insert(0, '.')
from DFRobot_ID809 import (
    DFRobot_ID809, LEDMode, LEDColor,
    ERR_SUCCESS, ERR_ID809
)


def main():
    fp = DFRobot_ID809(port="/dev/ttyTHS1", baudrate=115200)

    if not fp.begin():
        print("Không kết nối được module!")
        return

    count = fp.get_enroll_count()
    print(f"Module sẵn sàng! Số vân tay đã đăng ký: {count}")

    if count == 0:
        print("Chưa có vân tay nào! Hãy đăng ký trước.")
        fp.close()
        return

    print("\nĐang chờ quét vân tay... (Ctrl+C để thoát)\n")

    try:
        while True:
            # LED xanh dương nhấp nháy = đang chờ
            fp.ctrl_led(LEDMode.BREATHING, LEDColor.BLUE, 0)

            # Thu thập vân tay (chờ vô hạn)
            ret = fp.collection_fingerprint(timeout_s=0)
            if ret != ERR_SUCCESS:
                fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
                print(f"Lỗi thu thập: {fp.get_error_description()}")
                time.sleep(1)
                continue

            # So sánh 1:N
            matched_id = fp.search()
            if matched_id > 0:
                # LED xanh lá = nhận diện thành công
                fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.GREEN, 0)
                print(f"✓ Vân tay khớp! ID = {matched_id}")
            else:
                # LED đỏ = không nhận diện được
                fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.RED, 0)
                print("✗ Vân tay không khớp!")

            time.sleep(2)

            # Chờ nhấc ngón tay
            print("Nhấc ngón tay ra để quét tiếp...\n")
            while fp.detect_finger():
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nĐã dừng.")

    fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)
    fp.close()


if __name__ == "__main__":
    main()

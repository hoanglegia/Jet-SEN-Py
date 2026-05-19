#!/usr/bin/env python3
"""
Ví dụ: Đăng ký vân tay mới trên SEN0348 (Jetson Nano)

Quy trình: Thu thập 3 lần vân tay -> Merge -> Lưu vào ID

Cách chạy:
    python3 fingerprint_registration.py
"""

import sys
import time
sys.path.insert(0, '.')
from DFRobot_ID809 import (
    DFRobot_ID809, LEDMode, LEDColor,
    ERR_SUCCESS, ERR_ID809
)


def main():
    # Khởi tạo - thay đổi port nếu cần
    fp = DFRobot_ID809(port="/dev/ttyTHS1", baudrate=115200)

    if not fp.begin():
        print("Không kết nối được module! Kiểm tra dây nối.")
        return

    print("Module sẵn sàng!")
    print(f"Số vân tay đã đăng ký: {fp.get_enroll_count()}")

    # Tìm ID trống
    empty_id = fp.get_empty_id()
    if empty_id == ERR_ID809:
        print("Bộ nhớ đã đầy! Xóa bớt vân tay.")
        fp.close()
        return

    print(f"\nSẽ đăng ký vân tay vào ID: {empty_id}")
    print("Cần quét vân tay 3 lần.\n")

    # Thu thập 3 lần
    for i in range(3):
        # LED xanh nhấp nháy = đang chờ
        fp.ctrl_led(LEDMode.BREATHING, LEDColor.BLUE, 0)
        print(f"[Lần {i+1}/3] Đặt ngón tay lên cảm biến...")

        ret = fp.collection_fingerprint(timeout_s=10)
        if ret != ERR_SUCCESS:
            # LED đỏ = thất bại
            fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
            print(f"  Thất bại: {fp.get_error_description()}")
            fp.close()
            return

        # LED xanh lá = OK
        fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.GREEN, 3)
        print(f"  ✓ Thu thập thành công!")

        if i < 2:
            print("  Nhấc ngón tay ra...")
            while fp.detect_finger():
                time.sleep(0.1)
            time.sleep(0.5)

    # Lưu vân tay
    print(f"\nĐang lưu vân tay vào ID {empty_id}...")
    ret = fp.store_fingerprint(empty_id)
    if ret == ERR_SUCCESS:
        fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.GREEN, 0)
        print(f"✓ Đăng ký thành công! ID = {empty_id}")
        print(f"Tổng vân tay: {fp.get_enroll_count()}")
    else:
        fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.RED, 0)
        print(f"✗ Lưu thất bại: {fp.get_error_description()}")

    time.sleep(2)
    fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)
    fp.close()


if __name__ == "__main__":
    main()

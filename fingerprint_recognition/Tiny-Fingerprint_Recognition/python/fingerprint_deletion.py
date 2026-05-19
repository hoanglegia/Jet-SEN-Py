#!/usr/bin/env python3
"""
Ví dụ: Xóa vân tay trên SEN0348 (Jetson Nano)

Cách chạy:
    python3 fingerprint_deletion.py
"""

import sys
import time
sys.path.insert(0, '.')
from DFRobot_ID809 import (
    DFRobot_ID809, LEDMode, LEDColor,
    ERR_SUCCESS, ERR_ID809, DELALL
)


def main():
    fp = DFRobot_ID809(port="/dev/ttyTHS1", baudrate=115200)

    if not fp.begin():
        print("Không kết nối được module!")
        return

    count = fp.get_enroll_count()
    print(f"Số vân tay đã đăng ký: {count}")

    if count == 0:
        print("Không có vân tay nào để xóa.")
        fp.close()
        return

    print("\nChọn thao tác:")
    print("  1. Xóa vân tay theo ID")
    print("  2. Xóa TẤT CẢ vân tay")
    print("  3. Thoát")

    choice = input("\nNhập lựa chọn (1/2/3): ").strip()

    if choice == '1':
        try:
            fp_id = int(input("Nhập ID cần xóa: ").strip())
        except ValueError:
            print("ID không hợp lệ!")
            fp.close()
            return

        status = fp.get_status_id(fp_id)
        if status == 1:
            print(f"ID {fp_id} chưa được đăng ký!")
            fp.close()
            return

        ret = fp.del_fingerprint(fp_id)
        if ret == ERR_SUCCESS:
            fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.GREEN, 3)
            print(f"✓ Đã xóa vân tay ID {fp_id}")
        else:
            fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
            print(f"✗ Xóa thất bại: {fp.get_error_description()}")

    elif choice == '2':
        confirm = input("Xác nhận xóa TẤT CẢ? (y/n): ").strip().lower()
        if confirm == 'y':
            ret = fp.del_fingerprint(DELALL)
            if ret == ERR_SUCCESS:
                fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.GREEN, 3)
                print("✓ Đã xóa tất cả vân tay!")
            else:
                fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
                print(f"✗ Xóa thất bại: {fp.get_error_description()}")
        else:
            print("Đã hủy.")

    print(f"\nSố vân tay còn lại: {fp.get_enroll_count()}")

    time.sleep(2)
    fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)
    fp.close()


if __name__ == "__main__":
    main()

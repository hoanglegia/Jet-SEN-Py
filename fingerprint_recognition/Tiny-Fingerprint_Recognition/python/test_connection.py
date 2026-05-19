#!/usr/bin/env python3
"""
Script kiểm tra kết nối SEN0348 (DFRobot ID809) với Jetson Nano B01.

Chạy script này ĐẦU TIÊN để xác nhận module đã được nối dây đúng 
và giao tiếp UART hoạt động bình thường.

Cách chạy:
    python3 test_connection.py
    
Nếu muốn thay đổi cổng serial:
    python3 test_connection.py --port /dev/ttyUSB0
"""

import sys
import argparse
import time

# Thêm thư mục hiện tại vào path
sys.path.insert(0, '.')
from DFRobot_ID809 import (
    DFRobot_ID809, LEDMode, LEDColor,
    ERR_SUCCESS, ERR_ID809
)


def print_separator():
    print("=" * 60)


def print_header(text):
    print_separator()
    print(f"  {text}")
    print_separator()


def test_connection(port, baudrate, debug):
    """Kiểm tra kết nối cơ bản."""
    
    print_header("TEST KẾT NỐI SEN0348 VỚI JETSON NANO")
    print(f"  Cổng serial : {port}")
    print(f"  Baudrate    : {baudrate}")
    print(f"  Debug mode  : {'ON' if debug else 'OFF'}")
    print_separator()
    print()

    # ---- Bước 1: Khởi tạo serial ----
    print("[1/6] Khởi tạo kết nối serial...")
    fp = DFRobot_ID809(port=port, baudrate=baudrate, debug=debug)
    
    try:
        fp._ser = __import__('serial').Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=2.0
        )
        time.sleep(0.1)
        fp._ser.reset_input_buffer()
        fp._ser.reset_output_buffer()
        print("  ✓ Mở cổng serial thành công!")
    except Exception as e:
        print(f"  ✗ THẤT BẠI! Không mở được cổng serial: {e}")
        print()
        print("  Gợi ý khắc phục:")
        print("    1. Kiểm tra cổng serial có tồn tại: ls -l /dev/ttyTHS* /dev/ttyUSB*")
        print("    2. Thêm quyền truy cập: sudo usermod -aG dialout $USER")
        print("    3. Kiểm tra dây nối TX/RX đã đúng chưa")
        print("    4. Nếu dùng USB-UART adapter, thử cổng /dev/ttyUSB0")
        return False
    
    print()

    # ---- Bước 2: Test Connection Command ----
    print("[2/6] Gửi lệnh Test Connection (CMD 0x0001)...")
    connected = fp.is_connected()
    if connected:
        print("  ✓ Module phản hồi OK! Kết nối thành công!")
    else:
        print("  ✗ THẤT BẠI! Module không phản hồi.")
        print()
        print("  Gợi ý khắc phục:")
        print("    1. Kiểm tra nguồn cấp cho module (3.3V hoặc 5V)")
        print("    2. Kiểm tra đấu chéo TX-RX (TX Jetson -> RX Module)")
        print("    3. Kiểm tra GND chung giữa Jetson và Module")
        print("    4. Thử baudrate khác: 9600, 19200, 57600, 115200")
        print("    5. Nếu dùng UART trên GPIO, chạy: sudo systemctl stop nvgetty")
        fp.close()
        return False

    print()

    # ---- Bước 3: Đọc Device Info ----
    print("[3/6] Đọc thông tin thiết bị...")
    info = fp.get_device_info()
    if info:
        print(f"  ✓ Device Info: {info}")
        print(f"  ✓ Fingerprint Capacity: {fp.FINGERPRINT_CAPACITY}")
    else:
        print("  ⚠ Không đọc được thông tin thiết bị (không nghiêm trọng)")

    print()

    # ---- Bước 4: Đọc Module SN ----
    print("[4/6] Đọc serial number module...")
    sn = fp.get_module_sn()
    if sn:
        print(f"  ✓ Module SN: {sn}")
    else:
        print("  ⚠ Không đọc được SN (có thể chưa được đặt)")

    print()

    # ---- Bước 5: Đọc các tham số ----
    print("[5/6] Đọc các tham số cấu hình module...")
    
    device_id = fp.get_device_id()
    security = fp.get_security_level()
    dup_check = fp.get_duplication_check()
    baud = fp.get_baudrate()
    self_learn = fp.get_self_learn()
    
    if device_id != ERR_ID809:
        print(f"  ✓ Device ID       : {device_id}")
    if security != ERR_ID809:
        print(f"  ✓ Security Level  : {security}")
    if dup_check != ERR_ID809:
        print(f"  ✓ Duplication Check: {'ON' if dup_check else 'OFF'}")
    baud_map = {1: 9600, 2: 19200, 3: 38400, 4: 57600, 5: 115200}
    if baud != ERR_ID809:
        print(f"  ✓ Baudrate        : {baud_map.get(baud, 'Unknown')} (code={baud})")
    if self_learn != ERR_ID809:
        print(f"  ✓ Self Learn      : {'ON' if self_learn else 'OFF'}")
    
    enroll_count = fp.get_enroll_count()
    if enroll_count != ERR_ID809:
        print(f"  ✓ Registered FPs  : {enroll_count}")

    print()

    # ---- Bước 6: Test LED ----
    print("[6/6] Test LED (nhấp nháy xanh 3 lần)...")
    ret = fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.GREEN, 3)
    if ret == ERR_SUCCESS:
        print("  ✓ LED đang nhấp nháy! Kiểm tra bằng mắt.")
    else:
        print("  ⚠ Không điều khiển được LED")

    time.sleep(2)
    
    # Tắt LED
    fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)

    print()
    print_header("KẾT QUẢ: TẤT CẢ TESTS ĐẠT! KẾT NỐI OK!")
    print()
    print("  Module SEN0348 đã sẵn sàng sử dụng.")
    print("  Bạn có thể chạy các ví dụ khác:")
    print("    - fingerprint_registration.py  (Đăng ký vân tay)")
    print("    - fingerprint_matching.py      (Nhận diện vân tay)")
    print("    - fingerprint_deletion.py      (Xóa vân tay)")
    print()

    fp.close()
    return True


def scan_baudrates(port, debug):
    """Quét thử các baudrate để tìm baudrate đúng."""
    
    print_header("QUÉT BAUDRATE")
    print(f"  Thử kết nối trên cổng {port} với các baudrate khác nhau...")
    print()
    
    baudrates = [115200, 57600, 38400, 19200, 9600]
    
    for baud in baudrates:
        print(f"  Thử {baud} bps... ", end="", flush=True)
        fp = DFRobot_ID809(port=port, baudrate=baud, timeout=1.0, debug=debug)
        try:
            fp._ser = __import__('serial').Serial(
                port=port, baudrate=baud,
                bytesize=8, parity='N', stopbits=1, timeout=1.0
            )
            time.sleep(0.1)
            fp._ser.reset_input_buffer()
            if fp.is_connected():
                print(f"✓ KẾT NỐI THÀNH CÔNG!")
                fp.close()
                print()
                print(f"  >> Baudrate đúng là: {baud}")
                print(f"  >> Chạy lại với: python3 test_connection.py --baudrate {baud}")
                return True
            else:
                print("✗")
            fp.close()
        except Exception:
            print("✗ (không mở được)")
    
    print()
    print("  Không tìm thấy baudrate phù hợp!")
    print("  Kiểm tra lại dây nối và nguồn cấp cho module.")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra kết nối SEN0348 Fingerprint Sensor với Jetson Nano"
    )
    parser.add_argument(
        "--port", "-p", 
        default="/dev/ttyTHS1",
        help="Cổng serial (mặc định: /dev/ttyTHS1)"
    )
    parser.add_argument(
        "--baudrate", "-b", 
        type=int, default=115200,
        help="Baudrate (mặc định: 115200)"
    )
    parser.add_argument(
        "--debug", "-d", 
        action="store_true",
        help="Bật chế độ debug (hiện dữ liệu gửi/nhận)"
    )
    parser.add_argument(
        "--scan", "-s", 
        action="store_true",
        help="Quét thử các baudrate để tìm baudrate đúng"
    )
    
    args = parser.parse_args()
    
    if args.scan:
        scan_baudrates(args.port, args.debug)
    else:
        test_connection(args.port, args.baudrate, args.debug)


if __name__ == "__main__":
    main()

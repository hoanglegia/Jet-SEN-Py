"""
Ứng dụng nhận diện vân tay trên Jetson Nano B01.
Kết hợp: DFRobot_ID809 (SEN0348) + Siamese Network TFLite.

Quy trình:
  - Dùng DFRobot_ID809 để chụp ảnh vân tay 80x80 từ SEN0348
  - Dùng Siamese Network (TFLite) để so sánh ảnh vân tay
  - Lưu ảnh đã đăng ký trong enrolled_fingerprints/

Cách chạy:
  python3 main.py                                    # Mặc định
  python3 main.py --port /dev/ttyUSB0                # Đổi port
  python3 main.py --model my_model.tflite            # Đổi model
  python3 main.py --threshold 0.6                    # Đổi ngưỡng
"""
import argparse
import sys
import os
import time
import numpy as np

from config import (
    MATCH_THRESHOLD, ENROLL_SAMPLES,
    SENSOR_UART_PORT, SENSOR_BAUD_RATE, SENSOR_IMG_SIZE
)
from inference import FingerprintEngine
from fingerprint_db import FingerprintDB
from DFRobot_ID809 import DFRobot_ID809, LEDMode, LEDColor, ERR_SUCCESS


def capture_fingerprint_image(fp):
    """
    Chụp ảnh vân tay 80x80 từ SEN0348, trả về numpy array grayscale.
    Sử dụng get_quarter_finger_image() giống get_finger_image.py.
    """
    # Chụp ảnh
    image_data = fp.get_quarter_finger_image()
    if image_data is None:
        return None

    # Chuyển sang numpy array 80x80
    w, h = SENSOR_IMG_SIZE
    img = np.frombuffer(bytes(image_data), dtype=np.uint8)
    if len(img) != w * h:
        print(f"[Capture] Kích thước ảnh không đúng: {len(img)} (cần {w*h})")
        return None
    img = img.reshape((h, w))
    return img


class FingerprintApp:
    def __init__(self, fp_sensor, engine, db, threshold=None):
        self.fp = fp_sensor          # DFRobot_ID809 instance
        self.engine = engine         # TFLite inference engine
        self.db = db                 # Fingerprint database
        self.threshold = threshold or MATCH_THRESHOLD
        self.enrolled_cache = {}     # Cache ảnh đã đăng ký trong RAM

    def reload_cache(self):
        """Tải lại tất cả ảnh đã đăng ký vào RAM."""
        self.enrolled_cache = self.db.load_all_enrolled()
        total = sum(len(v) for v in self.enrolled_cache.values())
        print(f"[App] Cache: {len(self.enrolled_cache)} người dùng, {total} mẫu.")

    def _wait_and_capture(self, timeout_s=10):
        """Chờ ngón tay + chụp ảnh. Trả về numpy array hoặc None."""
        fp = self.fp
        fp.ctrl_led(LEDMode.BREATHING, LEDColor.BLUE, 0)
        print("  Đặt ngón tay lên cảm biến...")

        start = time.time()
        while not fp.detect_finger():
            time.sleep(0.05)
            if timeout_s > 0 and (time.time() - start) > timeout_s:
                print("  Hết thời gian chờ!")
                fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
                return None

        fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.CYAN, 0)
        img = capture_fingerprint_image(fp)
        if img is None:
            fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
            print(f"  Chụp thất bại: {fp.get_error_description()}")
            return None

        fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.GREEN, 3)
        return img

    def _wait_finger_release(self):
        """Chờ người dùng nhấc ngón tay."""
        print("  Nhấc ngón tay ra...")
        while self.fp.detect_finger():
            time.sleep(0.1)
        time.sleep(0.3)

    # ===================== ĐĂNG KÝ VÂN TAY =====================
    def enroll(self):
        """Đăng ký vân tay mới: quét N lần → lưu vào database."""
        user_id = input("Nhập ID người dùng: ").strip()
        if not user_id:
            print("ID không được để trống!")
            return
        name = input("Nhập tên người dùng: ").strip() or user_id

        print(f"\nCần quét vân tay {ENROLL_SAMPLES} lần để đăng ký.")
        images = []

        for i in range(ENROLL_SAMPLES):
            print(f"\n--- Lần quét {i+1}/{ENROLL_SAMPLES} ---")
            img = self._wait_and_capture()
            if img is None:
                print("Hủy đăng ký.")
                return
            images.append(img)
            print(f"  Đã chụp mẫu {i+1}.")

            if i < ENROLL_SAMPLES - 1:
                self._wait_finger_release()

        # Kiểm tra nhất quán giữa các mẫu
        if len(images) >= 2:
            for i in range(len(images) - 1):
                score, is_match, _ = self.engine.compare(images[i], images[i+1])
                if not is_match:
                    print(f"\n  Cảnh báo: Mẫu {i+1} và {i+2} không khớp (score={score:.3f}).")
                    confirm = input("  Vẫn tiếp tục đăng ký? (y/n): ").strip().lower()
                    if confirm != 'y':
                        print("  Đã hủy đăng ký.")
                        return

        success = self.db.enroll(user_id, name, images)
        if success:
            self.reload_cache()
            self.fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.GREEN, 0)
            print(f"\n=== Đăng ký thành công: {name} (ID: {user_id}) ===")
            time.sleep(1)
            self.fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)

    # ===================== XÁC THỰC VÂN TAY (1:1) =====================
    def verify(self):
        """Xác thực: so khớp với 1 người dùng cụ thể."""
        user_id = input("Nhập ID người dùng cần xác thực: ").strip()
        if user_id not in self.enrolled_cache:
            print(f"Không tìm thấy user '{user_id}' trong database.")
            return

        img = self._wait_and_capture()
        if img is None:
            return

        ref_images = self.enrolled_cache[user_id]
        best_score = 0.0
        total_time = 0.0

        for ref in ref_images:
            score, _, t = self.engine.compare(img, ref)
            total_time += t
            if score > best_score:
                best_score = score

        name = self.db.get_user_name(user_id)
        is_match = best_score > self.threshold

        if is_match:
            self.fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.GREEN, 0)
        else:
            self.fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.RED, 0)

        print(f"\n{'='*40}")
        if is_match:
            print(f"  XÁC THỰC THÀNH CÔNG")
            print(f"  Người dùng: {name} (ID: {user_id})")
        else:
            print(f"  XÁC THỰC THẤT BẠI")
        print(f"  Điểm tương đồng: {best_score:.4f}")
        print(f"  Thời gian: {total_time:.1f} ms")
        print(f"{'='*40}")
        time.sleep(2)
        self.fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)

    # ===================== NHẬN DIỆN VÂN TAY (1:N) =====================
    def identify(self):
        """Nhận diện: tìm trong toàn bộ database."""
        if not self.enrolled_cache:
            print("Database trống. Hãy đăng ký vân tay trước.")
            return

        img = self._wait_and_capture()
        if img is None:
            return

        best_user, best_score, total_time = self.engine.find_best_match(
            img, self.enrolled_cache, threshold=self.threshold
        )

        if best_user:
            self.fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.GREEN, 0)
        else:
            self.fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.RED, 0)

        print(f"\n{'='*40}")
        if best_user:
            name = self.db.get_user_name(best_user)
            print(f"  NHẬN DIỆN THÀNH CÔNG")
            print(f"  Người dùng: {name} (ID: {best_user})")
        else:
            print(f"  KHÔNG NHẬN DIỆN ĐƯỢC")
            print(f"  Vân tay không khớp với ai trong database.")
        print(f"  Điểm cao nhất: {best_score:.4f}")
        print(f"  Thời gian: {total_time:.1f} ms")
        print(f"  Số người trong DB: {len(self.enrolled_cache)}")
        print(f"{'='*40}")
        time.sleep(2)
        self.fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)

    # ===================== CHẾ ĐỘ REALTIME =====================
    def realtime_identify(self):
        """Nhận diện liên tục (loop) — dùng cho access control."""
        if not self.enrolled_cache:
            print("Database trống. Hãy đăng ký vân tay trước.")
            return

        print("Chế độ nhận diện liên tục. Nhấn Ctrl+C để dừng.\n")
        try:
            while True:
                self.fp.ctrl_led(LEDMode.BREATHING, LEDColor.BLUE, 0)

                # Chờ ngón tay (vô hạn)
                while not self.fp.detect_finger():
                    time.sleep(0.05)

                self.fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.CYAN, 0)
                img = capture_fingerprint_image(self.fp)
                if img is None:
                    self.fp.ctrl_led(LEDMode.FAST_BLINK, LEDColor.RED, 3)
                    time.sleep(1)
                    continue

                best_user, best_score, total_time = self.engine.find_best_match(
                    img, self.enrolled_cache, threshold=self.threshold
                )

                if best_user:
                    name = self.db.get_user_name(best_user)
                    self.fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.GREEN, 0)
                    print(f"  [MATCH] {name} (ID: {best_user}) | score={best_score:.3f} | {total_time:.0f}ms")
                else:
                    self.fp.ctrl_led(LEDMode.KEEPS_ON, LEDColor.RED, 0)
                    print(f"  [NO MATCH] score={best_score:.3f} | {total_time:.0f}ms")

                time.sleep(1.5)

                # Chờ nhấc tay
                while self.fp.detect_finger():
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nĐã dừng.")
            self.fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)

    # ===================== QUẢN LÝ DATABASE =====================
    def list_users(self):
        users = self.db.list_users()
        if not users:
            print("Database trống.")
            return
        print(f"\n{'='*55}")
        print(f"  {'ID':<12} {'Tên':<20} {'Mẫu':<6} {'Ngày đăng ký'}")
        print(f"  {'-'*53}")
        for u in users:
            print(f"  {u['user_id']:<12} {u['name']:<20} {u['num_samples']:<6} {u['enrolled_at']}")
        print(f"{'='*55}")
        print(f"  Tổng: {len(users)} người dùng")

    def delete_user(self):
        user_id = input("Nhập ID người dùng cần xóa: ").strip()
        confirm = input(f"Xác nhận xóa '{user_id}'? (y/n): ").strip().lower()
        if confirm == 'y':
            if self.db.delete(user_id):
                self.reload_cache()


def main():
    parser = argparse.ArgumentParser(description="Fingerprint Recognition - Jetson Nano B01")
    parser.add_argument('--port', default=SENSOR_UART_PORT,
                        help=f'UART port (mặc định: {SENSOR_UART_PORT})')
    parser.add_argument('--baud', type=int, default=SENSOR_BAUD_RATE,
                        help=f'Baudrate (mặc định: {SENSOR_BAUD_RATE})')
    parser.add_argument('--model', default=None, help='Đường dẫn file .tflite')
    parser.add_argument('--threshold', type=float, default=MATCH_THRESHOLD,
                        help=f'Ngưỡng khớp (mặc định: {MATCH_THRESHOLD})')
    args = parser.parse_args()

    # === 1. Kết nối cảm biến SEN0348 ===
    print(f"Kết nối SEN0348: {args.port} @ {args.baud}...")
    fp = DFRobot_ID809(port=args.port, baudrate=args.baud)
    if not fp.begin():
        print("Không kết nối được cảm biến SEN0348!")
        print("Kiểm tra: dây nối, UART port, baudrate.")
        sys.exit(1)
    print(f"SEN0348 OK! Capacity: {fp.FINGERPRINT_CAPACITY}")

    # === 2. Khởi tạo TFLite engine ===
    engine = FingerprintEngine(model_path=args.model, threshold=args.threshold)

    # === 3. Khởi tạo database ===
    db = FingerprintDB()
    app = FingerprintApp(fp, engine, db, threshold=args.threshold)
    app.reload_cache()

    # === Menu chính ===
    try:
        while True:
            print("\n" + "="*50)
            print("  NHẬN DIỆN VÂN TAY - JETSON NANO + SIAMESE NET")
            print("="*50)
            print("  1. Đăng ký vân tay mới          (Enroll)")
            print("  2. Xác thực vân tay 1:1          (Verify)")
            print("  3. Nhận diện vân tay 1:N          (Identify)")
            print("  4. Nhận diện liên tục             (Realtime)")
            print("  5. Liệt kê người dùng")
            print("  6. Xóa người dùng")
            print("  0. Thoát")
            print("-"*50)

            choice = input("Chọn chức năng: ").strip()

            if choice == '1':
                app.enroll()
            elif choice == '2':
                app.verify()
            elif choice == '3':
                app.identify()
            elif choice == '4':
                app.realtime_identify()
            elif choice == '5':
                app.list_users()
            elif choice == '6':
                app.delete_user()
            elif choice == '0':
                break
            else:
                print("Lựa chọn không hợp lệ.")
    except KeyboardInterrupt:
        print("\nThoát chương trình.")
    finally:
        fp.ctrl_led(LEDMode.NORMAL_CLOSE, LEDColor.GREEN, 0)
        fp.close()
        print("Đã ngắt kết nối cảm biến.")


if __name__ == "__main__":
    main()

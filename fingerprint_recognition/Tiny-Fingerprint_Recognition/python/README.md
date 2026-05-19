# SEN0348 (DFRobot ID809) - Python cho Jetson Nano B01

Thư viện Python cho module vân tay điện dung SEN0348, chuyển đổi từ thư viện Arduino C++.

## Yêu cầu

- Jetson Nano B01 (JetPack 4.x)
- Python 3.6+
- Module SEN0348 (DFRobot ID809 Capacitive Fingerprint Sensor)

### Cài đặt thư viện Python

```bash
pip3 install pyserial
```

---

## Sơ đồ nối dây (Jetson Nano B01 ↔ SEN0348)

```
Jetson Nano 40-Pin Header          SEN0348 Module
┌─────────────────────┐            ┌──────────────┐
│ Pin 1  (3.3V)   ────┼──── hoặc ──┤ VCC (đỏ)    │
│ Pin 2  (5V)     ────┼────────────┤ VCC (đỏ)    │  ← Dùng 5V nếu module yêu cầu
│ Pin 6  (GND)    ────┼────────────┤ GND (đen)   │
│ Pin 8  (TXD)    ────┼────────────┤ RX  (xanh)  │  ← TX Jetson → RX Module
│ Pin 10 (RXD)    ────┼────────────┤ TX  (vàng)  │  ← RX Jetson ← TX Module
└─────────────────────┘            └──────────────┘
```

> **⚠️ LƯU Ý QUAN TRỌNG:**
> - **TX nối RX, RX nối TX** (nối chéo)
> - **GND phải được nối chung** giữa Jetson và Module
> - Module SEN0348 hỗ trợ 3.3V~5V. Dùng **Pin 2 (5V)** nếu module cần 5V
> - UART trên Jetson Nano (Pin 8, 10) là **3.3V logic** - tương thích trực tiếp

### Cáp kết nối của SEN0348 (Gravity Interface)

| Màu dây | Chức năng |
|---------|-----------|
| Đỏ     | VCC       |
| Đen    | GND       |
| Xanh   | RX        |
| Vàng   | TX        |

---

## Cấu hình UART trên Jetson Nano

### Bước 1: Tắt serial console (chỉ cần làm 1 lần)

Mặc định, UART trên Pin 8/10 được dùng cho serial console. Phải tắt đi:

```bash
# Tắt nvgetty service
sudo systemctl stop nvgetty
sudo systemctl disable nvgetty

# Reboot
sudo reboot
```

### Bước 2: Kiểm tra cổng serial

```bash
# Liệt kê cổng serial
ls -l /dev/ttyTHS*

# Bạn sẽ thấy /dev/ttyTHS1 (UART2)
```

### Bước 3: Phân quyền

```bash
# Thêm user vào group dialout
sudo usermod -aG dialout $USER

# Hoặc cấp quyền trực tiếp (tạm thời)
sudo chmod 666 /dev/ttyTHS1
```

---

## Kiểm tra kết nối

### Cách nhanh nhất

```bash
cd python/
python3 test_connection.py
```

### Nếu không biết baudrate

```bash
python3 test_connection.py --scan
```

### Nếu dùng cổng serial khác (ví dụ USB-UART adapter)

```bash
python3 test_connection.py --port /dev/ttyUSB0
```

### Bật chế độ debug (xem chi tiết dữ liệu gửi/nhận)

```bash
python3 test_connection.py --debug
```

### Kết quả mong đợi

```
============================================================
  TEST KẾT NỐI SEN0348 VỚI JETSON NANO
============================================================
  Cổng serial : /dev/ttyTHS1
  Baudrate    : 115200
============================================================

[1/6] Khởi tạo kết nối serial...
  ✓ Mở cổng serial thành công!

[2/6] Gửi lệnh Test Connection (CMD 0x0001)...
  ✓ Module phản hồi OK! Kết nối thành công!

[3/6] Đọc thông tin thiết bị...
  ✓ Device Info: ID809-AE-V24
  ✓ Fingerprint Capacity: 80

[4/6] Đọc serial number module...
  ✓ Module SN: ...

[5/6] Đọc các tham số cấu hình module...
  ✓ Device ID       : 1
  ✓ Security Level  : 3
  ✓ Baudrate        : 115200

[6/6] Test LED (nhấp nháy xanh 3 lần)...
  ✓ LED đang nhấp nháy!

============================================================
  KẾT QUẢ: TẤT CẢ TESTS ĐẠT! KẾT NỐI OK!
============================================================
```

---

## Các ví dụ

| File | Mô tả |
|------|--------|
| `test_connection.py` | Kiểm tra kết nối cơ bản |
| `fingerprint_registration.py` | Đăng ký vân tay mới |
| `fingerprint_matching.py` | Nhận diện vân tay (1:N) |
| `fingerprint_deletion.py` | Xóa vân tay |

---

## Troubleshooting

### "Không mở được cổng serial"
```bash
# Kiểm tra cổng có tồn tại
ls -l /dev/ttyTHS* /dev/ttyUSB*

# Kiểm tra quyền
sudo chmod 666 /dev/ttyTHS1

# Tắt serial console
sudo systemctl stop nvgetty
sudo systemctl disable nvgetty
```

### "Module không phản hồi"
1. Kiểm tra nguồn cấp (đèn LED trên module có sáng không?)
2. Kiểm tra nối chéo TX↔RX
3. Kiểm tra GND chung
4. Thử quét baudrate: `python3 test_connection.py --scan`
5. Thử dùng USB-UART adapter để loại trừ lỗi phần cứng

### "Permission denied"
```bash
sudo usermod -aG dialout $USER
# Sau đó đăng xuất và đăng nhập lại
```

### Dùng USB-UART adapter thay vì GPIO UART
Nếu bạn dùng adapter USB-to-UART (CP2102, CH340, FT232):
```bash
# Tìm cổng
ls /dev/ttyUSB*

# Chạy với cổng USB
python3 test_connection.py --port /dev/ttyUSB0
```

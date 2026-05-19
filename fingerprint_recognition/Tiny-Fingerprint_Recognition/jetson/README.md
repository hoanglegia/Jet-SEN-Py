# Triển khai nhận diện vân tay trên Jetson Nano B01

## Tổng quan

Sử dụng **Siamese Network (TFLite)** + **DFRobot SEN0348** trên **Jetson Nano B01**.

- Model Siamese Network train trên Windows bằng dataset SOCOFing
- Xuất `.tflite` → copy sang Jetson Nano
- (Khuyến nghị) **Fine-tune** với ảnh thật từ SEN0348 để giảm domain gap
- Trên Jetson: `DFRobot_ID809.py` chụp ảnh 80x80, TFLite + CLAHE + FFT denoise so sánh

```
[PC] Train SOCOFing → .tflite → [Jetson] (collect SEN0348 → finetune) → main.py
```

## Pipeline đầy đủ (giảm domain gap SOCOFing ↔ SEN0348)

```
PC:    python -m scripts.run_pipeline           # Train với grid-noise aug + partial crop
       → fingerprint_model.tflite + .h5

Jetson:
       1) python3 collect_sensor_data.py --person 001 --samples 10
       2) python3 collect_sensor_data.py --person 002 --samples 10
       ...                                       (lặp với từng người dùng)
       3) python3 finetune_sensor.py --epochs 20 # Fine-tune với ảnh SEN0348 thật
       → fingerprint_model.tflite mới (đã quen với SEN0348)
       4) python3 main.py                        # Enroll + Verify + Identify
```

## Các file cần có trên Jetson Nano

```
jetson/
├── DFRobot_ID809.py           ← Copy từ python/ (driver SEN0348)
├── config.py                  # Cấu hình (port, threshold, CLAHE/denoise...)
├── inference.py               # Engine TFLite + CLAHE + FFT denoise
├── fingerprint_db.py          # Database vân tay (lưu file .npy)
├── main.py                    # Ứng dụng chính (Enroll/Verify/Identify)
├── collect_sensor_data.py     # Thu thập ảnh SEN0348 → BMP dataset
├── finetune_sensor.py         # Fine-tune model với ảnh SEN0348 thật
├── requirements.txt           # Dependencies
├── fingerprint_model.tflite   ← Copy từ PC (hoặc do finetune sinh ra)
├── sensor_dataset/            # Ảnh SEN0348 thu được (do collect tạo)
└── enrolled_fingerprints/     # Database người dùng (do main.py tạo)
```

**Lưu ý:** Cần copy `DFRobot_ID809.py` từ thư mục `python/` vào `jetson/`.

---

## Bước 1: Train model trên Windows

```bash
cd Tiny-Fingerprint_Recognition
python scripts/run_pipeline.py
```

Kết quả:
- `result/fingerprint_siamese_model.h5` — Model Keras
- `fingerprint_model.tflite` — Model TFLite (file cần copy)

Hoặc dùng script export riêng:
```bash
python export_for_jetson.py
```

## Bước 2: Chuẩn bị Jetson Nano

### 2.1. Cài TFLite Runtime

```bash
pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

### 2.2. Cài dependencies

```bash
sudo apt-get install -y python3-opencv python3-numpy
pip3 install pyserial
```

### 2.3. Cấu hình UART (xem chi tiết trong python/README.md)

```bash
sudo systemctl stop nvgetty
sudo systemctl disable nvgetty
sudo usermod -aG dialout $USER
sudo reboot
```

## Bước 3: Copy files sang Jetson

Cần copy **3 nhóm file** sang Jetson Nano (USB, thẻ nhớ, hoặc scp nếu có mạng):

| Từ PC (workspace) | Đến Jetson Nano |
|---|---|
| Toàn bộ thư mục `jetson/` | `~/jetson/` |
| `python/DFRobot_ID809.py` | `~/jetson/DFRobot_ID809.py` |
| `fingerprint_model.tflite` (ở root project) | `~/jetson/fingerprint_model.tflite` |
| `result/fingerprint_siamese_model.h5` (chỉ khi cần fine-tune) | `~/jetson/fingerprint_siamese_model.h5` |

### Cách 1: Dùng USB / thẻ nhớ
Copy 3 nhóm trên vào USB → cắm vào Jetson → copy vào `~/jetson/`.

### Cách 2: Dùng scp (nếu có mạng)
```bash
scp -r jetson/ <user>@<jetson-ip>:/home/<user>/
scp python/DFRobot_ID809.py <user>@<jetson-ip>:/home/<user>/jetson/
scp fingerprint_model.tflite <user>@<jetson-ip>:/home/<user>/jetson/
scp result/fingerprint_siamese_model.h5 <user>@<jetson-ip>:/home/<user>/jetson/
```

### Cấu trúc cuối cùng trên Jetson
```
~/jetson/
├── DFRobot_ID809.py
├── config.py
├── inference.py
├── fingerprint_db.py
├── main.py
├── collect_sensor_data.py
├── finetune_sensor.py
├── requirements.txt
├── fingerprint_model.tflite           ← bắt buộc
└── fingerprint_siamese_model.h5       ← chỉ cần khi fine-tune
```

> **KHÔNG cần** copy: thư mục `Real/`, `Altered/` (dataset SOCOFing chỉ dùng để train trên PC), `fingerprint_model_data.h` (chỉ cho MCU như STM32/ESP32).

## Bước 4: (Khuyến nghị) Thu thập + Fine-tune với SEN0348

Đây là bước **quan trọng nhất** để giảm domain gap giữa SOCOFing (full fingerprint
sạch) và ảnh thật từ SEN0348 (1/3 ngón tay + nhiễu lưới caro).

```bash
cd jetson/

# 4.1. Thu thập 10 ảnh từ mỗi người dùng (lặp với từng người)
python3 collect_sensor_data.py --person 001 --samples 10
python3 collect_sensor_data.py --person 002 --samples 10
# ...

# 4.2. Fine-tune (cần TensorFlow trên Jetson — xem requirements.txt)
python3 finetune_sensor.py --epochs 20 --lr 1e-4

# Output: jetson/fingerprint_model.tflite (đã thay thế model cũ)
```

## Bước 5: Chạy ứng dụng

```bash
cd jetson/
python3 main.py
```

Tuỳ chọn:
```bash
python3 main.py --port /dev/ttyUSB0          # Đổi port
python3 main.py --threshold 0.6              # Đổi ngưỡng (mặc định 0.5)
python3 main.py --model other_model.tflite   # Đổi model
```

---

## Chức năng main.py

| # | Chức năng | Mô tả |
|---|-----------|-------|
| 1 | **Enroll** | Quét 3 lần → so sánh nhất quán → lưu database |
| 2 | **Verify (1:1)** | So khớp với 1 người dùng cụ thể |
| 3 | **Identify (1:N)** | Tìm trong toàn bộ database |
| 4 | **Realtime** | Nhận diện liên tục (loop cho access control) |
| 5 | **List Users** | Liệt kê người đã đăng ký |
| 6 | **Delete** | Xóa vân tay khỏi database |

## Lưu ý

- SEN0348 xuất ảnh **80x80**, model train **90x90** → tự động resize
- **Tiền xử lý ảnh SEN0348**: CLAHE + FFT band-stop denoise (bật/tắt trong `config.py`)
- LED trên SEN0348: xanh dương = chờ, xanh lá = thành công, đỏ = thất bại
- Ngưỡng mặc định **0.5**, điều chỉnh trong `config.py` hoặc `--threshold`
- Inference TFLite: ~5-15ms / cặp ảnh trên Jetson Nano CPU

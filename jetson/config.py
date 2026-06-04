"""
Cấu hình cho hệ thống nhận diện vân tay trên Jetson Nano B01.
Sử dụng: DFRobot_ID809 (SEN0348) + Siamese Network TFLite.
"""
import os

# === Đường dẫn ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fingerprint_model.tflite")
DB_DIR = os.path.join(BASE_DIR, "enrolled_fingerprints")

# === Model Siamese Network ===
INPUT_SIZE = (90, 90)           # Kích thước đầu vào model (resize từ 80x80 của SEN0348)
MATCH_THRESHOLD = 0.65          # Ngưỡng để xác định khớp (>0.65 = match)
                                # Tăng từ 0.5 → 0.65 sau khi fine-tune để giảm false accept
                                # Chạy scripts/evaluate_model.py để tìm ngưỡng tối ưu

# === Tiền xử lý ảnh SEN0348 (giảm domain gap với SOCOFing) ===
USE_CLAHE = True                # CLAHE cân bằng contrast cho ảnh capacitive
USE_GRID_DENOISE = True         # FFT band-stop khử nhiễu lưới caro
CLAHE_CLIP = 2.0                # Clip limit cho CLAHE
CLAHE_GRID = (4, 4)             # Tile grid size cho CLAHE

# === Cache ảnh enrolled ===
PREPROCESS_ENROLLED = True      # True: preprocess ảnh enrolled 1 lần khi load cache
                                # → nhanh hơn khi nhận diện (không cần preprocess lại)
                                # False: preprocess mỗi lần so sánh (chậm hơn nhưng linh hoạt)

# === Cảm biến SEN0348 (DFRobot ID809) ===
SENSOR_UART_PORT = "/dev/ttyTHS1"   # UART port trên Jetson Nano (GPIO 8,10)
SENSOR_BAUD_RATE = 115200           # Baudrate mặc định của SEN0348
SENSOR_IMG_SIZE = (80, 80)          # SEN0348 xuất ảnh 80x80 (quarter image)

# === Giới hạn ===
MAX_ENROLLED = 80               # Tối đa 80 (giới hạn bộ nhớ SEN0348 capacity=80)
ENROLL_SAMPLES = 3              # Số lần quét khi đăng ký 1 vân tay

# === Kiểm tra chất lượng ảnh ===
MIN_IMAGE_VARIANCE = 200        # Ảnh có variance thấp hơn = quá mờ/đen → bỏ qua


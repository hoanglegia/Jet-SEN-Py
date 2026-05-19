"""
Engine suy luận (inference) dùng TFLite trên Jetson Nano B01.
Tải model Siamese Network (.tflite) đã train trên Windows
→ so sánh 2 ảnh vân tay → trả về điểm tương đồng.

Cải tiến để giảm domain gap SOCOFing ↔ SEN0348:
  - CLAHE để chuẩn hóa contrast ảnh capacitive
  - FFT band-stop để khử nhiễu lưới caro của cảm biến
  - Map input theo TÊN tensor (an toàn khi TFLite đảo thứ tự input)
"""
import numpy as np
import cv2
import time

try:
    from tflite_runtime.interpreter import Interpreter as tflite_Interpreter
    print("[Engine] Sử dụng tflite_runtime")
except ImportError:
    from tensorflow.lite import Interpreter as tflite_Interpreter
    print("[Engine] Sử dụng tensorflow.lite")

from config import (
    MODEL_PATH, INPUT_SIZE, MATCH_THRESHOLD,
    USE_CLAHE, USE_GRID_DENOISE, CLAHE_CLIP, CLAHE_GRID,
)


def remove_grid_noise(img, strength=0.85, peak_percentile=98.0):
    """
    Khử nhiễu lưới caro đặc trưng của cảm biến điện dung SEN0348
    bằng cách triệt tiêu các đỉnh tần số cao trong miền FFT.

    Args:
        img: ảnh grayscale uint8 (H, W)
        strength: 0..1 — càng cao càng khử mạnh
        peak_percentile: chỉ triệt tiêu các peak vượt percentile này
    Returns:
        ảnh uint8 đã khử nhiễu
    """
    h, w = img.shape[:2]
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Vùng tần số thấp = chứa thông tin vân tay → giữ nguyên
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * 0.18)
    Y, X = np.ogrid[:h, :w]
    low_pass = (Y - cy) ** 2 + (X - cx) ** 2 <= radius * radius

    high_mag = magnitude.copy()
    high_mag[low_pass] = 0
    if high_mag.max() > 0:
        thresh = np.percentile(high_mag[~low_pass], peak_percentile)
        peak_mask = high_mag > thresh
        fshift[peak_mask] *= (1.0 - strength)

    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    return np.clip(img_back, 0, 255).astype(np.uint8)


class FingerprintEngine:
    """
    Engine nhận diện vân tay sử dụng Siamese Network (TFLite).
    So sánh 2 ảnh vân tay → trả về điểm tương đồng (0.0 ~ 1.0).
    """

    def __init__(self, model_path=None, threshold=None,
                 use_clahe=None, use_grid_denoise=None):
        model_path = model_path or MODEL_PATH
        self.threshold = MATCH_THRESHOLD if threshold is None else threshold
        self.use_clahe = USE_CLAHE if use_clahe is None else use_clahe
        self.use_grid_denoise = USE_GRID_DENOISE if use_grid_denoise is None else use_grid_denoise

        self.interpreter = tflite_Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self._clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

        print(f"[Engine] Model loaded: {model_path}")
        print(f"[Engine] Threshold={self.threshold} | CLAHE={self.use_clahe} | GridDenoise={self.use_grid_denoise}")
        for i, inp in enumerate(self.input_details):
            print(f"  Input[{i}]: name={inp['name']}, shape={inp['shape']}, dtype={inp['dtype']}")

    # -------------------- Preprocess --------------------
    def preprocess(self, img):
        """
        Tiền xử lý ảnh vân tay từ SEN0348:
          1. Convert về grayscale 2D uint8
          2. Khử nhiễu lưới (FFT band-stop)
          3. CLAHE để cân bằng contrast
          4. Resize → 90x90, normalize [0,1], reshape (1,90,90,1)
        """
        if img is None:
            return None

        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        img = img.astype(np.uint8)

        if self.use_grid_denoise:
            img = remove_grid_noise(img)
        if self.use_clahe:
            img = self._clahe.apply(img)

        img = cv2.resize(img, INPUT_SIZE)
        img = img.astype(np.float32) / 255.0
        return img.reshape(1, INPUT_SIZE[0], INPUT_SIZE[1], 1)

    # -------------------- Inference --------------------
    def _set_inputs(self, t1, t2):
        d0, d1 = self.input_details[0], self.input_details[1]
        self.interpreter.set_tensor(d0['index'], t1.astype(d0['dtype']))
        self.interpreter.set_tensor(d1['index'], t2.astype(d1['dtype']))

    def compare(self, img1, img2):
        """So sánh 2 ảnh vân tay (uint8 grayscale). Returns (score, is_match, ms)."""
        t1 = self.preprocess(img1)
        t2 = self.preprocess(img2)
        if t1 is None or t2 is None:
            return 0.0, False, 0.0
        return self.compare_preprocessed(t1, t2)

    def compare_preprocessed(self, t1, t2):
        """So sánh 2 tensor đã preprocess sẵn (1,90,90,1, float32)."""
        self._set_inputs(t1, t2)
        start = time.time()
        self.interpreter.invoke()
        ms = (time.time() - start) * 1000
        score = float(self.interpreter.get_tensor(self.output_details[0]['index'])[0][0])
        return score, score > self.threshold, ms

    def compare_from_files(self, path1, path2):
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        return self.compare(img1, img2)

    def find_best_match(self, query_img, enrolled_db, threshold=None):
        """
        Tìm vân tay khớp nhất trong database. enrolled_db có thể chứa:
          - ảnh raw uint8 (sẽ tự preprocess)
          - tensor float32 shape (1,90,90,1) — đã preprocess sẵn (nhanh hơn)
        Returns: (best_user_id or None, best_score, total_ms)
        """
        thr = self.threshold if threshold is None else threshold
        q = self.preprocess(query_img)
        if q is None:
            return None, 0.0, 0.0

        best_user, best_score, total_ms = None, 0.0, 0.0
        for user_id, samples in enrolled_db.items():
            for ref in samples:
                if isinstance(ref, np.ndarray) and ref.ndim == 4 and ref.dtype == np.float32:
                    ref_t = ref
                else:
                    ref_t = self.preprocess(ref)
                    if ref_t is None:
                        continue
                score, _, ms = self.compare_preprocessed(q, ref_t)
                total_ms += ms
                if score > best_score:
                    best_score, best_user = score, user_id

        if best_score < thr:
            best_user = None
        return best_user, best_score, total_ms

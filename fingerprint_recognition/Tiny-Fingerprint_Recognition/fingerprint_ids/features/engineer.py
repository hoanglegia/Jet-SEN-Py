import numpy as np
import os
import cv2
from imgaug import augmenters as iaa


# ============================================================
# Augmenter mô phỏng nhiễu lưới caro của cảm biến điện dung
# (SEN0348). Dùng làm imgaug Lambda layer trong Sequential.
# ============================================================
def _add_grid_noise_batch(images, random_state, parents, hooks):
    """imgaug Lambda: cộng thêm nhiễu lưới sin 2D vào batch ảnh."""
    out = []
    for img in images:
        squeeze = False
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
            squeeze = True
        h, w = img.shape
        # Tần số ngẫu nhiên (số chu kỳ trên ảnh) — mô phỏng pitch của lưới
        fy = random_state.uniform(6, 14)
        fx = random_state.uniform(6, 14)
        phase_y = random_state.uniform(0, 2 * np.pi)
        phase_x = random_state.uniform(0, 2 * np.pi)
        amplitude = random_state.uniform(8, 25)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        grid = (np.sin(2 * np.pi * fy * yy / h + phase_y) *
                np.sin(2 * np.pi * fx * xx / w + phase_x))
        noisy = img.astype(np.float32) + amplitude * grid
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        if squeeze:
            noisy = noisy.reshape(h, w, 1)
        out.append(noisy)
    return out


def _keypoints_passthrough(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


def add_grid_noise_aug(p=0.5):
    """Lambda augmenter có thể chèn vào iaa.Sequential."""
    return iaa.Sometimes(p, iaa.Lambda(
        func_images=_add_grid_noise_batch,
        func_keypoints=_keypoints_passthrough,
    ))


def extract_label(img_path, is_altered=False):
    filename = os.path.basename(img_path).split('.')[0]
    subject_id, etc = filename.split('__')
    parts = etc.split('_')
    gender = 0 if parts[0] == 'M' else 1
    lr = 0 if parts[1] == 'Left' else 1

    finger_map = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'little': 4}
    finger = finger_map.get(parts[2], 0)
    return np.array([int(subject_id), gender, lr, finger], dtype=np.uint16)


def get_augmentor():
    return iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 0.5)),
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-30, 30),
            order=[0, 1],
            cval=255
        )
    ], random_order=True)


def get_partial_augmentor():
    """
    Augmentor mạnh hơn, mô phỏng ảnh partial từ SEN0348:
      - blur + Gaussian noise + affine + contrast (giả lập capacitive)
      - GRID NOISE sin 2D — mô phỏng nhiễu lưới caro đặc trưng
    """
    return iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, 1.0)),
        iaa.AdditiveGaussianNoise(scale=(0, 15)),
        add_grid_noise_aug(p=0.6),
        iaa.Affine(
            scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            rotate=(-45, 45),
            order=[0, 1],
            cval=255
        ),
        iaa.LinearContrast((0.7, 1.3)),
    ], random_order=False)


def random_partial_crop(img, crop_ratio=(0.3, 0.5)):
    """
    Crop ngẫu nhiên vùng của ảnh để mô phỏng partial fingerprint
    từ cảm biến SEN0348 (chỉ quét được ~1/3 ngón tay).
    """
    squeeze = False
    if img.ndim == 3:
        img = img[:, :, 0]
        squeeze = True

    h, w = img.shape
    ratio = np.random.uniform(crop_ratio[0], crop_ratio[1])
    crop_h = int(h * ratio)
    crop_w = int(w * ratio)

    max_y = h - crop_h
    max_x = w - crop_w
    y = np.random.randint(0, max(1, max_y))
    x = np.random.randint(0, max(1, max_x))

    cropped = img[y:y+crop_h, x:x+crop_w]
    resized = cv2.resize(cropped, (w, h))
    if squeeze:
        resized = resized.reshape(h, w, 1)
    return resized


def preprocess_image(img, target_size=(90, 90)):
    img = cv2.resize(img, target_size)
    return img.reshape(target_size[0], target_size[1], 1)


# ------------------------------------------------------------
# Tiền xử lý ảnh SEN0348 cho inference / fine-tune
# (đối xứng với jetson/inference.py để training/inference cùng pipeline)
# ------------------------------------------------------------
def remove_grid_noise_fft(img, strength=0.85, peak_percentile=98.0):
    """Khử nhiễu lưới caro bằng FFT band-stop."""
    h, w = img.shape[:2]
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

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


def preprocess_sensor_image(img, target_size=(90, 90),
                            denoise_grid=True, apply_clahe=True):
    """Tiền xử lý ảnh từ SEN0348 cho training / fine-tune / verify."""
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    img = img.astype(np.uint8)

    if denoise_grid:
        img = remove_grid_noise_fft(img)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img = clahe.apply(img)

    img = cv2.resize(img, target_size)
    return img.reshape(target_size[0], target_size[1], 1)

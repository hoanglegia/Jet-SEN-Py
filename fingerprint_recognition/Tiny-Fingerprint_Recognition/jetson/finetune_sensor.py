"""
Fine-tune model Siamese đã train trên SOCOFing với ảnh thật từ SEN0348.

Quy trình:
  1. Load model .h5 đã train (kết quả từ scripts/run_pipeline.py)
  2. Load ảnh BMP đã thu bằng collect_sensor_data.py
  3. Áp preprocess SEN0348 (FFT denoise + CLAHE) — ĐỒNG BỘ với inference
  4. Fine-tune với SensorPairGenerator (pair positive/negative)
  5. Export TFLite mới → copy sang Jetson dùng

Cách dùng:
    python3 finetune_sensor.py
    python3 finetune_sensor.py --data sensor_dataset --epochs 30 --lr 5e-5
    python3 finetune_sensor.py --no-freeze   # mở khóa toàn bộ layers

Yêu cầu:
    pip install tensorflow opencv-python imgaug scikit-learn
"""
import argparse
import os
import sys
import numpy as np

# Cho phép import package fingerprint_ids khi chạy từ jetson/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from fingerprint_ids.data.loader import FingerprintLoader, SensorPairGenerator
from fingerprint_ids.features.engineer import (
    preprocess_sensor_image, get_partial_augmentor,
)
from fingerprint_ids.models.classifiers import load_and_finetune
from fingerprint_ids.deploy.export import export_to_tflite


def preprocess_dataset(images):
    """Áp preprocess SEN0348 cho toàn bộ dataset trước khi fine-tune."""
    out = np.empty((len(images), 90, 90, 1), dtype=np.uint8)
    for i, img in enumerate(images):
        # img từ load_sensor_dataset có shape (90,90,1) sau resize
        # nhưng chưa FFT/CLAHE → gọi lại trên ảnh raw 80x80 cũng OK
        if img.ndim == 3:
            img2d = img[:, :, 0]
        else:
            img2d = img
        out[i] = preprocess_sensor_image(img2d, target_size=(90, 90))
    return out


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Siamese model với ảnh SEN0348")
    parser.add_argument('--model', default=os.path.join(PROJECT_ROOT, 'result', 'fingerprint_siamese_model.h5'),
                        help='Model Keras .h5 đã train trên SOCOFing')
    parser.add_argument('--data', default=os.path.join(THIS_DIR, 'sensor_dataset'),
                        help='Thư mục chứa person_XX/*.bmp')
    parser.add_argument('--out-h5', default=os.path.join(THIS_DIR, 'fingerprint_finetuned.h5'))
    parser.add_argument('--out-tflite', default=os.path.join(THIS_DIR, 'fingerprint_model.tflite'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--no-freeze', action='store_true', help='Không freeze FeatureExtractor')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Không tìm thấy model: {args.model}")
        print("Hãy chạy training trên PC trước: python -m scripts.run_pipeline")
        sys.exit(1)

    if not os.path.isdir(args.data):
        print(f"ERROR: Không tìm thấy thư mục dataset: {args.data}")
        print("Hãy thu thập ảnh trước: python3 collect_sensor_data.py --person 001 --samples 10")
        sys.exit(1)

    # 1. Load ảnh sensor
    print(f"Loading sensor data từ: {args.data}")
    images, labels = FingerprintLoader.load_sensor_dataset(args.data, target_size=(90, 90))
    if len(images) == 0:
        print("ERROR: Không có ảnh nào trong dataset.")
        sys.exit(1)
    print(f"Tổng: {len(images)} ảnh, {len(np.unique(labels))} người dùng")

    # 2. Preprocess (FFT denoise + CLAHE) đồng bộ với inference
    print("Preprocessing (FFT denoise + CLAHE)...")
    images = preprocess_dataset(images)

    # 3. Tạo generator
    augmentor = get_partial_augmentor()
    train_gen = SensorPairGenerator(images, labels, batch_size=args.batch, augmentor=augmentor)
    print(f"Steps per epoch: {len(train_gen)}")

    # 4. Load model + fine-tune
    print(f"Loading pretrained model: {args.model}")
    model = load_and_finetune(args.model, learning_rate=args.lr, freeze_feature=not args.no_freeze)
    model.summary()

    print(f"\nFine-tuning {args.epochs} epochs...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.out_h5, save_best_only=False, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6),
    ]
    model.fit(train_gen, epochs=args.epochs, callbacks=callbacks)

    # 5. Save + export TFLite
    model.save(args.out_h5)
    print(f"Đã lưu: {args.out_h5}")

    print(f"Exporting TFLite...")
    export_to_tflite(model, args.out_tflite)
    size_kb = os.path.getsize(args.out_tflite) / 1024
    print(f"Đã export: {args.out_tflite} ({size_kb:.1f} KB)")
    print("\nHoàn tất! Restart main.py để dùng model mới.")


if __name__ == "__main__":
    main()

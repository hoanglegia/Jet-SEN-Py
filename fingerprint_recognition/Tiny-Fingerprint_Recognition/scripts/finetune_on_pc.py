"""
Fine-tune model Siamese đã train trên SOCOFing bằng ảnh thật từ SEN0348.
Chạy trên WINDOWS (tận dụng GPU mạnh hơn Jetson Nano).

Quy trình:
  1. Thu thập ảnh SEN0348 trên Jetson (collect_sensor_data.py)
  2. Copy folder sensor_dataset/ sang Windows (SCP hoặc USB)
  3. Chạy script này trên Windows:
       python scripts/finetune_on_pc.py
       python scripts/finetune_on_pc.py --data sensor_dataset --epochs 30 --lr 5e-5
       python scripts/finetune_on_pc.py --no-freeze  # mở toàn bộ layers

  Kết quả:
    - result/fingerprint_finetuned.h5     → model đã fine-tune
    - result/fingerprint_finetuned.tflite → TFLite quantized, sẵn sàng deploy Jetson

Yêu cầu:
    pip install tensorflow opencv-python imgaug scikit-learn
"""
import argparse
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from fingerprint_ids.data.loader import FingerprintLoader, SensorPairGenerator
from fingerprint_ids.features.engineer import (
    preprocess_sensor_image, get_partial_augmentor,
)
from fingerprint_ids.models.classifiers import load_and_finetune
from fingerprint_ids.deploy.export import export_to_tflite


def preprocess_dataset(images, target_size=(90, 90)):
    """
    Áp preprocessing SEN0348 (FFT denoise + CLAHE) cho toàn bộ dataset.
    Pipeline này ĐỒNG BỘ với jetson/inference.py để đảm bảo
    training và inference xử lý ảnh giống hệt nhau.
    """
    out = np.empty((len(images), target_size[0], target_size[1], 1), dtype=np.uint8)
    for i, img in enumerate(images):
        if img.ndim == 3:
            img2d = img[:, :, 0]
        else:
            img2d = img
        out[i] = preprocess_sensor_image(img2d, target_size=target_size)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Siamese model trên Windows với ảnh SEN0348"
    )
    parser.add_argument(
        '--model',
        default=os.path.join(_PROJECT_ROOT, 'result', 'fingerprint_siamese_model.h5'),
        help='Model Keras .h5 đã train trên SOCOFing'
    )
    parser.add_argument(
        '--data',
        default=os.path.join(_PROJECT_ROOT, 'sensor_dataset'),
        help='Thư mục chứa ảnh SEN0348 (person_XXX/sample_N.bmp)'
    )
    parser.add_argument(
        '--out-h5',
        default=os.path.join(_PROJECT_ROOT, 'result', 'fingerprint_finetuned.h5'),
        help='Đường dẫn output model .h5'
    )
    parser.add_argument(
        '--out-tflite',
        default=os.path.join(_PROJECT_ROOT, 'result', 'fingerprint_finetuned.tflite'),
        help='Đường dẫn output .tflite'
    )
    parser.add_argument('--epochs', type=int, default=30,
                        help='Số epochs fine-tune (mặc định 30)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (mặc định 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (mặc định 1e-4)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Tỉ lệ validation split (mặc định 0.2)')
    parser.add_argument('--no-freeze', action='store_true',
                        help='Không freeze FeatureExtractor (train toàn bộ)')
    args = parser.parse_args()

    # === 1. Kiểm tra inputs ===
    if not os.path.exists(args.model):
        print(f"ERROR: Không tìm thấy model: {args.model}")
        print("Hãy chạy training trước: python scripts/run_pipeline.py")
        sys.exit(1)

    if not os.path.isdir(args.data):
        print(f"ERROR: Không tìm thấy thư mục sensor_dataset: {args.data}")
        print()
        print("=== HƯỚNG DẪN THU THẬP ẢNH SEN0348 ===")
        print("1. Trên Jetson Nano, chạy:")
        print("     python3 collect_sensor_data.py --person 001 --samples 20")
        print("     python3 collect_sensor_data.py --person 002 --samples 20")
        print("     python3 collect_sensor_data.py --person 003 --samples 20")
        print("2. Copy folder sensor_dataset/ sang Windows:")
        print("     scp -r jetson@<JETSON_IP>:~/jetson/sensor_dataset ./sensor_dataset")
        print("3. Chạy lại script này.")
        sys.exit(1)

    # === 2. Load ảnh SEN0348 ===
    print(f"\n{'='*60}")
    print(f"  FINE-TUNE SIAMESE MODEL VỚI ẢNH SEN0348")
    print(f"{'='*60}")
    print(f"  Model gốc   : {args.model}")
    print(f"  Dataset      : {args.data}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Freeze       : {'NO (all layers trainable)' if args.no_freeze else 'YES (FeatureExtractor frozen)'}")
    print(f"{'='*60}\n")

    images, labels = FingerprintLoader.load_sensor_dataset(args.data, target_size=(90, 90))
    if len(images) == 0:
        print("ERROR: Không có ảnh nào trong dataset.")
        sys.exit(1)

    unique_labels = np.unique(labels)
    print(f"Tổng: {len(images)} ảnh, {len(unique_labels)} người dùng")
    for lbl in unique_labels:
        count = np.sum(labels == lbl)
        print(f"  person_{lbl:03d}: {count} ảnh")

    if len(unique_labels) < 2:
        print("\nWARNING: Chỉ có 1 người dùng → model không thể học phân biệt.")
        print("Cần ít nhất 2 người (khuyến nghị ≥3 người).")
        sys.exit(1)

    # === 3. Preprocess (FFT denoise + CLAHE) ===
    print("\nPreprocessing (FFT denoise + CLAHE)...")
    images = preprocess_dataset(images)

    # === 4. Train/Val split ===
    # Split theo stratified labels để đảm bảo mỗi person có mặt trong cả train và val
    if args.val_split > 0 and len(images) >= 10:
        # Kiểm tra mỗi label có đủ ảnh để split không
        min_samples = min(np.sum(labels == lbl) for lbl in unique_labels)
        if min_samples >= 4:
            train_imgs, val_imgs, train_labels, val_labels = train_test_split(
                images, labels, test_size=args.val_split,
                stratify=labels, random_state=42
            )
            print(f"Split: {len(train_imgs)} train / {len(val_imgs)} val")
        else:
            print(f"WARNING: Một person chỉ có {min_samples} ảnh, không đủ để split.")
            print("Dùng toàn bộ cho training (không có validation).")
            train_imgs, train_labels = images, labels
            val_imgs, val_labels = None, None
    else:
        train_imgs, train_labels = images, labels
        val_imgs, val_labels = None, None

    # === 5. Tạo generators ===
    augmentor = get_partial_augmentor()
    train_gen = SensorPairGenerator(
        train_imgs, train_labels,
        batch_size=args.batch, augmentor=augmentor
    )

    val_gen = None
    if val_imgs is not None and len(val_imgs) > 0:
        val_gen = SensorPairGenerator(
            val_imgs, val_labels,
            batch_size=args.batch, augmentor=None
        )

    print(f"Train steps/epoch: {len(train_gen)}")
    if val_gen:
        print(f"Val steps/epoch:   {len(val_gen)}")

    # === 6. Load model + fine-tune ===
    print(f"\nLoading pretrained model: {args.model}")
    model = load_and_finetune(
        args.model,
        learning_rate=args.lr,
        freeze_feature=not args.no_freeze
    )
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(args.out_h5), exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.out_h5,
            monitor='val_loss' if val_gen else 'loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if val_gen else 'loss',
            factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_gen else 'loss',
            patience=10, restore_best_weights=True, verbose=1
        ),
    ]

    # === 7. Training ===
    print(f"\n{'='*40}")
    print(f"  Bắt đầu Fine-tuning {args.epochs} epochs...")
    print(f"{'='*40}\n")

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # === 8. Save model ===
    model.save(args.out_h5)
    print(f"\nĐã lưu model: {args.out_h5}")

    # === 9. Export TFLite ===
    print("Exporting TFLite (quantized)...")
    export_to_tflite(model, args.out_tflite)
    size_kb = os.path.getsize(args.out_tflite) / 1024
    print(f"Đã export: {args.out_tflite} ({size_kb:.1f} KB)")

    # === 10. Tóm tắt ===
    print(f"\n{'='*60}")
    print(f"  HOÀN TẤT FINE-TUNING!")
    print(f"{'='*60}")

    # In kết quả training
    final_loss = history.history['loss'][-1]
    final_acc = history.history['acc'][-1]
    print(f"  Train Loss: {final_loss:.4f}")
    print(f"  Train Acc:  {final_acc:.4f}")
    if val_gen and 'val_loss' in history.history:
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_acc'][-1]
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")

    print(f"\n  Files đã tạo:")
    print(f"    Model H5:   {os.path.abspath(args.out_h5)}")
    print(f"    TFLite:     {os.path.abspath(args.out_tflite)}")

    print(f"\n  === BƯỚC TIẾP THEO ===")
    print(f"  1. Đánh giá model:")
    print(f"       python scripts/evaluate_model.py --model {args.out_h5} --data {args.data}")
    print(f"  2. Copy TFLite sang Jetson Nano:")
    print(f"       scp {os.path.abspath(args.out_tflite)} jetson@<IP>:~/jetson/fingerprint_model.tflite")
    print(f"  3. Chạy ứng dụng trên Jetson:")
    print(f"       python3 main.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

import argparse
import os
import sys

# Cho phép chạy bằng cả 2 cách:
#   python -m scripts.run_pipeline
#   python scripts/run_pipeline.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from fingerprint_ids.data.loader import (
    FingerprintLoader, FingerprintGenerator, SensorPairGenerator,
)
from fingerprint_ids.features.engineer import (
    get_augmentor, get_partial_augmentor, preprocess_sensor_image,
)
from fingerprint_ids.models.classifiers import build_siamese_model, load_and_finetune
from fingerprint_ids.deploy.export import export_to_tflite
from fingerprint_ids.deploy.c_export import tflite_to_c_array
from sklearn.model_selection import train_test_split

PATH_REAL = 'Real/*.BMP'
PATH_EASY = 'Altered/Altered-Easy/*.BMP'
PATH_MEDIUM = 'Altered/Altered-Medium/*.BMP'
PATH_HARD = 'Altered/Altered-Hard/*.BMP'
SENSOR_DATASET_DIR = 'sensor_dataset'


def _preprocess_sensor_dataset(images, target_size=(90, 90)):
    """Áp preprocessing SEN0348 (FFT denoise + CLAHE) cho toàn bộ dataset."""
    out = np.empty((len(images), target_size[0], target_size[1], 1), dtype=np.uint8)
    for i, img in enumerate(images):
        img2d = img[:, :, 0] if img.ndim == 3 else img
        out[i] = preprocess_sensor_image(img2d, target_size=target_size)
    return out


def run(epochs=20, partial_crop=True, sensor_aug=True, finetune_epochs=30):
    """
    Pipeline training trên PC gồm 2 phase:

    Phase 1: Train Siamese trên SOCOFing (pretrain)
    Phase 2: Fine-tune trên ảnh SEN0348 nếu có (auto-detect sensor_dataset/)

    Args:
        epochs: số epoch cho Phase 1 (SOCOFing)
        partial_crop: bật crop ngẫu nhiên 30-50% mô phỏng SEN0348
        sensor_aug: dùng augmentor mạnh (blur+noise+grid) cho domain SEN0348
                    True = khuyến nghị khi deploy lên Jetson + SEN0348
                    False = augment nhẹ (giống notebook gốc)
        finetune_epochs: số epoch cho Phase 2 (fine-tune SEN0348)
    """
    # Đảm bảo cwd = project root để các path "Real/*.BMP" hoạt động đúng
    os.chdir(_PROJECT_ROOT)

    # ============================================================
    # PHASE 1: Train trên SOCOFing
    # ============================================================
    print("\n" + "=" * 60)
    print("  PHASE 1: TRAIN SIAMESE TRÊN SOCOFing")
    print("=" * 60 + "\n")

    # 1. Load dataset
    x_real, y_real = FingerprintLoader.load_raw_dataset(PATH_REAL)
    label_dict = FingerprintLoader.create_label_dict(y_real)

    print("Loading Altered-Easy...")
    x_easy, y_easy = FingerprintLoader.load_raw_dataset(PATH_EASY)
    print("Loading Altered-Medium...")
    x_med, y_med = FingerprintLoader.load_raw_dataset(PATH_MEDIUM)
    print("Loading Altered-Hard...")
    x_hard, y_hard = FingerprintLoader.load_raw_dataset(PATH_HARD)

    # 2. Split train/val
    x_altered = np.concatenate([x_easy, x_med, x_hard], axis=0)
    y_altered = np.concatenate([y_easy, y_med, y_hard], axis=0)
    print(f"Total altered images: {len(x_altered)}")
    x_train, x_val, y_train, y_val = train_test_split(x_altered, y_altered, test_size=0.1)

    # 3. Generators (augmentor mạnh + partial crop để giảm domain gap với SEN0348)
    augmentor = get_partial_augmentor() if sensor_aug else get_augmentor()
    train_gen = FingerprintGenerator(
        x_train, y_train, x_real, label_dict,
        augmentor=augmentor, partial_crop=partial_crop
    )
    val_gen = FingerprintGenerator(x_val, y_val, x_real, label_dict, shuffle_data=False)

    # 4. Build & train với callbacks
    model = build_siamese_model()

    os.makedirs("result", exist_ok=True)
    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(
            "result/fingerprint_siamese_model.h5",
            monitor='val_loss', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks_phase1)

    # 5. Save
    model.save("result/fingerprint_siamese_model.h5")
    print("Phase 1 model saved: result/fingerprint_siamese_model.h5")

    # 6. Export TFLite + C header (Phase 1 — baseline)
    export_to_tflite(model, "fingerprint_model.tflite")
    tflite_to_c_array("fingerprint_model.tflite", "fingerprint_model_data.h")

    # ============================================================
    # PHASE 2: Fine-tune trên ảnh SEN0348 (nếu có)
    # ============================================================
    sensor_dir = os.path.join(_PROJECT_ROOT, SENSOR_DATASET_DIR)
    if os.path.isdir(sensor_dir):
        print("\n" + "=" * 60)
        print("  PHASE 2: FINE-TUNE TRÊN ẢNH SEN0348")
        print("=" * 60 + "\n")

        sensor_imgs, sensor_labels = FingerprintLoader.load_sensor_dataset(
            sensor_dir, target_size=(90, 90)
        )

        if len(sensor_imgs) > 0 and len(np.unique(sensor_labels)) >= 2:
            print(f"Sensor dataset: {len(sensor_imgs)} ảnh, "
                  f"{len(np.unique(sensor_labels))} người")

            # Preprocess (FFT denoise + CLAHE)
            print("Preprocessing sensor images (FFT denoise + CLAHE)...")
            sensor_imgs = _preprocess_sensor_dataset(sensor_imgs)

            # Load model + fine-tune
            ft_model = load_and_finetune(
                "result/fingerprint_siamese_model.h5",
                learning_rate=1e-4,
                freeze_feature=True
            )

            sensor_aug_ft = get_partial_augmentor()
            sensor_gen = SensorPairGenerator(
                sensor_imgs, sensor_labels,
                batch_size=8, augmentor=sensor_aug_ft
            )

            callbacks_phase2 = [
                tf.keras.callbacks.ModelCheckpoint(
                    "result/fingerprint_finetuned.h5",
                    monitor='loss', save_best_only=True, verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss', patience=10,
                    restore_best_weights=True, verbose=1
                ),
            ]

            ft_model.fit(sensor_gen, epochs=finetune_epochs, callbacks=callbacks_phase2)
            ft_model.save("result/fingerprint_finetuned.h5")
            print("Phase 2 model saved: result/fingerprint_finetuned.h5")

            # Export TFLite fine-tuned (đây là model chính để deploy)
            export_to_tflite(ft_model, "result/fingerprint_finetuned.tflite")
            print("Fine-tuned TFLite: result/fingerprint_finetuned.tflite")

            print("\n=== HOÀN TẤT (PHASE 1 + PHASE 2) ===")
            print("File để deploy lên Jetson Nano:")
            print("  → result/fingerprint_finetuned.tflite  (KHUYẾN NGHỊ — đã fine-tune)")
            print("  → fingerprint_model.tflite              (backup — chưa fine-tune)")
        else:
            if len(sensor_imgs) == 0:
                print(f"WARNING: Folder {sensor_dir} tồn tại nhưng không có ảnh.")
            else:
                print(f"WARNING: Chỉ có {len(np.unique(sensor_labels))} người, cần ≥2 người.")
            print("Bỏ qua Phase 2. Dùng model Phase 1 (chưa fine-tune).")
    else:
        print(f"\n[INFO] Không tìm thấy {SENSOR_DATASET_DIR}/ → bỏ qua Phase 2.")

    # ============================================================
    # Hướng dẫn tiếp theo
    # ============================================================
    print("\n" + "=" * 60)
    print("  BƯỚC TIẾP THEO")
    print("=" * 60)

    if os.path.exists("result/fingerprint_finetuned.tflite"):
        tflite_file = "result/fingerprint_finetuned.tflite"
    else:
        tflite_file = "fingerprint_model.tflite"

    if not os.path.isdir(sensor_dir):
        print("\n  *** BẠN CHƯA CÓ ẢNH SEN0348 → CẦN THU THẬP ***\n")
        print("  1. Trên Jetson Nano, thu thập ảnh SEN0348:")
        print("       python3 collect_sensor_data.py --person 001 --samples 20")
        print("       python3 collect_sensor_data.py --person 002 --samples 20")
        print("       python3 collect_sensor_data.py --person 003 --samples 20")
        print("  2. Copy folder sensor_dataset/ sang Windows (cùng cấp với Real/):")
        print("       scp -r jetson@<IP>:~/jetson/sensor_dataset ./sensor_dataset")
        print("  3. Chạy lại pipeline (sẽ tự động fine-tune):")
        print("       python scripts/run_pipeline.py")
    else:
        print(f"\n  1. Đánh giá model trước khi deploy:")
        print(f"       python scripts/evaluate_model.py --data {SENSOR_DATASET_DIR}")
        print(f"  2. Copy TFLite sang Jetson Nano:")
        print(f"       scp {tflite_file} jetson@<IP>:~/jetson/fingerprint_model.tflite")
        print(f"  3. Chạy ứng dụng trên Jetson:")
        print(f"       python3 main.py")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint Siamese Training Pipeline")
    parser.add_argument('--epochs', type=int, default=20,
                        help='Số epochs Phase 1 — SOCOFing (mặc định 20)')
    parser.add_argument('--finetune-epochs', type=int, default=30,
                        help='Số epochs Phase 2 — Fine-tune SEN0348 (mặc định 30)')
    parser.add_argument('--no-partial-crop', action='store_true',
                        help='Tắt random crop (không mô phỏng SEN0348)')
    parser.add_argument('--no-sensor-aug', action='store_true',
                        help='Dùng augmentor nhẹ (giống notebook gốc)')
    args = parser.parse_args()
    run(epochs=args.epochs,
        partial_crop=not args.no_partial_crop,
        sensor_aug=not args.no_sensor_aug,
        finetune_epochs=args.finetune_epochs)

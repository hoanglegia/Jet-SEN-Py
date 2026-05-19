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
from fingerprint_ids.data.loader import FingerprintLoader, FingerprintGenerator
from fingerprint_ids.features.engineer import get_augmentor, get_partial_augmentor
from fingerprint_ids.models.classifiers import build_siamese_model
from fingerprint_ids.deploy.export import export_to_tflite
from fingerprint_ids.deploy.c_export import tflite_to_c_array
from sklearn.model_selection import train_test_split

PATH_REAL = 'Real/*.BMP'
PATH_EASY = 'Altered/Altered-Easy/*.BMP'
PATH_MEDIUM = 'Altered/Altered-Medium/*.BMP'
PATH_HARD = 'Altered/Altered-Hard/*.BMP'


def run(epochs=20, partial_crop=True, sensor_aug=True):
    """
    Pipeline training trên PC.

    Args:
        epochs: số epoch
        partial_crop: bật crop ngẫu nhiên 30-50% mô phỏng SEN0348
        sensor_aug: dùng augmentor mạnh (blur+noise+grid) cho domain SEN0348
                    True = khuyến nghị khi deploy lên Jetson + SEN0348
                    False = augment nhẹ (giống notebook gốc)
    """
    # Đảm bảo cwd = project root để các path "Real/*.BMP" hoạt động đúng
    os.chdir(_PROJECT_ROOT)

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

    # 4. Build & train
    model = build_siamese_model()
    model.fit(train_gen, epochs=epochs, validation_data=val_gen)

    # 5. Save
    os.makedirs("result", exist_ok=True)
    model.save("result/fingerprint_siamese_model.h5")
    print("Model saved: result/fingerprint_siamese_model.h5")

    # 6. Export TFLite + C header
    export_to_tflite(model, "fingerprint_model.tflite")
    tflite_to_c_array("fingerprint_model.tflite", "fingerprint_model_data.h")

    print("\n=== Tiếp theo trên Jetson Nano ===")
    print("1. Copy 'fingerprint_model.tflite' và 'result/fingerprint_siamese_model.h5' sang Jetson Nano (vào jetson/)")
    print("2. Copy 'python/DFRobot_ID809.py' vào jetson/ trên Jetson Nano")
    print("3. Thu thập ảnh SEN0348:")
    print("     python3 jetson/collect_sensor_data.py --person 001 --samples 10")
    print("4. Fine-tune để giảm domain gap (khuyến nghị):")
    print("     python3 jetson/finetune_sensor.py --epochs 20")
    print("5. Chạy ứng dụng:")
    print("     python3 jetson/main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fingerprint Siamese Training Pipeline")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-partial-crop', action='store_true',
                        help='Tắt random crop (không mô phỏng SEN0348)')
    parser.add_argument('--no-sensor-aug', action='store_true',
                        help='Dùng augmentor nhẹ (giống notebook gốc)')
    args = parser.parse_args()
    run(epochs=args.epochs,
        partial_crop=not args.no_partial_crop,
        sensor_aug=not args.no_sensor_aug)

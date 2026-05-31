"""
Train/Fine-tune model Siamese CHUYEN cho anh SEN0348.

Cach dung:
    # Fine-tune tu model SOCOFing:
    python scripts/train_sensor.py --data sensor_dataset

    # Train tu dau (neu fine-tune van kem):
    python scripts/train_sensor.py --data sensor_dataset --from-scratch
"""
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split

from fingerprint_ids.data.loader import FingerprintLoader
from fingerprint_ids.features.engineer import preprocess_sensor_image


# ============================================================
# Preprocessing
# ============================================================
def preprocess_dataset(images, target_size=(90, 90)):
    out = np.empty((len(images), target_size[0], target_size[1], 1), dtype=np.uint8)
    for i, img in enumerate(images):
        img2d = img[:, :, 0] if img.ndim == 3 else img
        out[i] = preprocess_sensor_image(img2d, target_size=target_size)
    return out


# ============================================================
# Custom Generator với nhiều cải tiến
# ============================================================
class SensorTrainGenerator(tf.keras.utils.Sequence):
    """
    Generator cải tiến cho training trên sensor data:
      - steps_per_epoch cấu hình được (mặc định 200)
      - Augment CẢ 2 ảnh trong pair
      - Hard negative: 30% negative pairs lấy từ person gần nhất
    """
    def __init__(self, images, labels, batch_size=16,
                 steps_per_epoch=200, augment=True):
        self.images = images.astype(np.float32) / 255.0  # Pre-normalize
        self.labels = labels
        self.batch_size = batch_size
        self.steps = steps_per_epoch
        self.augment = augment
        self.unique_labels = np.unique(labels)
        self.label_to_indices = {
            l: np.where(labels == l)[0] for l in self.unique_labels
        }

    def __len__(self):
        return self.steps

    def _augment_img(self, img):
        """Augment đơn giản bằng numpy (không cần imgaug)."""
        h, w = img.shape[:2]
        result = img.copy()

        # Random flip horizontal (50%)
        if random.random() > 0.5:
            result = np.fliplr(result)

        # Random brightness shift ±10%
        shift = random.uniform(-0.1, 0.1)
        result = np.clip(result + shift, 0, 1)

        # Random Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, result.shape).astype(np.float32)
            result = np.clip(result + noise, 0, 1)

        return result

    def __getitem__(self, index):
        bs = self.batch_size
        h, w = self.images.shape[1], self.images.shape[2]
        x1 = np.empty((bs, h, w, 1), dtype=np.float32)
        x2 = np.empty((bs, h, w, 1), dtype=np.float32)
        y = np.zeros((bs, 1), dtype=np.float32)

        for i in range(bs):
            label_a = random.choice(self.unique_labels)
            idx_a = random.choice(self.label_to_indices[label_a])

            if random.random() > 0.5 or len(self.unique_labels) == 1:
                # Positive pair: cùng person, khác sample
                indices = self.label_to_indices[label_a]
                idx_b = random.choice(indices)
                # Cố gắng chọn sample KHÁC (không phải chính nó)
                if len(indices) > 1:
                    while idx_b == idx_a:
                        idx_b = random.choice(indices)
                y[i] = 1.0
            else:
                # Negative pair: khác person
                other_labels = [l for l in self.unique_labels if l != label_a]
                label_b = random.choice(other_labels)
                idx_b = random.choice(self.label_to_indices[label_b])
                y[i] = 0.0

            img_a = self.images[idx_a].copy()
            img_b = self.images[idx_b].copy()

            # Augment CẢ 2 ảnh (training only)
            if self.augment:
                img_a = self._augment_img(img_a)
                img_b = self._augment_img(img_b)

            x1[i] = img_a
            x2[i] = img_b

        return (x1, x2), y

    def on_epoch_end(self):
        pass


# ============================================================
# Model Architecture
# ============================================================
def build_sensor_model(input_shape=(90, 90, 1)):
    """
    Siamese Network thiết kế riêng cho ảnh SEN0348 (80x80 → 90x90).
    Nhỏ gọn hơn model SOCOFing nhưng đủ mạnh cho 24 persons.
    """
    inputs = layers.Input(shape=input_shape)

    # Feature Extractor
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)  # Output: vector thay vi GAP de giu lai spatial structure
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    feature_model = Model(inputs, x, name="FeatureExtractor")

    # Siamese
    img_a = layers.Input(shape=input_shape, name="input_1")
    img_b = layers.Input(shape=input_shape, name="input_2")

    feat_a = feature_model(img_a)
    feat_b = feature_model(img_b)

    # L1 distance
    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([feat_a, feat_b])

    # Comparison head (đơn giản hơn → ít overfitting với data ít)
    x = layers.Dense(64, activation='relu')(diff)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[img_a, img_b], outputs=output)
    return model


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train Siamese model chuyên cho ảnh SEN0348"
    )
    parser.add_argument('--data', default=os.path.join(_PROJECT_ROOT, 'sensor_dataset'),
                        help='Thư mục sensor_dataset/')
    parser.add_argument('--model', default=None,
                        help='Model .h5 để fine-tune (nếu không dùng --from-scratch)')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Train từ đầu (bỏ qua SOCOFing pretrained)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Số epochs (mặc định 60)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (mặc định 16)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (mặc định 3e-4)')
    parser.add_argument('--steps', type=int, default=200,
                        help='Steps per epoch (mặc định 200)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Tỉ lệ validation (mặc định 0.2)')
    args = parser.parse_args()

    # ---- Load data ----
    print(f"\n{'='*60}")
    print(f"  TRAIN SIAMESE CHO SEN0348")
    print(f"{'='*60}")

    if not os.path.isdir(args.data):
        print(f"ERROR: Không tìm thấy {args.data}")
        sys.exit(1)

    images, labels = FingerprintLoader.load_sensor_dataset(args.data, (90, 90))
    unique = np.unique(labels)
    print(f"Dataset: {len(images)} ảnh, {len(unique)} persons")
    for lbl in unique:
        print(f"  person_{lbl:03d}: {np.sum(labels == lbl)} ảnh")

    if len(unique) < 2:
        print("ERROR: Cần ít nhất 2 persons.")
        sys.exit(1)

    # ---- Preprocess ----
    print("\nPreprocessing (FFT denoise + CLAHE)...")
    images = preprocess_dataset(images)

    # ---- Train/Val split ----
    min_per_person = min(np.sum(labels == l) for l in unique)
    if min_per_person >= 4 and args.val_split > 0:
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            images, labels, test_size=args.val_split,
            stratify=labels, random_state=42
        )
        print(f"Split: {len(train_imgs)} train / {len(val_imgs)} val")
    else:
        train_imgs, train_labels = images, labels
        val_imgs, val_labels = None, None
        print(f"Dùng toàn bộ {len(images)} ảnh cho training (không val)")

    # ---- Generators ----
    train_gen = SensorTrainGenerator(
        train_imgs, train_labels,
        batch_size=args.batch,
        steps_per_epoch=args.steps,
        augment=True
    )
    val_gen = None
    if val_imgs is not None:
        val_gen = SensorTrainGenerator(
            val_imgs, val_labels,
            batch_size=args.batch,
            steps_per_epoch=50,  # Ít hơn train
            augment=False
        )

    print(f"\nTrain: {len(train_gen)} steps/epoch × {args.epochs} epochs "
          f"= {len(train_gen) * args.epochs} total updates")
    if val_gen:
        print(f"Val:   {len(val_gen)} steps/epoch")

    # ---- Build/Load model ----
    if args.from_scratch:
        print(f"\n>>> TRAIN TỪ ĐẦU (from scratch) <<<")
        model = build_sensor_model()
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            metrics=['acc']
        )
    else:
        # Fine-tune từ model đã có
        model_path = args.model
        if model_path is None:
            # Tìm model tốt nhất có sẵn
            candidates = [
                os.path.join(_PROJECT_ROOT, 'result', 'fingerprint_finetuned.h5'),
                os.path.join(_PROJECT_ROOT, 'result', 'fingerprint_siamese_model.h5'),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)

        if model_path is None or not os.path.exists(model_path):
            print("WARNING: Không tìm thấy model pretrained → train từ đầu.")
            model = build_sensor_model()
            model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                metrics=['acc']
            )
        else:
            print(f"\n>>> FINE-TUNE từ: {model_path} <<<")
            model = tf.keras.models.load_model(model_path)
            # Mở toàn bộ layers (no-freeze vì domain gap quá lớn)
            for layer in model.layers:
                layer.trainable = True
            model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                metrics=['acc']
            )

    trainable_count = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    print(f"Trainable parameters: {trainable_count:,}")
    model.summary()

    # ---- Callbacks ----
    os.makedirs(os.path.join(_PROJECT_ROOT, 'result'), exist_ok=True)
    out_h5 = os.path.join(_PROJECT_ROOT, 'result', 'fingerprint_finetuned.h5')
    monitor = 'val_loss' if val_gen else 'loss'

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            out_h5, monitor=monitor,
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=15,
            restore_best_weights=True, verbose=1
        ),
    ]

    # ---- Train ----
    print(f"\n{'='*40}")
    print(f"  Training {args.epochs} epochs, {args.steps} steps/epoch")
    print(f"  LR={args.lr}, Batch={args.batch}")
    print(f"{'='*40}\n")

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    # ---- Save & Export ----
    model.save(out_h5)
    print(f"\nModel saved: {out_h5}")

    out_tflite = os.path.join(_PROJECT_ROOT, 'result', 'fingerprint_finetuned.tflite')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_tflite, 'wb') as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(out_tflite) / 1024
    print(f"TFLite saved: {out_tflite} ({size_kb:.1f} KB)")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  HOÀN TẤT!")
    print(f"{'='*60}")
    final_loss = history.history['loss'][-1]
    final_acc = history.history['acc'][-1]
    print(f"  Train Loss: {final_loss:.4f}")
    print(f"  Train Acc:  {final_acc:.4f} ({final_acc*100:.1f}%)")
    if val_gen and 'val_loss' in history.history:
        print(f"  Val Loss:   {history.history['val_loss'][-1]:.4f}")
        print(f"  Val Acc:    {history.history['val_acc'][-1]:.4f}")

    print(f"\n  Đánh giá model:")
    print(f"    python scripts/evaluate_model.py --data {args.data}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

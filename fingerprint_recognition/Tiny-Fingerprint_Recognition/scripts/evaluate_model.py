"""
Đánh giá model Siamese trước khi deploy lên Jetson Nano.

Kiểm tra:
  - Accuracy (tỉ lệ đúng) trên ảnh SEN0348
  - FAR (False Accept Rate) — nhận nhầm người lạ
  - FRR (False Reject Rate) — từ chối nhầm người quen
  - Confusion matrix
  - Tìm ngưỡng (threshold) tối ưu

Cách dùng:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model result/fingerprint_finetuned.h5 --data sensor_dataset
    python scripts/evaluate_model.py --tflite result/fingerprint_finetuned.tflite --data sensor_dataset
"""
import argparse
import os
import sys
import itertools

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from fingerprint_ids.data.loader import FingerprintLoader
from fingerprint_ids.features.engineer import preprocess_sensor_image


def preprocess_dataset(images, target_size=(90, 90)):
    """Áp preprocessing SEN0348 cho toàn bộ dataset."""
    out = np.empty((len(images), target_size[0], target_size[1], 1), dtype=np.uint8)
    for i, img in enumerate(images):
        if img.ndim == 3:
            img2d = img[:, :, 0]
        else:
            img2d = img
        out[i] = preprocess_sensor_image(img2d, target_size=target_size)
    return out


def load_keras_model(model_path):
    """Load model Keras .h5 và trả về hàm predict."""
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded Keras model: {model_path}")

    def predict_pair(img1, img2):
        """img1, img2: (90,90,1) uint8 → score float"""
        t1 = img1.astype(np.float32).reshape(1, 90, 90, 1) / 255.0
        t2 = img2.astype(np.float32).reshape(1, 90, 90, 1) / 255.0
        score = model.predict([t1, t2], verbose=0)[0][0]
        return float(score)

    return predict_pair


def load_tflite_model(tflite_path):
    """Load model TFLite và trả về hàm predict."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Loaded TFLite model: {tflite_path}")
    for i, inp in enumerate(input_details):
        print(f"  Input[{i}]: name={inp['name']}, shape={inp['shape']}, dtype={inp['dtype']}")

    def predict_pair(img1, img2):
        t1 = img1.astype(np.float32).reshape(1, 90, 90, 1) / 255.0
        t2 = img2.astype(np.float32).reshape(1, 90, 90, 1) / 255.0
        d0, d1 = input_details[0], input_details[1]
        interpreter.set_tensor(d0['index'], t1.astype(d0['dtype']))
        interpreter.set_tensor(d1['index'], t2.astype(d1['dtype']))
        interpreter.invoke()
        score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        return score

    return predict_pair


def generate_all_pairs(images, labels):
    """
    Sinh tất cả cặp ảnh có thể:
      - positive pair: cùng person
      - negative pair: khác person
    Trả về list[(idx_a, idx_b, is_same_person)]
    """
    n = len(images)
    unique_labels = np.unique(labels)
    label_to_indices = {l: np.where(labels == l)[0] for l in unique_labels}

    positive_pairs = []
    negative_pairs = []

    # Positive: tất cả cặp cùng person
    for lbl in unique_labels:
        indices = label_to_indices[lbl]
        for i, j in itertools.combinations(indices, 2):
            positive_pairs.append((i, j, True))

    # Negative: lấy ngẫu nhiên, số lượng = positive
    # (để balance dataset)
    np.random.seed(42)
    for _ in range(len(positive_pairs)):
        lbl_a, lbl_b = np.random.choice(unique_labels, 2, replace=False)
        idx_a = np.random.choice(label_to_indices[lbl_a])
        idx_b = np.random.choice(label_to_indices[lbl_b])
        negative_pairs.append((idx_a, idx_b, False))

    all_pairs = positive_pairs + negative_pairs
    np.random.shuffle(all_pairs)
    return all_pairs


def evaluate(predict_fn, images, labels, thresholds=None):
    """
    Đánh giá model trên tất cả cặp ảnh.

    Returns:
        dict chứa kết quả chi tiết cho từng threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    pairs = generate_all_pairs(images, labels)
    print(f"\nSố cặp: {len(pairs)} (positive: {sum(1 for p in pairs if p[2])}, "
          f"negative: {sum(1 for p in pairs if not p[2])})")

    # Tính score cho tất cả cặp
    scores = []
    ground_truth = []
    total = len(pairs)

    print("Đang tính score cho tất cả cặp...")
    for idx, (i, j, is_same) in enumerate(pairs):
        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"  [{idx+1}/{total}]", end='\r')
        score = predict_fn(images[i], images[j])
        scores.append(score)
        ground_truth.append(is_same)

    scores = np.array(scores)
    ground_truth = np.array(ground_truth)
    print()

    # Đánh giá từng threshold
    results = []
    best_acc = 0
    best_threshold = 0.5

    print(f"\n{'Threshold':>10} {'Accuracy':>10} {'FAR':>8} {'FRR':>8} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}")
    print("-" * 72)

    for thr in thresholds:
        predictions = scores > thr

        tp = int(np.sum(predictions & ground_truth))        # True Positive
        tn = int(np.sum(~predictions & ~ground_truth))      # True Negative
        fp = int(np.sum(predictions & ~ground_truth))       # False Positive (FAR)
        fn = int(np.sum(~predictions & ground_truth))       # False Negative (FRR)

        total_pos = int(np.sum(ground_truth))
        total_neg = int(np.sum(~ground_truth))

        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
        far = fp / total_neg if total_neg > 0 else 0        # False Accept Rate
        frr = fn / total_pos if total_pos > 0 else 0        # False Reject Rate

        results.append({
            'threshold': float(thr),
            'accuracy': accuracy,
            'far': far,
            'frr': frr,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })

        marker = ""
        if accuracy > best_acc:
            best_acc = accuracy
            best_threshold = float(thr)
            marker = " ← BEST"

        print(f"{thr:>10.2f} {accuracy:>10.4f} {far:>8.4f} {frr:>8.4f} "
              f"{tp:>6} {tn:>6} {fp:>6} {fn:>6}{marker}")

    # Tóm tắt
    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ ĐÁNH GIÁ")
    print(f"{'='*60}")
    print(f"  Ngưỡng tối ưu      : {best_threshold:.2f}")
    print(f"  Accuracy tốt nhất  : {best_acc:.4f} ({best_acc*100:.1f}%)")

    # Tìm kết quả tại best threshold
    best_result = [r for r in results if abs(r['threshold'] - best_threshold) < 0.01][0]
    print(f"  FAR (nhận nhầm)    : {best_result['far']:.4f} ({best_result['far']*100:.2f}%)")
    print(f"  FRR (từ chối nhầm) : {best_result['frr']:.4f} ({best_result['frr']*100:.2f}%)")

    # Đánh giá chất lượng
    if best_acc >= 0.95:
        quality = "XUẤT SẮC ✓ — Sẵn sàng deploy"
    elif best_acc >= 0.85:
        quality = "TỐT ✓ — Có thể deploy, cân nhắc thu thêm ảnh"
    elif best_acc >= 0.70:
        quality = "TRUNG BÌNH ⚠ — Cần thu thêm ảnh hoặc thử --no-freeze"
    else:
        quality = "CHƯA ĐẠT ✗ — Cần thu thêm nhiều ảnh và train lại"

    print(f"  Đánh giá           : {quality}")

    print(f"\n  === GỢI Ý ===")
    print(f"  Dùng ngưỡng {best_threshold:.2f} trong config.py:")
    print(f"    MATCH_THRESHOLD = {best_threshold}")
    print(f"{'='*60}")

    # Distribution thống kê
    pos_scores = scores[ground_truth]
    neg_scores = scores[~ground_truth]
    print(f"\n  Score distribution:")
    print(f"    Positive (cùng người): mean={pos_scores.mean():.4f}, "
          f"std={pos_scores.std():.4f}, min={pos_scores.min():.4f}, max={pos_scores.max():.4f}")
    print(f"    Negative (khác người): mean={neg_scores.mean():.4f}, "
          f"std={neg_scores.std():.4f}, min={neg_scores.min():.4f}, max={neg_scores.max():.4f}")

    separation = pos_scores.mean() - neg_scores.mean()
    print(f"    Separation (gap)     : {separation:.4f}")
    if separation < 0.2:
        print(f"    ⚠ Separation quá thấp! Model không phân biệt tốt.")
        print(f"    → Thử: thu thêm ảnh, train thêm epochs, hoặc dùng --no-freeze")

    return results, best_threshold


def main():
    parser = argparse.ArgumentParser(description="Đánh giá model Siamese trước khi deploy")
    parser.add_argument('--model',
                        default=os.path.join(_PROJECT_ROOT, 'result', 'fingerprint_finetuned.h5'),
                        help='Model Keras .h5')
    parser.add_argument('--tflite', default=None,
                        help='Model TFLite (nếu cung cấp, sẽ dùng thay .h5)')
    parser.add_argument('--data',
                        default=os.path.join(_PROJECT_ROOT, 'sensor_dataset'),
                        help='Thư mục sensor_dataset/')
    args = parser.parse_args()

    # Load dataset
    if not os.path.isdir(args.data):
        print(f"ERROR: Không tìm thấy thư mục: {args.data}")
        sys.exit(1)

    images, labels = FingerprintLoader.load_sensor_dataset(args.data, target_size=(90, 90))
    if len(images) == 0:
        print("ERROR: Không có ảnh nào.")
        sys.exit(1)

    print(f"Dataset: {len(images)} ảnh, {len(np.unique(labels))} người")

    # Preprocess
    print("Preprocessing (FFT denoise + CLAHE)...")
    images = preprocess_dataset(images)

    # Load model
    if args.tflite and os.path.exists(args.tflite):
        predict_fn = load_tflite_model(args.tflite)
    elif os.path.exists(args.model):
        predict_fn = load_keras_model(args.model)
    else:
        print(f"ERROR: Không tìm thấy model.")
        print(f"  Thử: {args.model}")
        if args.tflite:
            print(f"  Hoặc: {args.tflite}")
        sys.exit(1)

    # Evaluate
    evaluate(predict_fn, images, labels)


if __name__ == "__main__":
    main()

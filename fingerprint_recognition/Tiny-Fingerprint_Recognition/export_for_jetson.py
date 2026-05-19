"""
Script xuất model đã train → TFLite tối ưu cho Jetson Nano.
Chạy trên Windows sau khi train xong.

Cách dùng:
  python export_for_jetson.py
  python export_for_jetson.py --model result/fingerprint_siamese_model.h5 --output jetson/fingerprint_model.tflite
"""
import argparse
import os
import shutil
import tensorflow as tf


def export_tflite(model_path, output_path, quantize=True):
    """
    Chuyển đổi model Keras (.h5) sang TFLite tối ưu cho Jetson Nano.

    Args:
        model_path: Đường dẫn model Keras (.h5)
        output_path: Đường dẫn output file .tflite
        quantize: Bật dynamic range quantization (giảm dung lượng ~4x)
    """
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    model.summary()

    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Dynamic range quantization: ON")

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nExported: {output_path} ({size_mb:.2f} MB)")

    return output_path


def verify_tflite(tflite_path):
    """Kiểm tra model TFLite đã export."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    print(f"\n--- TFLite Model Info ---")
    print(f"Number of inputs: {len(inputs)}")
    for i, inp in enumerate(inputs):
        print(f"  Input[{i}]: name={inp['name']}, shape={inp['shape']}, dtype={inp['dtype']}")
    print(f"Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output[{i}]: name={out['name']}, shape={out['shape']}, dtype={out['dtype']}")
    print("Model verification: OK")


def main():
    parser = argparse.ArgumentParser(description="Export model for Jetson Nano deployment")
    parser.add_argument('--model', default='result/fingerprint_siamese_model.h5',
                        help='Đường dẫn model Keras (.h5)')
    parser.add_argument('--output', default='jetson/fingerprint_model.tflite',
                        help='Đường dẫn output TFLite')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Tắt quantization (giữ float32)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Không tìm thấy model: {args.model}")
        print("Hãy chạy training trước: python scripts/run_pipeline.py")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output = export_tflite(args.model, args.output, quantize=not args.no_quantize)
    verify_tflite(output)

    print(f"\n{'='*50}")
    print(f"  Hoàn tất! Copy file sau sang Jetson Nano:")
    print(f"  → {os.path.abspath(output)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

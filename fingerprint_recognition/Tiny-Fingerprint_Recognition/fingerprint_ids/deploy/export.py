import tensorflow as tf

def export_to_tflite(model, export_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Optimized for capacity
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(export_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model exported to {export_path}")
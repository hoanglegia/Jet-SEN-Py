from tensorflow.keras import layers, Model
import tensorflow as tf


def build_siamese_model(input_shape=(90, 90, 1)):
    """
    Siamese Network cho nhận diện vân tay.

    Kiến trúc:
      - FeatureExtractor (shared weights): 3 Conv blocks + BatchNorm + Dropout
      - Comparison head: L1 distance → Conv → Dense → sigmoid

    Cải tiến so với bản gốc:
      - Thêm BatchNormalization → ổn định training
      - Thêm Dropout → giảm overfitting, quan trọng khi data ít
      - 3 Conv blocks thay vì 2 → capture thêm features
      - Giữ nhỏ gọn (32-64 filters) để chạy được trên Jetson Nano 4GB
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Block 2
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # Block 3
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Dropout(0.25)(x)

    feature_model = Model(inputs=inputs, outputs=x, name="FeatureExtractor")

    # Siamese twin inputs
    img_a = layers.Input(shape=input_shape)
    img_b = layers.Input(shape=input_shape)

    feat_a = feature_model(img_a)
    feat_b = feature_model(img_b)

    # L1 distance giữa 2 feature maps
    combined = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([feat_a, feat_b])

    # Comparison head
    combined = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.MaxPooling2D(pool_size=2)(combined)
    combined = layers.Flatten()(combined)
    combined = layers.Dropout(0.3)(combined)
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)
    combined = layers.Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[img_a, img_b], outputs=combined)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def load_and_finetune(model_path, learning_rate=1e-4, freeze_feature=True):
    """
    Load model đã train → chuẩn bị cho fine-tuning.

    Args:
        model_path: đường dẫn file .h5
        learning_rate: learning rate nhỏ cho fine-tuning (mặc định 1e-4)
        freeze_feature: True = đóng băng FeatureExtractor (chỉ train comparison head)
                        False = mở khóa toàn bộ (cần nhiều data hơn)

    Returns:
        model đã compile với learning rate mới, sẵn sàng .fit()
    """
    model = tf.keras.models.load_model(model_path)

    if freeze_feature:
        # Tìm và freeze FeatureExtractor
        for layer in model.layers:
            if layer.name == "FeatureExtractor":
                layer.trainable = False
                print(f"[Fine-tune] Frozen: {layer.name} ({len(layer.layers)} sub-layers)")
                break
        else:
            # Fallback: nếu không tìm thấy FeatureExtractor,
            # freeze 60% layers đầu
            total = len(model.layers)
            freeze_until = int(total * 0.6)
            for i, layer in enumerate(model.layers):
                if i < freeze_until:
                    layer.trainable = False
            print(f"[Fine-tune] Frozen {freeze_until}/{total} layers (fallback)")
    else:
        print("[Fine-tune] All layers trainable (no-freeze mode)")

    # Compile lại với learning rate nhỏ
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['acc']
    )

    trainable = sum(1 for l in model.layers if l.trainable)
    total = len(model.layers)
    print(f"[Fine-tune] Trainable: {trainable}/{total} layers, lr={learning_rate}")
    return model
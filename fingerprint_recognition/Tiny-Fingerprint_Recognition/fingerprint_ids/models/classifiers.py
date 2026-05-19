from tensorflow.keras import layers, Model
import tensorflow as tf

def build_siamese_model(input_shape=(90, 90, 1)):
    inputs = layers.Input(shape=input_shape)
    feature = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    feature = layers.MaxPooling2D(pool_size=2)(feature)
    feature = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(feature)
    feature = layers.MaxPooling2D(pool_size=2)(feature)
    
    feature_model = Model(inputs=inputs, outputs=feature, name="FeatureExtractor")

    img_a = layers.Input(shape=input_shape)
    img_b = layers.Input(shape=input_shape)

    feat_a = feature_model(img_a)
    feat_b = feature_model(img_b)

    combined = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([feat_a, feat_b])
    combined = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(combined)
    combined = layers.MaxPooling2D(pool_size=2)(combined)
    combined = layers.Flatten()(combined)
    combined = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = Model(inputs=[img_a, img_b], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model
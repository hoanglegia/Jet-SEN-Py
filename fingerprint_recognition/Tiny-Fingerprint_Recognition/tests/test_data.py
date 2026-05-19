import numpy as np
from fingerprint_ids.data.loader import FingerprintLoader, FingerprintGenerator

def test_create_label_dict():
    y_real = np.array([[1, 0, 0, 1], [2, 1, 1, 0]])
    label_dict = FingerprintLoader.create_label_dict(y_real)
    
    # Check key format (zfill 6)
    assert "01001" in label_dict or "001001" in label_dict 
    assert label_dict[list(label_dict.keys())[0]] == 0

def test_generator_output():
    # Create simulated data
    x_train = np.random.randint(0, 255, (64, 90, 90, 1), dtype=np.uint8)
    y_train = np.array([[1, 0, 0, 1]] * 32 + [[2, 1, 1, 0]] * 32)
    x_real = x_train
    label_dict = FingerprintLoader.create_label_dict(y_train)
    
    gen = FingerprintGenerator(x_train, y_train, x_real, label_dict, batch_size=32)
    
    (img_batch_1, img_batch_2), label_batch = gen[0]
    
    assert img_batch_1.shape == (32, 90, 90, 1)
    assert label_batch.shape == (32, 1)
    assert np.max(img_batch_1) <= 1.0 # Check normalization
import numpy as np
import pytest
from fingerprint_ids.features.engineer import extract_label, preprocess_image

def test_extract_label():
    # Simulate file paths in SOCOFing format.
    test_path = "dataset/Real/100__M_Left_index_finger.BMP"
    label = extract_label(test_path)
    
    assert isinstance(label, np.ndarray)
    assert label[0] == 100 # Subject ID
    assert label[1] == 0   # Male
    assert label[2] == 0   # Left
    assert label[3] == 1   # Index finger

def test_preprocess_image():
    dummy_img = np.zeros((100, 100), dtype=np.uint8)
    processed = preprocess_image(dummy_img, target_size=(90, 90))
    
    assert processed.shape == (90, 90, 1)
    assert processed.dtype == np.uint8
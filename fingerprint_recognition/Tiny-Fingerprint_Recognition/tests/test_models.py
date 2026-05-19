from fingerprint_ids.models.classifiers import build_siamese_model
from fingerprint_ids.models.compress import symmetric_min_max_quantization
import numpy as np

def test_siamese_structure():
    model = build_siamese_model(input_shape=(90, 90, 1))
    
    # Check the number of inputs.
    assert len(model.inputs) == 2
    # Check output activation
    assert model.layers[-1].activation.__name__ == 'sigmoid'

def test_quantization_logic():
    weights = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    q_weights, scale = symmetric_min_max_quantization(weights)
    
    assert q_weights.dtype == np.int8
    assert q_weights[0] == -127
    assert q_weights[2] == 127
    assert scale > 0
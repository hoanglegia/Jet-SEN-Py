import numpy as np

def symmetric_min_max_quantization(weights):
    """Perform weighted compression to int8 format using Symmetric Min-Max."""
    max_val = np.max(np.abs(weights))
    scale = 127 / max_val
    quantized_weights = np.round(weights * scale).astype(np.int8)
    return quantized_weights, scale

def quantize_model_layers(model):
    quantized_data = {}
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0:
            w, b = layer.get_weights()
            qw, sw = symmetric_min_max_quantization(w)
            qb, sb = symmetric_min_max_quantization(b)
            quantized_data[layer.name] = {'weights': qw, 'scale_w': sw, 'bias': qb, 'scale_b': sb}
    return quantized_data
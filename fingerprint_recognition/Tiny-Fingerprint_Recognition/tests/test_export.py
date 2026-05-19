import os
from fingerprint_ids.deploy.c_export import tflite_to_c_array

def test_c_header_generation(tmp_path):
    # Create a simulated tflite file.
    tflite_file = tmp_path / "test_model.tflite"
    tflite_file.write_bytes(b"\x00\x01\x02\x03")
    
    header_file = tmp_path / "test_model.h"
    tflite_to_c_array(str(tflite_file), str(header_file))
    
    assert os.path.exists(header_file)
    content = header_file.read_text()
    assert "unsigned char test_model" in content
    assert "0x00, 0x01, 0x02, 0x03" in content
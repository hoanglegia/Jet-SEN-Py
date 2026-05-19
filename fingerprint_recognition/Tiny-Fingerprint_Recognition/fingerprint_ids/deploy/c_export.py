import os

def tflite_to_c_array(tflite_path, c_path):
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    var_name = os.path.splitext(os.path.basename(tflite_path))[0]
    
    with open(c_path, 'w') as f:
        f.write(f'unsigned char {var_name}[] = {{\n')
        for i, byte in enumerate(data):
            f.write(f'0x{byte:02x}, ' if i % 12 != 11 else f'0x{byte:02x},\n')
        f.write('\n};\n')
        f.write(f'unsigned int {var_name}_len = {len(data)};\n')
    print(f"C Header generated at {c_path}")
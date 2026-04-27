import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

# Fixed key matching backend/src/utils/crypto.js
KEY = bytes.fromhex('59713d2f939379854746ba1f39c0cc3f59713d2f939379854746ba1f39c0cc3f')

def decrypt_file(input_path, output_path):
    """
    Decrypts a file encrypted with Node.js crypto (AES-256-CBC, IV in first 16 bytes).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, 'rb') as f:
        iv = f.read(16)
        if len(iv) < 16:
            raise ValueError("Failed to read 16-byte IV from file header")
        
        encrypted_data = f.read()
        
    cipher = Cipher(algorithms.AES(KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    
    try:
        padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Node.js 'aes-256-cbc' uses PKCS7 padding (128-bit block size)
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        
        with open(output_path, 'wb') as f:
            f.write(data)
    except Exception as e:
        raise ValueError(f"Decryption or unpadding failed: {str(e)}")

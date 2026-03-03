import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_password(key: bytes, plaintext: str):
    """
    Encrypts password using AES-GCM (AEAD)
    Returns nonce, ciphertext
    """

    aesgcm = AESGCM(key)

    nonce = os.urandom(12)  # 96-bit nonce (standard for GCM)

    ciphertext = aesgcm.encrypt(
        nonce,
        plaintext.encode(),
        None  # Associated Data (can add metadata later)
    )

    return nonce, ciphertext


def decrypt_password(key: bytes, nonce: bytes, ciphertext: bytes):
    """
    Decrypts AES-GCM encrypted password
    """

    aesgcm = AESGCM(key)

    plaintext = aesgcm.decrypt(
        nonce,
        ciphertext,
        None
    )

    return plaintext.decode()
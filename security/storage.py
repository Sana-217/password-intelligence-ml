import json
import base64
from security.key_derivation import derive_key
from security.encryption import encrypt_password, decrypt_password


def store_password(master_password: str, generated_password: str, filename="secure_store.json"):
    """
    Encrypt and store password in zero-knowledge format.
    """

    key, salt = derive_key(master_password)

    nonce, ciphertext = encrypt_password(key, generated_password)

    data = {
        "salt": base64.b64encode(salt).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode()
    }

    with open(filename, "w") as f:
        json.dump(data, f)

    print("Password securely stored (encrypted).")


def retrieve_password(master_password: str, filename="secure_store.json"):
    """
    Decrypt stored password.
    """

    with open(filename, "r") as f:
        data = json.load(f)

    salt = base64.b64decode(data["salt"])
    nonce = base64.b64decode(data["nonce"])
    ciphertext = base64.b64decode(data["ciphertext"])

    key, _ = derive_key(master_password, salt)

    decrypted = decrypt_password(key, nonce, ciphertext)

    return decrypted
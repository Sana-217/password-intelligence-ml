import os
from argon2.low_level import hash_secret_raw, Type


def derive_key(password: str, salt: bytes = None):
    """
    Derives a 256-bit encryption key using Argon2id.
    Returns (derived_key, salt)
    """

    if salt is None:
        salt = os.urandom(16)  # 128-bit salt

    key = hash_secret_raw(
        secret=password.encode(),
        salt=salt,
        time_cost=3,          # iterations
        memory_cost=65536,    # 64 MB memory
        parallelism=2,
        hash_len=32,          # 256-bit key
        type=Type.ID
    )

    return key, salt
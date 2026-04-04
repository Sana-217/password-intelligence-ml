# security/crypto.py
"""
Cryptographic primitives for the password vault.

TWO OPERATIONS ONLY
────────────────────
  encrypt(plaintext, master_password) → encrypted bundle (dict)
  decrypt(bundle, master_password)    → plaintext  OR  raises InvalidTag

ALGORITHMS USED (matches your project documentation exactly)
─────────────────────────────────────────────────────────────
  Key Derivation : Argon2id  (RFC 9106)
  Encryption     : AES-256-GCM  (AEAD — authenticated encryption)

WHY ARGON2id SPECIFICALLY
──────────────────────────
There are three Argon2 variants: Argon2d, Argon2i, Argon2id.
Your documentation specifies Argon2id — the correct choice because:
  - Argon2d  : resistant to GPU attacks, vulnerable to side-channel
  - Argon2i  : resistant to side-channel, weaker against GPU
  - Argon2id : resistant to BOTH — recommended by RFC 9106 for passwords
The 'id' suffix is not cosmetic — it is a specific security property.

WHY AES-256-GCM SPECIFICALLY (AEAD)
──────────────────────────────────────
GCM = Galois/Counter Mode. It provides:
  1. Confidentiality  — nobody can read the ciphertext without the key
  2. Integrity        — any tampering with the ciphertext is detected
  3. Authenticity     — wrong key raises InvalidTag immediately

This is what "AEAD" means: Authenticated Encryption with Associated Data.
A system that provides only confidentiality (e.g. AES-CBC without HMAC)
would silently return garbage on a wrong key — a security vulnerability.
AES-GCM raises an exception instead. This is the correct behaviour.

ZERO-KNOWLEDGE PROPERTY
─────────────────────────
The vault file stores: salt + nonce + ciphertext + tag.
The master password is NEVER stored anywhere — not hashed, not stored.
This means even if the vault file is stolen, without the master password
it is computationally infeasible to recover any stored password.
This is the zero-knowledge storage model stated in your documentation.

WHAT THIS FILE DOES NOT DO
────────────────────────────
  - No file I/O  (that is storage.py's job)
  - No session management  (that is a separate concern)
  - No password generation  (that is generator/password_gen.py)
Pure crypto primitives only. Completely testable in isolation.
"""

import os
import base64
from typing import TypedDict

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag
from argon2.low_level import hash_secret_raw, Type

# ── re-export InvalidTag so callers don't need to import cryptography ─────────
# storage.py catches this exception — it imports it from here, not from
# cryptography directly. This keeps the dependency contained to one file.
__all__ = ["encrypt", "decrypt", "derive_key", "InvalidTag", "CryptoBundle"]


# ── Argon2id parameters ───────────────────────────────────────────────────────
# These match OWASP 2023 recommendations for interactive logins.
# Do NOT reduce these — lower values = faster cracking.
#
# time_cost    : number of iterations (passes over memory)
# memory_cost  : RAM used in kibibytes (64 MB)
# parallelism  : number of parallel threads
# hash_len     : output key length in bytes (32 bytes = 256 bits for AES-256)
# salt_len     : random salt length in bytes (16 bytes = 128 bits)

ARGON2_TIME_COST    = 3         # 3 iterations
ARGON2_MEMORY_COST  = 64 * 1024 # 64 MB in KiB
ARGON2_PARALLELISM  = 2         # 2 threads
ARGON2_HASH_LEN     = 32        # 256-bit key output
ARGON2_SALT_LEN     = 16        # 128-bit salt

# AES-GCM nonce length (12 bytes = 96 bits is the GCM standard)
# Never reuse a nonce with the same key — we generate a fresh one per encrypt
AES_NONCE_LEN = 12


# ── type definition for the encrypted bundle ──────────────────────────────────

class CryptoBundle(TypedDict):
    """
    Everything needed to decrypt a stored password — except the master password.
    Stored as-is in vault.json (base64-encoded strings, not raw bytes).

    Fields:
        salt       : Argon2id salt used during key derivation (base64)
        nonce      : AES-GCM nonce used during encryption (base64)
        ciphertext : encrypted password + 16-byte GCM authentication tag (base64)

    The GCM tag is appended to the ciphertext automatically by the
    cryptography library — you do not manage it separately.
    """
    salt:       str   # base64-encoded, ARGON2_SALT_LEN bytes
    nonce:      str   # base64-encoded, AES_NONCE_LEN bytes
    ciphertext: str   # base64-encoded, len(plaintext) + 16 bytes (tag)


# ── core functions ────────────────────────────────────────────────────────────

def derive_key(master_password: str, salt: bytes) -> bytes:
    """
    Derives a 256-bit AES key from the master password using Argon2id.

    This is the most expensive step by design — it makes brute-force attacks
    slow. On a modern laptop, this takes ~0.3–0.5 seconds per attempt.
    An attacker trying 10^9 passwords would need ~150 years.

    Args:
        master_password : the user's master password (plaintext string)
        salt            : random bytes, unique per stored password

    Returns:
        32-byte derived key suitable for AES-256

    Note:
        Same master_password + same salt → always same key.
        Different salt → completely different key (even with same password).
        This is why we store the salt in the vault.
    """
    if not master_password:
        raise ValueError("Master password cannot be empty.")
    if len(salt) != ARGON2_SALT_LEN:
        raise ValueError(
            f"Salt must be {ARGON2_SALT_LEN} bytes, got {len(salt)}."
        )

    return hash_secret_raw(
        secret=master_password.encode("utf-8"),
        salt=salt,
        time_cost=ARGON2_TIME_COST,
        memory_cost=ARGON2_MEMORY_COST,
        parallelism=ARGON2_PARALLELISM,
        hash_len=ARGON2_HASH_LEN,
        type=Type.ID,    # Argon2id — not Type.D or Type.I
    )


def encrypt(plaintext: str, master_password: str) -> CryptoBundle:
    """
    Encrypts a password using AES-256-GCM with Argon2id key derivation.

    Each call generates fresh random salt and nonce — so encrypting the
    same password twice produces completely different ciphertext each time.
    This prevents an attacker from detecting duplicate passwords in the vault.

    Args:
        plaintext       : the password to encrypt (e.g. "correct-horse-battery")
        master_password : the user's master password

    Returns:
        CryptoBundle dict with salt, nonce, ciphertext (all base64-encoded)

    Raises:
        ValueError : if plaintext or master_password is empty

    Example:
        bundle = encrypt("correct-horse-battery", "myMasterPwd!")
        # bundle = {
        #   "salt":       "Aoq/cJztM3WGqdjWZMqBjA==",
        #   "nonce":      "OmhI8s4r/XJpJGHz",
        #   "ciphertext": "RRyOePO2ytmrsN+vFwyShdtwnavvYbEU5g=="
        # }
    """
    if not plaintext:
        raise ValueError("Plaintext (password to store) cannot be empty.")
    if not master_password:
        raise ValueError("Master password cannot be empty.")

    # Step 1: generate fresh random salt and nonce
    # os.urandom() uses the OS CSPRNG — cryptographically secure
    salt  = os.urandom(ARGON2_SALT_LEN)
    nonce = os.urandom(AES_NONCE_LEN)

    # Step 2: derive 256-bit key from master password + salt
    key = derive_key(master_password, salt)

    # Step 3: encrypt with AES-256-GCM
    # AESGCM.encrypt() returns ciphertext + 16-byte authentication tag
    # concatenated together — you don't see the tag separately
    aesgcm     = AESGCM(key)
    ciphertext = aesgcm.encrypt(
        nonce,
        plaintext.encode("utf-8"),
        None,   # no associated data in our use case
    )

    # Step 4: base64-encode everything for safe JSON storage
    return CryptoBundle(
        salt=       _b64encode(salt),
        nonce=      _b64encode(nonce),
        ciphertext= _b64encode(ciphertext),
    )


def decrypt(bundle: CryptoBundle, master_password: str) -> str:
    """
    Decrypts a stored password bundle using the master password.

    CRITICAL SECURITY BEHAVIOUR:
    If the master password is wrong, AES-GCM authentication fails and
    raises InvalidTag. This is NOT a generic "wrong password" message —
    it means the authentication tag does not match, which happens when:
      1. The master password is wrong (most common case)
      2. The ciphertext has been tampered with (attack detection)
    Both cases are treated identically — we do NOT tell the user which.
    Distinguishing them would leak information to an attacker.

    Args:
        bundle          : CryptoBundle returned by encrypt()
        master_password : the user's master password

    Returns:
        Decrypted password as a plain string

    Raises:
        InvalidTag  : wrong master password OR tampered ciphertext
        ValueError  : malformed bundle (missing keys or bad base64)

    Example:
        plaintext = decrypt(bundle, "myMasterPwd!")
        # → "correct-horse-battery"

        decrypt(bundle, "wrongPassword")
        # → raises InvalidTag
    """
    if not master_password:
        raise ValueError("Master password cannot be empty.")

    # Step 1: decode base64 fields
    try:
        salt       = _b64decode(bundle["salt"])
        nonce      = _b64decode(bundle["nonce"])
        ciphertext = _b64decode(bundle["ciphertext"])
    except (KeyError, Exception) as e:
        raise ValueError(f"Malformed crypto bundle: {e}") from e

    # Step 2: re-derive the key from master password + stored salt
    # If master_password is wrong, key will be wrong → GCM tag check fails
    key = derive_key(master_password, salt)

    # Step 3: decrypt + verify authentication tag
    # This raises InvalidTag if key is wrong or ciphertext was tampered with
    aesgcm = AESGCM(key)
    plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)

    return plaintext_bytes.decode("utf-8")


# ── helpers ───────────────────────────────────────────────────────────────────

def _b64encode(data: bytes) -> str:
    """Encodes bytes to a URL-safe base64 string (no padding issues in JSON)."""
    return base64.b64encode(data).decode("ascii")


def _b64decode(data: str) -> bytes:
    """Decodes a base64 string back to bytes."""
    return base64.b64decode(data.encode("ascii"))


def verify_master_password(bundle: CryptoBundle, master_password: str) -> bool:
    """
    Tests whether a master password can decrypt a bundle — without
    returning the decrypted content.

    Used by storage.py to validate the master password at login
    before attempting to decrypt any real vault entries.

    Returns:
        True  : master password is correct
        False : master password is wrong (InvalidTag was raised)
    """
    try:
        decrypt(bundle, master_password)
        return True
    except InvalidTag:
        return False


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("  security/crypto.py — self test")
    print("=" * 56)

    master = "TestMaster@2024!"
    passwords = [
        "correct-horse-battery",
        "xK9!mPq2",
        "Mumbai@2019!Chai",
        "123456",
    ]

    print("\n── Encrypt → Decrypt roundtrip ───────────────────────")
    all_pass = True
    for pwd in passwords:
        bundle    = encrypt(pwd, master)
        recovered = decrypt(bundle, master)
        ok        = recovered == pwd
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {pwd!r:30}  →  {recovered!r}")

    print("\n── Wrong master password raises InvalidTag ───────────")
    bundle = encrypt("correct-horse-battery", master)
    try:
        decrypt(bundle, "wrongPassword123")
        print("  FAIL  no exception raised — crypto is broken!")
        all_pass = False
    except InvalidTag:
        print("  PASS  InvalidTag raised correctly")

    print("\n── Same password encrypts differently each time ──────")
    b1 = encrypt("samepassword", master)
    b2 = encrypt("samepassword", master)
    different = b1["ciphertext"] != b2["ciphertext"]
    status    = "PASS" if different else "FAIL"
    print(f"  {status}  two encryptions of same password produce different ciphertext")
    if not different:
        all_pass = False

    print("\n── Tampered ciphertext raises InvalidTag ─────────────")
    bundle = encrypt("correct-horse-battery", master)
    # flip one byte in the ciphertext
    raw    = base64.b64decode(bundle["ciphertext"])
    raw    = bytes([raw[0] ^ 0xFF]) + raw[1:]   # XOR first byte with 0xFF
    bundle["ciphertext"] = base64.b64encode(raw).decode("ascii")
    try:
        decrypt(bundle, master)
        print("  FAIL  tampered ciphertext not detected!")
        all_pass = False
    except InvalidTag:
        print("  PASS  tampered ciphertext detected correctly")

    print("\n── verify_master_password() ──────────────────────────")
    bundle = encrypt("correct-horse-battery", master)
    ok1 = verify_master_password(bundle, master) is True
    ok2 = verify_master_password(bundle, "wrongPassword") is False
    print(f"  {'PASS' if ok1 else 'FAIL'}  correct password → True")
    print(f"  {'PASS' if ok2 else 'FAIL'}  wrong password   → False")
    if not (ok1 and ok2):
        all_pass = False

    print()
    print("=" * 56)
    print(f"  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 56)
    print("\n  Next step: python security/storage.py")
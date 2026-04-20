"""
security/storage.py  — Multi-User Version
─────────────────────────────────────────────────────────────────────────────
Drop-in replacement for the original single-user storage.py.

CHANGES FROM SINGLE-USER VERSION
──────────────────────────────────
  - Each user gets their own vault file: vault_<username>.json
  - vault_meta.json now stores ALL registered users
  - VaultManager accepts a username parameter
  - initialise_vault(master, username, hint) creates a per-user vault
  - vault_exists(username) checks for a specific user's vault
  - list_users() returns all registered usernames

BACKWARDS COMPATIBILITY
────────────────────────
  - All existing routes (store, retrieve, delete, list) work unchanged
  - VaultManager API is identical — constructor gets optional username param
  - Flask app passes username from session["username"]

VAULT FILE STRUCTURE
─────────────────────
  vault_<username>.json  — encrypted entries for that user
  vault_meta.json        — { "users": { "sana": { "hint": "..." }, ... } }
"""

import json
import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


# ── Exceptions ────────────────────────────────────────────────────────────────
class VaultError(Exception):               pass
class WrongMasterPasswordError(VaultError): pass
class EntryNotFoundError(VaultError):       pass
class EntryAlreadyExistsError(VaultError):  pass
class UserAlreadyExistsError(VaultError):   pass
class UserNotFoundError(VaultError):        pass


# ── Path helpers ──────────────────────────────────────────────────────────────

def _sanitise(username: str) -> str:
    """Sanitise username for safe use in filename."""
    return "".join(c for c in username.lower() if c.isalnum() or c in "_-")


def _vault_path(username: str) -> Path:
    """Return vault file path for a given username."""
    return ROOT / f"vault_{_sanitise(username)}.json"


def _meta_path() -> Path:
    return ROOT / "vault_meta.json"


# ── Meta file helpers ─────────────────────────────────────────────────────────

def _load_meta() -> dict:
    p = _meta_path()
    if not p.exists():
        return {"users": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"users": {}}


def _save_meta(meta: dict) -> None:
    tmp = _meta_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    tmp.replace(_meta_path())


# ── Vault file helpers ────────────────────────────────────────────────────────

def _load_vault(username: str) -> dict:
    p = _vault_path(username)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_vault(username: str, data: dict) -> None:
    """Atomic write — write to .tmp then rename."""
    p   = _vault_path(username)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


# ── Public helper functions ───────────────────────────────────────────────────

def vault_exists(username: str = "") -> bool:
    """
    Return True if a vault exists.
      - username given  → checks that specific user's vault file
      - no username     → checks if ANY user is registered (login page)
    """
    if username:
        return _vault_path(username).exists()
    meta = _load_meta()
    return bool(meta.get("users"))


def list_users() -> list:
    """Return list of all registered display names."""
    meta = _load_meta()
    return [
        v.get("display_name", k)
        for k, v in meta.get("users", {}).items()
    ]


def get_user_hint(username: str) -> str:
    """Return the master password hint for a user."""
    meta = _load_meta()
    user = meta.get("users", {}).get(_sanitise(username), {})
    return user.get("hint", "")


def user_exists(username: str) -> bool:
    """Return True if username is already registered."""
    meta = _load_meta()
    return _sanitise(username) in meta.get("users", {})


def initialise_vault(master_password: str,
                     username: str = "default",
                     hint: str = "") -> None:
    """
    Create a new empty vault for a user.
    Raises UserAlreadyExistsError if username already registered.
    """
    if not username:
        raise VaultError("Username cannot be empty.")
    if not master_password:
        raise VaultError("Master password cannot be empty.")

    key  = _sanitise(username)
    meta = _load_meta()

    if key in meta.get("users", {}):
        raise UserAlreadyExistsError(
            f"Username '{username}' is already registered. "
            f"Please choose a different name or log in."
        )

    # Create empty vault file
    _save_vault(key, {})

    # Register in meta
    if "users" not in meta:
        meta["users"] = {}
    meta["users"][key] = {
        "display_name": username,
        "hint":         hint,
    }
    _save_meta(meta)


def delete_user_vault(username: str) -> None:
    """Delete a user's vault and remove from meta. Irreversible."""
    key        = _sanitise(username)
    vault_file = _vault_path(key)
    if vault_file.exists():
        os.remove(vault_file)
    meta = _load_meta()
    meta.get("users", {}).pop(key, None)
    _save_meta(meta)


# ── VaultManager ──────────────────────────────────────────────────────────────

class VaultManager:
    """
    Manages encrypted password storage for a single user session.

    Usage (API unchanged from single-user version):
        vm = VaultManager(username="sana")
        vm.unlock(master_password)
        vm.store("gmail", "correct-horse-battery")
        pwd = vm.retrieve("gmail")
        vm.lock()
    """

    def __init__(self, username: str = "default"):
        self._username = _sanitise(username) if username else "default"
        self._master: Optional[str] = None
        self._locked: bool = True

    def unlock(self, master_password: str) -> None:
        """
        Derive the 256-bit AES key from the master password.
        Verifies correctness against the first stored entry.
        Raises WrongMasterPasswordError if the password is wrong.
        """
        from security.crypto import derive_key, decrypt
        import base64

        vault = _load_vault(self._username)

        if not vault:
            # Empty vault — derive key with fresh salt (stored on first write)
            salt         = os.urandom(16)
            self._key    = derive_key(master_password, salt)
            self._salt   = salt
            self._locked = False
            return

        # Verify against the first stored entry
        first_label = next(iter(vault))
        entry = vault[first_label]
        try:
            salt       = base64.b64decode(entry["salt"])
            nonce      = base64.b64decode(entry["nonce"])
            ciphertext = base64.b64decode(entry["ciphertext"])
            key        = derive_key(master_password, salt)
            decrypt(nonce, ciphertext, key)   # raises InvalidTag if wrong key
            self._key    = key
            self._salt   = salt
            self._locked = False
        except WrongMasterPasswordError:
            raise
        except Exception:
            raise WrongMasterPasswordError("Incorrect master password.")

    def lock(self) -> None:
        """Clear key from memory."""
        self._key    = None
        self._locked = True

    def _check_unlocked(self):
        if self._locked or self._key is None:
            raise VaultError("Vault is locked. Call unlock() first.")

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def unlock(self, master_password: str) -> None:
        from security.crypto import decrypt, InvalidTag
        import base64

        vault = _load_vault(self._username)

        if not vault:
            # Empty vault — store master for later use
            self._master  = master_password
            self._locked  = False
            return

    # Verify against first entry
        first_label = next(iter(vault))
        entry = vault[first_label]
        try:
            from security.crypto import decrypt
            result = decrypt(entry, master_password)
            self._master  = master_password
            self._locked  = False
        except InvalidTag:
            raise WrongMasterPasswordError("Incorrect master password.")
        except Exception:
            raise WrongMasterPasswordError("Incorrect master password.")

    def lock(self) -> None:
        self._master  = None
        self._locked  = True

    def _check_unlocked(self):
        if self._locked or not hasattr(self, '_master') or self._master is None:
            raise VaultError("Vault is locked. Call unlock() first.")

    def store(self, label: str, password: str) -> None:
        self._check_unlocked()
        from security.crypto import encrypt

        vault = _load_vault(self._username)
        if label in vault:
            raise EntryAlreadyExistsError(
                f"'{label}' already exists. Use update() to overwrite."
            )

        bundle = encrypt(password, self._master)
        vault[label] = bundle
        _save_vault(self._username, vault)

    def retrieve(self, label: str) -> str:
        self._check_unlocked()
        from security.crypto import decrypt, InvalidTag

        vault = _load_vault(self._username)
        if label not in vault:
            raise EntryNotFoundError(f"No entry found for '{label}'.")

        try:
            return decrypt(vault[label], self._master)
        except InvalidTag:
            raise WrongMasterPasswordError(
                "Decryption failed — wrong master password."
            )

    def update(self, label: str, new_password: str) -> None:
        self._check_unlocked()
        from security.crypto import encrypt

        vault = _load_vault(self._username)
        if label not in vault:
            raise EntryNotFoundError(f"No entry found for '{label}'.")

        vault[label] = encrypt(new_password, self._master)
        _save_vault(self._username, vault)

    def delete(self, label: str) -> None:
        self._check_unlocked()
        vault = _load_vault(self._username)
        if label not in vault:
            raise EntryNotFoundError(f"No entry found for '{label}'.")
        del vault[label]
        _save_vault(self._username, vault)

    def list_labels(self) -> list:
        self._check_unlocked()
        return list(_load_vault(self._username).keys())
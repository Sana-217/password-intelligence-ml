# security/storage.py
"""
Encrypted password vault — read/write operations.

WHAT THIS FILE DOES
────────────────────
Manages a JSON file (vault.json) that stores passwords encrypted
with the crypto primitives from security/crypto.py.

VAULT STRUCTURE (vault.json)
──────────────────────────────
{
  "version": 1,
  "verification_bundle": { salt, nonce, ciphertext },
  "entries": {
    "gmail":   { "salt": "...", "nonce": "...", "ciphertext": "..." },
    "github":  { "salt": "...", "nonce": "...", "ciphertext": "..." }
  }
}

VERIFICATION BUNDLE — why it exists
──────────────────────────────────────
We need a way to check "is this master password correct?" before
attempting to decrypt real entries. The verification_bundle stores
a known plaintext ("PASSGUARD_VERIFY") encrypted with the master
password. On login:
  1. Try to decrypt verification_bundle with the supplied password
  2. If InvalidTag → wrong master password → reject immediately
  3. If decrypts to "PASSGUARD_VERIFY" → correct password → proceed

This means we never attempt to decrypt real vault entries with a
wrong password. It also means we NEVER store the master password
itself — only a bundle that can verify it.

ZERO-KNOWLEDGE PROPERTY (preserved)
──────────────────────────────────────
vault.json contains: salt + nonce + ciphertext for each entry.
Without the master password, all entries are computationally
infeasible to decrypt. The vault file can be safely backed up
to cloud storage — it reveals nothing without the master password.

DESIGN RULE: no crypto logic here
────────────────────────────────────
This file calls crypto.py — it does NOT implement any crypto itself.
All encrypt/decrypt/derive_key calls go through crypto.py only.
This separation means security auditing is focused on one file.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidTag

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from security.crypto import encrypt, decrypt, verify_master_password, CryptoBundle

# ── vault location ────────────────────────────────────────────────────────────
# Default path — can be overridden in tests via VaultManager(vault_path=...)
DEFAULT_VAULT_PATH = ROOT / "vault.json"

# Sentinel plaintext stored in verification_bundle
_VERIFY_SENTINEL = "PASSGUARD_VERIFY"

# Vault file format version — increment if structure changes
_VAULT_VERSION = 1


# ── exceptions ────────────────────────────────────────────────────────────────

class VaultError(Exception):
    """Base exception for all vault operations."""

class VaultNotInitialisedError(VaultError):
    """Raised when vault.json does not exist yet."""

class WrongMasterPasswordError(VaultError):
    """Raised when master password fails verification."""

class EntryNotFoundError(VaultError):
    """Raised when a label does not exist in the vault."""

class EntryAlreadyExistsError(VaultError):
    """Raised when trying to add a label that already exists."""


# ── VaultManager class ────────────────────────────────────────────────────────

class VaultManager:
    """
    Single class for all vault operations.

    Usage pattern:
        vm = VaultManager()

        # First time — create vault
        vm.initialise("myMasterPassword!")

        # Every subsequent use
        vm.unlock("myMasterPassword!")
        vm.store("gmail", "xK9!mPq2")
        pwd = vm.retrieve("gmail")
        vm.lock()

    The vault is "locked" by default — you must call unlock() before
    store() or retrieve(). This mirrors how a real password manager works.
    """

    def __init__(self, vault_path: Optional[Path] = None):
        self._vault_path  = Path(vault_path or DEFAULT_VAULT_PATH)
        self._master_pwd  = None   # only set while unlocked
        self._unlocked    = False

    # ── vault lifecycle ───────────────────────────────────────────────────────

    def initialise(self, master_password: str) -> None:
        """
        Creates a new vault file protected by master_password.
        Raises VaultError if vault already exists — use it only once.

        Args:
            master_password : chosen master password for this vault

        After calling this, the vault is automatically unlocked.
        """
        if not master_password or len(master_password) < 8:
            raise VaultError(
                "Master password must be at least 8 characters."
            )
        if self._vault_path.exists():
            raise VaultError(
                f"Vault already exists at {self._vault_path}.\n"
                f"Delete it manually to start fresh, or call unlock()."
            )

        # Encrypt the sentinel string — this becomes the verification bundle
        verification_bundle = encrypt(_VERIFY_SENTINEL, master_password)

        vault_data = {
            "version":             _VAULT_VERSION,
            "verification_bundle": dict(verification_bundle),
            "entries":             {},
        }
        self._write_vault(vault_data)

        # Auto-unlock after initialisation
        self._master_pwd = master_password
        self._unlocked   = True
        print(f"[vault] Initialised at {self._vault_path}")

    def unlock(self, master_password: str) -> None:
        """
        Verifies master_password against the vault and unlocks it.

        Raises:
            VaultNotInitialisedError : vault.json does not exist
            WrongMasterPasswordError : password failed verification
        """
        if not self._vault_path.exists():
            raise VaultNotInitialisedError(
                f"No vault found at {self._vault_path}.\n"
                f"Call initialise(master_password) to create one."
            )

        vault_data = self._read_vault()
        bundle     = CryptoBundle(**vault_data["verification_bundle"])

        if not verify_master_password(bundle, master_password):
            raise WrongMasterPasswordError(
                "Incorrect master password. Access denied."
            )

        self._master_pwd = master_password
        self._unlocked   = True

    def lock(self) -> None:
        """
        Clears the master password from memory and locks the vault.
        Call this when the user logs out or the session ends.
        """
        self._master_pwd = None
        self._unlocked   = False

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked

    def is_initialised(self) -> bool:
        return self._vault_path.exists()

    # ── CRUD operations ───────────────────────────────────────────────────────

    def store(self, label: str, password: str) -> None:
        """
        Encrypts and stores a password under a label.

        Args:
            label    : identifier for this entry, e.g. "gmail" or "github"
            password : the password to store (plaintext)

        Raises:
            VaultError             : vault is locked
            EntryAlreadyExistsError: label already in vault (use update())
        """
        self._require_unlocked()
        label = _normalise_label(label)

        vault_data = self._read_vault()
        if label in vault_data["entries"]:
            raise EntryAlreadyExistsError(
                f"Entry '{label}' already exists. Use update() to overwrite."
            )

        bundle = encrypt(password, self._master_pwd)
        vault_data["entries"][label] = dict(bundle)
        self._write_vault(vault_data)

    def retrieve(self, label: str) -> str:
        """
        Decrypts and returns the password stored under label.

        Args:
            label : identifier used when storing, e.g. "gmail"

        Returns:
            Plaintext password string

        Raises:
            VaultError        : vault is locked
            EntryNotFoundError: label does not exist
        """
        self._require_unlocked()
        label = _normalise_label(label)

        vault_data = self._read_vault()
        if label not in vault_data["entries"]:
            raise EntryNotFoundError(
                f"No entry found for '{label}'.\n"
                f"Available: {self.list_labels()}"
            )

        bundle = CryptoBundle(**vault_data["entries"][label])
        return decrypt(bundle, self._master_pwd)

    def update(self, label: str, new_password: str) -> None:
        """
        Replaces the stored password for an existing label.
        Generates fresh salt + nonce — does not reuse the old bundle.

        Raises:
            VaultError        : vault is locked
            EntryNotFoundError: label does not exist (use store())
        """
        self._require_unlocked()
        label = _normalise_label(label)

        vault_data = self._read_vault()
        if label not in vault_data["entries"]:
            raise EntryNotFoundError(
                f"No entry found for '{label}'. Use store() to create it."
            )

        bundle = encrypt(new_password, self._master_pwd)
        vault_data["entries"][label] = dict(bundle)
        self._write_vault(vault_data)

    def delete(self, label: str) -> None:
        """
        Permanently removes an entry from the vault.

        Raises:
            VaultError        : vault is locked
            EntryNotFoundError: label does not exist
        """
        self._require_unlocked()
        label = _normalise_label(label)

        vault_data = self._read_vault()
        if label not in vault_data["entries"]:
            raise EntryNotFoundError(
                f"No entry found for '{label}'."
            )

        del vault_data["entries"][label]
        self._write_vault(vault_data)

    def list_labels(self) -> list[str]:
        """
        Returns a sorted list of all stored labels.
        Does NOT require vault to be unlocked — labels are not sensitive.
        (The existence of an entry for 'gmail' is less sensitive than
        the password itself. This matches how real password managers work.)
        """
        if not self._vault_path.exists():
            return []
        vault_data = self._read_vault()
        return sorted(vault_data["entries"].keys())

    def entry_count(self) -> int:
        """Returns the number of stored passwords."""
        return len(self.list_labels())

    def change_master_password(
        self,
        current_password: str,
        new_password: str,
    ) -> None:
        """
        Re-encrypts all vault entries under a new master password.

        This is the correct way to change the master password:
          1. Decrypt every entry with current_password
          2. Re-encrypt every entry with new_password
          3. Replace vault.json atomically

        A naive approach (just updating the verification bundle) would
        leave old entries encrypted under the old key — inaccessible.

        Args:
            current_password : must match the current master password
            new_password     : the new master password (min 8 chars)

        Raises:
            WrongMasterPasswordError : current_password is wrong
            VaultError               : new_password too short
        """
        if not new_password or len(new_password) < 8:
            raise VaultError("New master password must be at least 8 characters.")

        # Verify current password before doing anything destructive
        self.unlock(current_password)

        vault_data = self._read_vault()
        labels     = list(vault_data["entries"].keys())

        # Decrypt all entries with old key
        plaintexts = {}
        for label in labels:
            bundle = CryptoBundle(**vault_data["entries"][label])
            plaintexts[label] = decrypt(bundle, current_password)

        # Re-encrypt all entries with new key
        new_entries = {}
        for label, plaintext in plaintexts.items():
            new_entries[label] = dict(encrypt(plaintext, new_password))

        # Replace verification bundle
        new_verification = dict(encrypt(_VERIFY_SENTINEL, new_password))

        new_vault = {
            "version":             _VAULT_VERSION,
            "verification_bundle": new_verification,
            "entries":             new_entries,
        }
        self._write_vault(new_vault)

        # Update in-memory master password
        self._master_pwd = new_password
        print(f"[vault] Master password changed. {len(labels)} entries re-encrypted.")

    # ── file I/O ──────────────────────────────────────────────────────────────

    def _read_vault(self) -> dict:
        """
        Reads and parses vault.json.
        Uses atomic read — the file is read once and closed immediately.
        """
        try:
            with open(self._vault_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise VaultError(f"vault.json is corrupted: {e}") from e
        except OSError as e:
            raise VaultError(f"Cannot read vault: {e}") from e

    def _write_vault(self, data: dict) -> None:
        """
        Writes vault data to vault.json atomically.

        Atomic write strategy:
          1. Write to vault.json.tmp
          2. Rename .tmp → vault.json  (atomic on POSIX, near-atomic on Windows)
        This prevents a half-written vault if the process is interrupted
        mid-write (power cut, crash). A partial write would corrupt the vault.
        """
        tmp_path = self._vault_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._vault_path)   # atomic rename
        except OSError as e:
            raise VaultError(f"Cannot write vault: {e}") from e
        finally:
            # Clean up tmp file if rename failed
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _require_unlocked(self) -> None:
        """Raises VaultError if vault is not unlocked."""
        if not self._unlocked:
            raise VaultError(
                "Vault is locked. Call unlock(master_password) first."
            )


# ── module-level convenience functions (used by app/app.py) ──────────────────
# These wrap VaultManager for simple one-shot calls from Flask routes.
# They create a new VaultManager each call — stateless, no session leak.

def store_password(master: str, label: str, password: str) -> None:
    """One-call store. Unlocks vault, stores entry, locks vault."""
    vm = VaultManager()
    vm.unlock(master)
    vm.store(label, password)
    vm.lock()


def retrieve_password(master: str, label: str) -> str:
    """One-call retrieve. Unlocks vault, retrieves entry, locks vault."""
    vm = VaultManager()
    vm.unlock(master)
    pwd = vm.retrieve(label)
    vm.lock()
    return pwd


def list_labels() -> list[str]:
    """Returns all stored labels without unlocking."""
    return VaultManager().list_labels()


def initialise_vault(master: str) -> None:
    """Creates a new vault. Fails if vault already exists."""
    VaultManager().initialise(master)


def vault_exists() -> bool:
    """Returns True if vault.json exists."""
    return DEFAULT_VAULT_PATH.exists()


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalise_label(label: str) -> str:
    """
    Lowercases and strips whitespace from a label.
    Ensures "Gmail", "GMAIL", "gmail " all map to the same entry.
    """
    if not label or not label.strip():
        raise VaultError("Label cannot be empty.")
    return label.strip().lower()


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=" * 56)
    print("  security/storage.py — self test")
    print("=" * 56)

    # Use a temp file so we don't pollute the real vault
    with tempfile.TemporaryDirectory() as tmpdir:
        test_vault = Path(tmpdir) / "test_vault.json"
        vm         = VaultManager(vault_path=test_vault)
        master     = "TestMaster@2024!"
        all_pass   = True

        def check(label, result, expected=True):
            global all_pass
            status = "PASS" if result == expected else "FAIL"
            if result != expected:
                all_pass = False
            print(f"  {status}  {label}")

        # ── initialise
        print("\n── Initialise vault ──────────────────────────────────")
        vm.initialise(master)
        check("vault file created",      test_vault.exists())
        check("vault is unlocked",       vm.is_unlocked)
        check("entry count = 0",         vm.entry_count() == 0)

        # ── store and retrieve
        print("\n── Store + retrieve ──────────────────────────────────")
        vm.store("gmail",  "correct-horse-battery")
        vm.store("github", "xK9!mPq2")
        vm.store("bank",   "Mumbai@2019!Chai")

        check("entry count = 3",         vm.entry_count() == 3)
        check("retrieve gmail",          vm.retrieve("gmail")  == "correct-horse-battery")
        check("retrieve github",         vm.retrieve("github") == "xK9!mPq2")
        check("retrieve bank",           vm.retrieve("bank")   == "Mumbai@2019!Chai")
        check("label normalisation",     vm.retrieve("GMAIL")  == "correct-horse-battery")

        # ── list labels
        print("\n── List labels ───────────────────────────────────────")
        labels = vm.list_labels()
        check("labels sorted",           labels == ["bank", "github", "gmail"])

        # ── update
        print("\n── Update ────────────────────────────────────────────")
        vm.update("gmail", "newGmailPassword!99")
        check("updated gmail",           vm.retrieve("gmail") == "newGmailPassword!99")

        # ── delete
        print("\n── Delete ────────────────────────────────────────────")
        vm.delete("bank")
        check("entry count after delete", vm.entry_count() == 2)
        try:
            vm.retrieve("bank")
            check("deleted entry raises",  False)
        except EntryNotFoundError:
            check("deleted entry raises EntryNotFoundError", True)

        # ── lock / unlock cycle
        print("\n── Lock / unlock cycle ───────────────────────────────")
        vm.lock()
        check("locked after lock()",     not vm.is_unlocked)
        try:
            vm.retrieve("gmail")
            check("retrieve while locked raises", False)
        except VaultError:
            check("retrieve while locked raises VaultError", True)

        vm.unlock(master)
        check("unlocked again",          vm.is_unlocked)
        check("data intact after cycle", vm.retrieve("gmail") == "newGmailPassword!99")

        # ── wrong master password
        print("\n── Wrong master password ─────────────────────────────")
        vm.lock()
        try:
            vm.unlock("wrongPassword123")
            check("wrong password raises", False)
        except WrongMasterPasswordError:
            check("wrong password raises WrongMasterPasswordError", True)
        check("still locked after fail", not vm.is_unlocked)

        # ── change master password
        print("\n── Change master password ────────────────────────────")
        vm.unlock(master)
        new_master = "NewMaster@9999!"
        vm.change_master_password(master, new_master)
        vm.lock()

        try:
            vm.unlock(master)
            check("old password rejected", False)
        except WrongMasterPasswordError:
            check("old password rejected", True)

        vm.unlock(new_master)
        check("new password works",      vm.is_unlocked)
        check("data intact after rekey", vm.retrieve("gmail") == "newGmailPassword!99")
        vm.lock()

        # ── duplicate label
        print("\n── Duplicate label ───────────────────────────────────")
        vm.unlock(new_master)
        try:
            vm.store("gmail", "anotherPassword")
            check("duplicate label raises", False)
        except EntryAlreadyExistsError:
            check("duplicate raises EntryAlreadyExistsError", True)

        # ── vault.json structure inspection
        print("\n── vault.json structure ──────────────────────────────")
        with open(test_vault) as f:
            raw = json.load(f)
        check("has version field",            "version" in raw)
        check("has verification_bundle",      "verification_bundle" in raw)
        check("has entries",                  "entries" in raw)
        check("no plaintext passwords",       "correct-horse-battery" not in str(raw))
        check("no master password stored",    master not in str(raw))

    print()
    print("=" * 56)
    print(f"  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 56)
    print("\n  Next step: python generator/password_gen.py")
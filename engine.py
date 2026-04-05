# engine.py
"""
PassGuard Engine — single orchestration point for all modules.

WHY THIS FILE EXISTS
─────────────────────
Your project documentation describes an "engine" that coordinates
all system components. This file fulfils that promise.

It is the ONLY file that imports across all packages.
All other modules import downward only (into utils/preprocessing).
This prevents circular imports and makes the system auditable.

USAGE
──────
From CLI or any script:
    from engine import PassGuardEngine
    engine = PassGuardEngine()
    result = engine.generate("passphrase")
    analysis = engine.analyze("correct-horse-battery")
    engine.store("gmail", "correct-horse-battery", "myMasterPwd!")

From app/app.py:
    The Flask app calls individual modules directly for performance.
    engine.py is used for CLI and batch operations.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── imports from all packages ─────────────────────────────────────────────────
from generator.password_gen          import generate_best, score_password
from generator.memory_aid            import generate_memory_aids
from generator.passphrase_transformer import transform_passphrase
from security.storage                import (
    VaultManager,
    vault_exists,
    WrongMasterPasswordError,
    EntryNotFoundError,
    VaultError,
)
from preprocessing.feature_extraction import extract_features


class PassGuardEngine:
    """
    Unified interface to all PassGuard capabilities.

    Instantiate once, use throughout a session.

    Example:
        engine = PassGuardEngine()

        # Generate
        result = engine.generate(mode="passphrase", n_words=4)
        print(result["best"]["password"])

        # Analyze
        scores = engine.analyze("correct-horse-battery")
        print(scores["strength_label"], scores["memorability_label"])

        # Transform
        result = engine.transform("my dog name is Bruno", year=2024)
        print(result["password"])

        # Vault
        engine.init_vault("myMasterPwd!")
        engine.store("gmail", "correct-horse-battery", "myMasterPwd!")
        pwd = engine.retrieve("gmail", "myMasterPwd!")
    """

    def __init__(self):
        self._vault = VaultManager()

    # ── generation ────────────────────────────────────────────────────────────

    def generate(
        self,
        mode:         str = "passphrase",
        n_candidates: int = 5,
        **kwargs,
    ) -> dict:
        """
        Generates passwords and ranks them by combined ML score.

        Args:
            mode         : "passphrase" | "pattern" | "random"
            n_candidates : number of candidates to generate and rank
            **kwargs     : mode-specific options
                           passphrase: n_words, separator, capitalise
                           pattern:    pattern (str)
                           random:     length, use_uppercase, use_digits, use_special

        Returns:
            { "best": {...}, "all": [...], "mode": str }
        """
        return generate_best(mode=mode, n_candidates=n_candidates, **kwargs)

    def transform(self, phrase: str, year: int = None, **kwargs) -> dict:
        """
        Transforms a user phrase into a secure memorable password.

        Args:
            phrase : e.g. "my dog name is Bruno"
            year   : optional year to append e.g. 2024

        Returns:
            { "password": str, "pipeline": dict, "candidates": list, ... }
        """
        return transform_passphrase(phrase, year_hint=year, **kwargs)

    # ── analysis ──────────────────────────────────────────────────────────────

    def analyze(self, password: str) -> dict:
        """
        Scores a password through both ML models.

        Returns:
            {
              "password", "strength_label", "memorability_label",
              "combined_score", "strength_proba", "memorability_proba"
            }
        """
        return score_password(password)

    def features(self, password: str) -> dict:
        """
        Extracts all 15 linguistic and entropy features from a password.
        Useful for understanding why a password scored the way it did.

        Returns:
            { "length": int, "entropy": float, "syllable_count": int, ... }
        """
        return extract_features(password)

    def memory_aids(self, password: str) -> dict:
        """
        Generates all four cognitive memory techniques for a password.

        Returns:
            { "chunking": {...}, "phonetic": {...},
              "acronym": {...}, "story": {...} }
        """
        return generate_memory_aids(password)

    # ── vault ─────────────────────────────────────────────────────────────────

    def init_vault(self, master_password: str) -> None:
        """Creates a new vault. Fails if one already exists."""
        self._vault.initialise(master_password)

    def store(self, label: str, password: str, master_password: str) -> None:
        """Encrypts and stores a password under a label."""
        self._vault.unlock(master_password)
        self._vault.store(label, password)
        self._vault.lock()

    def retrieve(self, label: str, master_password: str) -> str:
        """Decrypts and returns a stored password."""
        self._vault.unlock(master_password)
        pwd = self._vault.retrieve(label)
        self._vault.lock()
        return pwd

    def delete(self, label: str, master_password: str) -> None:
        """Deletes a stored password entry."""
        self._vault.unlock(master_password)
        self._vault.delete(label)
        self._vault.lock()

    def list_labels(self) -> list:
        """Returns all stored labels without unlocking."""
        return self._vault.list_labels()

    def vault_exists(self) -> bool:
        """Returns True if vault.json exists."""
        return vault_exists()

    # ── combined workflow ─────────────────────────────────────────────────────

    def generate_and_store(
        self,
        label:          str,
        master_password: str,
        mode:           str = "passphrase",
        **kwargs,
    ) -> dict:
        """
        Generates the best password AND stores it in one call.
        Convenience method for the most common workflow.

        Returns the full generation result including scores.
        """
        result = self.generate(mode=mode, **kwargs)
        best   = result["best"]["password"]
        self.store(label, best, master_password)
        return result

    def full_report(self, password: str) -> dict:
        """
        Runs complete analysis on a password:
          - ML scores (strength + memorability)
          - All 15 features
          - All 4 memory aids

        Returns combined dict — useful for detailed reporting.
        """
        scores = self.analyze(password)
        feats  = self.features(password)
        aids   = self.memory_aids(password)

        return {
            "password":     password,
            "ml_scores":    scores,
            "features":     feats,
            "memory_aids":  aids,
        }


# ── quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  PassGuard Engine — demo")
    print("=" * 60)

    engine = PassGuardEngine()

    # 1. Generate
    print("\n[1] Generating passphrase...")
    result = engine.generate(mode="passphrase", n_words=4)
    best   = result["best"]
    print(f"    Password    : {best['password']}")
    print(f"    Strength    : {best['strength_label']}")
    print(f"    Memorability: {best['memorability_label']}")
    print(f"    Score       : {best['combined_score']}")

    # 2. Transform
    print("\n[2] Transforming phrase...")
    t = engine.transform("my dog name is Bruno", year=2024)
    print(f"    Input    : my dog name is Bruno")
    print(f"    Output   : {t['password']}")
    print(f"    Pipeline : {t['pipeline']['core']} + {t['pipeline']['numeric_suffix']}{t['pipeline']['special_char']}")

    # 3. Analyze
    print("\n[3] Analyzing passwords...")
    for pwd in ["123456", "correct-horse-battery", "Mumbai@2019!Chai"]:
        s = engine.analyze(pwd)
        print(f"    {pwd:<28} {s['strength_label']:<8} {s['memorability_label']}")

    # 4. Full report
    print("\n[4] Full report for 'correct-horse-battery'...")
    report = engine.full_report("correct-horse-battery")
    print(f"    Strength     : {report['ml_scores']['strength_label']}")
    print(f"    Memorability : {report['ml_scores']['memorability_label']}")
    print(f"    Syllables    : {report['features']['syllable_count']}")
    print(f"    Chunking     : {report['memory_aids']['chunking']['display']}")

    print("\n" + "=" * 60)
    print("  Engine demo complete")
    print("=" * 60)
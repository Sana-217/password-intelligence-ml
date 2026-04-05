# generator/password_gen.py
"""
Password generator — the core feature of the entire project.

WHAT THIS FILE DOES
────────────────────
Generates passwords using three modes, then scores every candidate
through both trained ML models, and returns the best result.

THREE GENERATION MODES
───────────────────────
  1. PASSPHRASE  — picks N random words from EFF wordlist
                   e.g. "correct-horse-battery-staple"
                   High entropy, highly memorable, easy to type

  2. PATTERN     — builds password from a user-defined template
                   e.g. pattern "Www#ddd!" → "Cat#847!"
                   Predictable structure, memorable, meets site rules

  3. RANDOM      — cryptographically random characters from charset
                   e.g. "xK9!mPq2Tz@"
                   Highest entropy, least memorable — for high-security use

WHY ML MODELS ARE USED HERE
─────────────────────────────
Most password generators just produce a random string and stop.
This system generates MULTIPLE candidates per mode, scores each
one through the strength and memorability models, and returns
the candidate with the best combined score.

This is the key innovation of your project:
  generation + ML evaluation + ranking → best candidate returned

The user gets a password that is both strong AND memorable,
not just randomly strong.

IMPORTS FROM YOUR BUILT FILES
───────────────────────────────
  preprocessing/feature_extraction.py → extract_feature_vector()
  models/train_strength.py            → load_strength_model()
  models/train_memorability.py        → load_memorability_model()
"""

import os
import re
import sys
import secrets
import string
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.train_strength      import load_strength_model
from models.train_memorability  import load_memorability_model
from preprocessing.feature_extraction import (
    extract_feature_vector,
    get_word_count,
    get_syllable_count,
)

# ── paths ─────────────────────────────────────────────────────────────────────
EFF_WORDLIST_PATH = ROOT / "dataset" / "wordlist.txt"

# ── pattern tokens ────────────────────────────────────────────────────────────
# Used in PATTERN mode. Each character in the pattern string maps to a rule:
#   W = random uppercase letter
#   w = random lowercase letter
#   d = random digit
#   s = random special character  (!@#$%^&*)
#   * = random character from any category
#   any other char = used as-is (literal)
#
# Example patterns:
#   "Wwww#ddd!"   → "Cats#847!"
#   "Www-Www-ddd" → "Sky-Box-294"
#   "WwwdWwwd!!"  → "Cat4Dog8!!"

PATTERN_TOKENS = {
    "W": string.ascii_uppercase,
    "w": string.ascii_lowercase,
    "d": string.digits,
    "s": "!@#$%^&*",
    "*": string.ascii_letters + string.digits + "!@#$%^&*",
}

# ── scoring weights ───────────────────────────────────────────────────────────
# When ranking candidates, we combine strength and memorability scores.
# These weights reflect the project's core thesis:
#   security matters slightly more than memorability, but both count.
# Adjust these to shift the balance.
STRENGTH_WEIGHT      = 0.6   # 60% weight on strength
MEMORABILITY_WEIGHT  = 0.4   # 40% weight on memorability


# ── model loader (lazy, cached) ───────────────────────────────────────────────
# Models are loaded once on first use, then cached.
# Loading a pickle file takes ~0.1s — we don't want that on every call.

_strength_model     = None
_memorability_model = None


def _get_models():
    """
    Loads both ML models on first call, returns cached versions after.
    Raises FileNotFoundError with a clear message if models aren't trained yet.
    """
    global _strength_model, _memorability_model
    if _strength_model is None:
        try:
            _strength_model, _   = load_strength_model()
            _memorability_model, _ = load_memorability_model()
        except FileNotFoundError:
            raise FileNotFoundError(
                "\n[password_gen] Trained models not found.\n"
                "Run these first:\n"
                "  python models/train_strength.py\n"
                "  python models/train_memorability.py"
            )
    return _strength_model, _memorability_model


# ── wordlist loader ───────────────────────────────────────────────────────────

_wordlist = None

def _get_wordlist() -> list[str]:
    """
    Loads the EFF wordlist on first call, caches it after.
    Falls back to NLTK words if wordlist.txt is missing.
    """
    global _wordlist
    if _wordlist is not None:
        return _wordlist

    if EFF_WORDLIST_PATH.exists():
        with open(EFF_WORDLIST_PATH, encoding="utf-8") as f:
            # EFF wordlist format: "11111\tcorrect\n" (dice roll + tab + word)
            # Plain wordlist format: one word per line
            words = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Handle tab-separated EFF format
                parts = line.split("\t")
                word = parts[-1].strip().lower()
                if len(word) >= 3:
                    words.append(word)
            _wordlist = words
    else:
        # Fallback: use NLTK words (always available after Step 1)
        import nltk
        try:
            from nltk.corpus import words as nltk_words
            nltk_words.words()
        except LookupError:
            nltk.download("words", quiet=True)
            from nltk.corpus import words as nltk_words
        _wordlist = [
            w.lower() for w in nltk_words.words()
            if 3 <= len(w) <= 8 and w.isalpha()
        ]

    if not _wordlist:
        raise RuntimeError("No wordlist available. Add dataset/wordlist.txt")

    return _wordlist


# ── ML scoring ────────────────────────────────────────────────────────────────

def score_password(password: str) -> dict:
    """
    Runs a password through both ML models and returns a structured result.
    Includes a hard rule override to fix false positives on random strings.
    """
    str_model, mem_model = _get_models()
    features = extract_feature_vector(password)

    # Strength prediction
    str_int        = int(str_model.predict([features])[0])
    str_proba      = str_model.predict_proba([features])[0]
    str_confidence = float(str_proba[2])
    str_labels     = {0: "weak", 1: "medium", 2: "strong"}

    # Memorability prediction
    mem_int        = int(mem_model.predict([features])[0])
    mem_proba      = mem_model.predict_proba([features])[0]
    mem_confidence = float(mem_proba[1])
    mem_labels     = {0: "not_memorable", 1: "memorable"}

    # Hard rule: if zero real words AND fewer than 2 syllables
    # → cannot be memorable regardless of what model predicts
    # Fixes false positives on purely random strings like "7K+NAAY61OKZRN#e"
    if get_word_count(password) == 0 and get_syllable_count(password) < 3:
        mem_int        = 0
        mem_confidence = 0.1

    combined = (
        str_confidence * STRENGTH_WEIGHT +
        mem_confidence * MEMORABILITY_WEIGHT
    )

    return {
        "password":           password,
        "strength_label":     str_labels[str_int],
        "strength_int":       str_int,
        "strength_proba":     round(str_confidence, 4),
        "memorability_label": mem_labels[mem_int],
        "memorability_int":   mem_int,
        "memorability_proba": round(mem_confidence, 4),
        "combined_score":     round(combined, 4),
    }


# ── mode 1: passphrase ────────────────────────────────────────────────────────

def generate_passphrase(
    n_words:    int  = 4,
    separator:  str  = "-",
    capitalise: bool = True,
) -> str:
    """
    Generates a passphrase by picking n_words random words from the wordlist.

    Uses secrets.choice() — cryptographically secure random selection.
    secrets module uses os.urandom() internally, unlike random.choice().

    Args:
        n_words   : number of words (default 4 — ~51 bits of entropy)
        separator : character between words (default "-")
        capitalise: capitalise first letter of each word (default True)

    Returns:
        Passphrase string e.g. "Correct-Horse-Battery-Staple"

    Entropy calculation:
        EFF large wordlist: 7776 words = 12.9 bits per word
        4 words: 4 × 12.9 = 51.6 bits  (strong)
        5 words: 5 × 12.9 = 64.6 bits  (very strong)
    """
    wordlist = _get_wordlist()
    words    = [secrets.choice(wordlist) for _ in range(n_words)]
    if capitalise:
        words = [w.capitalize() for w in words]
    return separator.join(words)


# ── mode 2: pattern-based ─────────────────────────────────────────────────────

def generate_from_pattern(pattern: str) -> str:
    """
    Generates a password by expanding a pattern string.

    Token map:
        W → random uppercase letter
        w → random lowercase letter
        d → random digit  (0–9)
        s → random special char  (!@#$%^&*)
        * → random char from all categories
        anything else → literal character

    Args:
        pattern : template string e.g. "Wwww-ddd-s"

    Returns:
        Expanded password e.g. "Cats-847-!"

    Examples:
        "Wwww#ddd"   → "Bear#294"
        "Www-Www-dd" → "Sky-Box-47"
        "WwwdWwwd!!" → "Cat4Dog8!!"
        "****dd**"   → "xK9!28mP"
    """
    if not pattern:
        raise ValueError("Pattern cannot be empty.")

    result = []
    for char in pattern:
        if char in PATTERN_TOKENS:
            result.append(secrets.choice(PATTERN_TOKENS[char]))
        else:
            result.append(char)   # literal — hyphens, dots, @ signs etc.
    return "".join(result)


# ── mode 3: random ────────────────────────────────────────────────────────────

def generate_random(
    length:           int  = 16,
    use_uppercase:    bool = True,
    use_digits:       bool = True,
    use_special:      bool = True,
) -> str:
    """
    Generates a cryptographically random password from a chosen charset.

    Uses secrets.choice() in a loop — each character is independently
    and uniformly random. This is the correct way to generate random
    passwords in Python (not random.choice, not uuid).

    Guarantees at least one character from each enabled category,
    then fills the rest randomly. This prevents degenerate outputs
    like "aaaaaaaaaaaaaaa" which technically satisfy the charset
    rules but fail in practice.

    Args:
        length        : total character count (min 8)
        use_uppercase : include A–Z
        use_digits    : include 0–9
        use_special   : include !@#$%^&*()-_=+

    Returns:
        Random password string of exactly `length` characters
    """
    if length < 8:
        raise ValueError("Password length must be at least 8.")

    charset  = string.ascii_lowercase
    required = []

    if use_uppercase:
        charset  += string.ascii_uppercase
        required.append(secrets.choice(string.ascii_uppercase))
    if use_digits:
        charset  += string.digits
        required.append(secrets.choice(string.digits))
    if use_special:
        special   = "!@#$%^&*()-_=+"
        charset  += special
        required.append(secrets.choice(special))

    # Fill remaining positions
    remaining = [secrets.choice(charset) for _ in range(length - len(required))]

    # Shuffle so guaranteed chars aren't always at the start
    pool = required + remaining
    secrets.SystemRandom().shuffle(pool)
    return "".join(pool)


# ── main generator: generates + ranks candidates ──────────────────────────────

def generate_best(
    mode:        str            = "passphrase",
    n_candidates: int           = 5,
    # passphrase options
    n_words:     int            = 4,
    separator:   str            = "-",
    capitalise:  bool           = True,
    # pattern options
    pattern:     Optional[str]  = None,
    # random options
    length:      int            = 16,
    use_uppercase: bool         = True,
    use_digits:    bool         = True,
    use_special:   bool         = True,
) -> dict:
    """
    Generates n_candidates passwords, scores each with both ML models,
    and returns the one with the highest combined score.

    This is the function called by app/app.py and cli.py.

    Args:
        mode         : "passphrase" | "pattern" | "random"
        n_candidates : how many to generate and rank (default 5)
        ...          : mode-specific options (see individual generators)

    Returns:
        {
          "best": { score_dict for the winning password },
          "all":  [ score_dicts for all candidates, sorted best→worst ],
          "mode": "passphrase",
        }

    Example:
        result = generate_best(mode="passphrase", n_words=4)
        print(result["best"]["password"])
        # → "Correct-Horse-Battery-Staple"
        print(result["best"]["combined_score"])
        # → 0.871
        print(result["best"]["strength_label"])
        # → "strong"
        print(result["best"]["memorability_label"])
        # → "memorable"
    """
    mode = mode.lower().strip()
    if mode not in ("passphrase", "pattern", "random"):
        raise ValueError(
            f"Unknown mode '{mode}'. Use: 'passphrase', 'pattern', or 'random'."
        )

    if mode == "pattern" and not pattern:
        raise ValueError(
            "Pattern mode requires a pattern string. "
            "Example: pattern='Wwww-ddd-s'"
        )

    # Generate n_candidates passwords using the chosen mode
    candidates = []
    for _ in range(n_candidates):
        if mode == "passphrase":
            pwd = generate_passphrase(
                n_words=n_words,
                separator=separator,
                capitalise=capitalise,
            )
        elif mode == "pattern":
            pwd = generate_from_pattern(pattern)
        else:
            pwd = generate_random(
                length=length,
                use_uppercase=use_uppercase,
                use_digits=use_digits,
                use_special=use_special,
            )
        candidates.append(pwd)

    # Score all candidates
    scored = [score_password(pwd) for pwd in candidates]

    # Sort by combined_score descending
    scored.sort(key=lambda x: x["combined_score"], reverse=True)

    return {
        "best": scored[0],
        "all":  scored,
        "mode": mode,
    }


# ── convenience wrappers (used by app/app.py routes) ─────────────────────────

def generate_passphrase_best(**kwargs) -> dict:
    """Shortcut: generate_best(mode='passphrase', ...)"""
    return generate_best(mode="passphrase", **kwargs)


def generate_pattern_best(pattern: str, **kwargs) -> dict:
    """Shortcut: generate_best(mode='pattern', pattern=pattern, ...)"""
    return generate_best(mode="pattern", pattern=pattern, **kwargs)


def generate_random_best(**kwargs) -> dict:
    """Shortcut: generate_best(mode='random', ...)"""
    return generate_best(mode="random", **kwargs)


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  generator/password_gen.py — self test")
    print("=" * 64)

    # ── test individual generators (no ML needed)
    print("\n── Raw generators (no ML) ────────────────────────────────────")

    pp = generate_passphrase(n_words=4)
    print(f"  Passphrase (4 words):   {pp}")
    assert "-" in pp or len(pp) > 10, "Passphrase too short"

    pat = generate_from_pattern("Wwww-ddd-s")
    print(f"  Pattern (Wwww-ddd-s):   {pat}")
    assert len(pat) == 10, f"Expected 10 chars, got {len(pat)}"

    rnd = generate_random(length=16)
    print(f"  Random (16 chars):      {rnd}")
    assert len(rnd) == 16, f"Expected 16 chars, got {len(rnd)}"

    # ── test ML scoring
    print("\n── ML scoring ────────────────────────────────────────────────")
    test_passwords = [
        "123456",
        "correct-horse-battery",
        "xK9!mPq2Tz@W",
        "Mumbai@2019!Chai",
    ]
    print(f"  {'Password':<28} {'Strength':<10} {'Memorability':<16} {'Score':>6}")
    print(f"  {'-'*64}")
    for pwd in test_passwords:
        s = score_password(pwd)
        print(
            f"  {s['password']:<28} "
            f"{s['strength_label']:<10} "
            f"{s['memorability_label']:<16} "
            f"{s['combined_score']:>6.3f}"
        )

    # ── test generate_best() all three modes
    print("\n── generate_best() — all three modes ────────────────────────")

    for mode, kwargs in [
        ("passphrase", {"n_words": 4, "n_candidates": 5}),
        ("pattern",    {"pattern": "Wwww-ddd-s", "n_candidates": 5}),
        ("random",     {"length": 16, "n_candidates": 5}),
    ]:
        result = generate_best(mode=mode, **kwargs)
        best   = result["best"]
        print(f"\n  Mode: {mode}")
        print(f"    Best password  : {best['password']}")
        print(f"    Strength       : {best['strength_label']}  "
              f"(p={best['strength_proba']:.3f})")
        print(f"    Memorability   : {best['memorability_label']}  "
              f"(p={best['memorability_proba']:.3f})")
        print(f"    Combined score : {best['combined_score']:.3f}")
        print(f"    All candidates :")
        for i, c in enumerate(result["all"], 1):
            marker = " ← best" if i == 1 else ""
            print(f"      {i}. {c['password']:<30} score={c['combined_score']:.3f}{marker}")

    print("\n" + "=" * 64)
    print("  ALL TESTS PASSED")
    print("=" * 64)
    print("\n  Next step: python generator/memory_aid.py")
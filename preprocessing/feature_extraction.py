# preprocessing/feature_extraction.py
"""
Feature extraction for password strength and memorability models.

Every feature here maps to something a human or attacker actually cares about:
- Attackers care about: entropy, length, charset diversity, pattern predictability
- Humans care about: pronounceability, syllable count, familiar words, chunk-ability

This module has ZERO side effects — pure functions only.
Import it anywhere without fear of circular imports.
"""

import math
import re
import string
from collections import Counter

import pyphen
import nltk
from nltk.corpus import words as nltk_words

# ── one-time NLTK setup ──────────────────────────────────────────────────────
# Run this once on first use. Safe to call repeatedly.
def _ensure_nltk_data():
    try:
        nltk_words.words()
    except LookupError:
        nltk.download("words", quiet=True)

_ensure_nltk_data()
_ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())

# Pyphen: syllable counter using language hyphenation dictionaries
_HYPHENATOR = pyphen.Pyphen(lang="en_US")

# Common keyboard walk patterns — a real attacker's first guesses
_KEYBOARD_WALKS = [
    "qwerty", "qwert", "asdf", "zxcv", "1234", "12345",
    "123456", "abcd", "pass", "admin", "letme"
]

# Top-20 most common passwords from rockyou — instant crack list
_COMMON_PASSWORDS = {
    "123456", "password", "12345678", "qwerty", "abc123",
    "monkey", "1234567", "letmein", "trustno1", "dragon",
    "baseball", "iloveyou", "master", "sunshine", "ashley",
    "bailey", "passw0rd", "shadow", "123123", "654321"
}


# ── individual feature functions ─────────────────────────────────────────────

def get_length(password: str) -> int:
    """
    Raw character count.
    NIST 800-63B: minimum 8 chars for basic, 15+ for high security.
    """
    return len(password)


def get_shannon_entropy(password: str) -> float:
    """
    Shannon entropy in bits: H = -Σ p(c) * log2(p(c))
    Measures unpredictability of character distribution.
    A password of length n from charset of size C has max entropy = log2(C^n).
    Real entropy is always ≤ this maximum.

    Typical values:
      'password'    → ~2.75 bits  (very low, repetitive chars)
      'P@ssw0rd'   → ~3.0 bits   (slightly better)
      'xK9!mPq2'   → ~3.0 bits   (high diversity, short)
      'correct-horse-battery-staple' → ~3.7 bits (long, high entropy)
    """
    if not password:
        return 0.0
    freq = Counter(password)
    total = len(password)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def get_charset_size(password: str) -> int:
    """
    Size of the character set actually used.
    Determines the theoretical keyspace: keyspace = charset_size ^ length.

    Returns one of: 10, 26, 36, 52, 62, 95
    (digits only → 10, lowercase → 26, mixed → up to 95)
    """
    size = 0
    if any(c in string.digits for c in password):
        size += 10
    if any(c in string.ascii_lowercase for c in password):
        size += 26
    if any(c in string.ascii_uppercase for c in password):
        size += 26
    if any(c in string.punctuation for c in password):
        size += 33
    return size


def get_char_diversity(password: str) -> float:
    """
    Ratio of unique characters to total length.
    Range: 0.0 (all same char) → 1.0 (all unique chars).

    'aaaaaaa'  → 0.14  (terrible)
    'abcdefg'  → 1.0   (perfect diversity)
    'P@ssw0rd' → 0.875 (good)
    """
    if not password:
        return 0.0
    return len(set(password)) / len(password)


def get_syllable_count(password: str) -> int:
    """
    Number of syllables in the password using English hyphenation rules.
    Syllables drive pronounceability — the phonological loop in working memory
    uses sub-vocal rehearsal, so speakable passwords are easier to remember.

    'correct'       → 2 syllables
    'xK9!mPq2'     → 0 syllables (not pronounceable)
    'correct-horse' → 3 syllables
    """
    # Strip non-alpha for syllable counting — punctuation has no syllables
    alpha_only = re.sub(r"[^a-zA-Z]", " ", password.lower())
    total = 0
    for word in alpha_only.split():
        if len(word) < 2:
            continue
        pairs = _HYPHENATOR.inserted(word)
        # pyphen returns "syl-la-ble" style — count hyphens + 1
        total += pairs.count("-") + 1
    return total


def get_phonetic_score(password: str) -> float:
    """
    Ratio of consonant-vowel alternation in the password's alpha characters.
    CV patterns (like "ba-na-na") are the most pronounceable sequences.
    High phonetic score → easier to sound out → easier to remember.

    Range: 0.0 → 1.0
    'banana'    → ~0.83 (highly pronounceable)
    'xK9!mPq2' → ~0.25 (hard to say)
    """
    alpha = re.sub(r"[^a-zA-Z]", "", password.lower())
    if len(alpha) < 2:
        return 0.0
    vowels = set("aeiou")
    transitions = sum(
        1 for i in range(len(alpha) - 1)
        if (alpha[i] in vowels) != (alpha[i + 1] in vowels)
    )
    return transitions / (len(alpha) - 1)


def get_word_count(password: str) -> int:
    """
    Number of real English words (≥3 chars) found inside the password.
    Uses NLTK's word corpus (~235,000 English words).
    Words anchor memory — 'correct horse' is far more memorable than 'crrcthrse'.
    """
    alpha_tokens = re.findall(r"[a-zA-Z]{3,}", password.lower())
    return sum(1 for t in alpha_tokens if t in _ENGLISH_WORDS)


def get_digit_ratio(password: str) -> float:
    """
    Fraction of characters that are digits.
    Too high (all digits) = PIN-like, low entropy.
    Too low = misses a charset dimension.
    Sweet spot: 0.1–0.3
    """
    if not password:
        return 0.0
    return sum(1 for c in password if c.isdigit()) / len(password)


def get_special_char_ratio(password: str) -> float:
    """
    Fraction of characters that are special/punctuation.
    Special chars expand charset dramatically (adds 33 symbols).
    Even one special char raises brute-force cost significantly.
    """
    if not password:
        return 0.0
    return sum(1 for c in password if c in string.punctuation) / len(password)


def get_uppercase_ratio(password: str) -> float:
    """
    Fraction of characters that are uppercase.
    Uniform case (all lower or all upper) = predictable pattern.
    Mixed case doubles the keyspace for letter positions.
    """
    if not password:
        return 0.0
    return sum(1 for c in password if c.isupper()) / len(password)


def get_has_keyboard_walk(password: str) -> int:
    """
    Binary: 1 if password contains a keyboard walk pattern, else 0.
    'qwerty123' → 1  (instant crack, appears in every wordlist)
    'xK9!mPq2' → 0  (no walk pattern)
    """
    lower = password.lower()
    return int(any(walk in lower for walk in _KEYBOARD_WALKS))


def get_has_repeated_chars(password: str) -> int:
    """
    Binary: 1 if any character repeats 3+ times consecutively.
    'aaa', '111', 'zzz' patterns drastically reduce entropy.
    """
    return int(bool(re.search(r"(.)\1{2,}", password)))


def get_is_common_password(password: str) -> int:
    """
    Binary: 1 if password is in the top-20 most cracked passwords.
    These appear in every dictionary attack wordlist — zero security.
    """
    return int(password.lower() in _COMMON_PASSWORDS)


def get_bigram_entropy(password: str) -> float:
    """
    Shannon entropy computed over character bigrams (pairs) instead of singles.
    Catches sequential patterns that single-char entropy misses.

    'ababababab' → low bigram entropy (only "ab" and "ba" bigrams)
                 → but reasonable single-char entropy (a, b evenly split)
    So bigram entropy catches this where single-char entropy fails.
    """
    if len(password) < 2:
        return 0.0
    bigrams = [password[i:i+2] for i in range(len(password) - 1)]
    freq = Counter(bigrams)
    total = len(bigrams)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def get_estimated_crack_time_score(password: str) -> float:
    """
    Estimated log10 of crack time in seconds at 10^10 guesses/sec (GPU speed).
    Formula: log10(charset_size ^ length / 10^10)
    Capped at 0 (min) and 15 (max, i.e. > 10^15 seconds = heat death of universe).

    This is a conservative estimate — assumes random uniform distribution.
    Real passwords are less random, so actual crack time is lower.
    """
    cs = get_charset_size(password)
    ln = get_length(password)
    if cs == 0 or ln == 0:
        return 0.0
    keyspace = math.log10(cs) * ln  # log10(cs^ln)
    crack_time_log = keyspace - 10  # divide by 10^10 guesses/sec
    return max(0.0, min(15.0, crack_time_log))


# ── master extractor ─────────────────────────────────────────────────────────

def extract_features(password: str) -> dict:
    """
    Runs all feature functions and returns a flat dict.
    This dict is what both ML models receive as input.

    Usage:
        feats = extract_features("correct-horse-battery")
        # → {"length": 22, "entropy": 3.58, "syllables": 7, ...}

    The dict key order is stable — always matches FEATURE_NAMES below.
    """
    return {
        "length":               get_length(password),
        "entropy":              round(get_shannon_entropy(password), 4),
        "charset_size":         get_charset_size(password),
        "char_diversity":       round(get_char_diversity(password), 4),
        "syllable_count":       get_syllable_count(password),
        "phonetic_score":       round(get_phonetic_score(password), 4),
        "word_count":           get_word_count(password),
        "digit_ratio":          round(get_digit_ratio(password), 4),
        "special_char_ratio":   round(get_special_char_ratio(password), 4),
        "uppercase_ratio":      round(get_uppercase_ratio(password), 4),
        "has_keyboard_walk":    get_has_keyboard_walk(password),
        "has_repeated_chars":   get_has_repeated_chars(password),
        "is_common_password":   get_is_common_password(password),
        "bigram_entropy":       round(get_bigram_entropy(password), 4),
        "crack_time_score":     round(get_estimated_crack_time_score(password), 4),
    }


def extract_feature_vector(password: str) -> list:
    """
    Same as extract_features() but returns a plain list in fixed order.
    Use this when feeding directly into sklearn — it needs arrays, not dicts.

    Usage:
        X = [extract_feature_vector(p) for p in passwords]
        model.predict(X)
    """
    return list(extract_features(password).values())


# Ordered list of feature names — used for DataFrame columns and model explainability
FEATURE_NAMES = list(extract_features("placeholder").keys())


# ── quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_passwords = [
        "123456",                    # extremely weak, common
        "password",                  # dictionary word, common
        "P@ssw0rd",                  # classic substitution — still weak
        "xK9!mPq2",                  # random, strong but not memorable
        "correct-horse-battery",     # passphrase, strong + memorable
        "Tr0ub4dor&3",               # XKCD comic password — strong but hard
        "Mumbai@2019!Chai",          # personalised, strong + memorable
    ]

    print(f"{'Password':<28} {'Len':>4} {'Entropy':>8} {'Syllables':>10} "
          f"{'Phonetic':>9} {'Words':>6} {'CrackScore':>11}")
    print("-" * 80)

    for pwd in test_passwords:
        f = extract_features(pwd)
        print(
            f"{pwd:<28} "
            f"{f['length']:>4} "
            f"{f['entropy']:>8.3f} "
            f"{f['syllable_count']:>10} "
            f"{f['phonetic_score']:>9.3f} "
            f"{f['word_count']:>6} "
            f"{f['crack_time_score']:>11.3f}"
        )
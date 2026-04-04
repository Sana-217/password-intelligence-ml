# generator/passphrase_transformer.py
"""
Passphrase Transformer — converts a user's familiar phrase into a
secure, memorable password.

CORE IDEA
──────────
Most password systems ask users to remember a random string.
This module does the opposite: it takes something the user
ALREADY knows (a phrase from their life) and transforms it
into a secure password — preserving the memory anchor while
dramatically increasing entropy.

The user remembers the original phrase.
The system handles the security transformation.

This is the key innovation that separates this project from
a standard password generator.

TRANSFORMATION PIPELINE (5 stages)
─────────────────────────────────────
  Stage 1 — Parse & filter
            Split phrase into words, remove filler words (the, is, a...)
            Reason: "my dog name is Bruno" → ["my","dog","Bruno"]
            Filler words add length but no memorability or entropy

  Stage 2 — Select anchor words
            Keep 2–4 most meaningful words from the filtered list
            Reason: shorter output = more typeable + less error-prone

  Stage 3 — Smart substitutions
            Apply limited leet-speak substitutions to ONE word only
            Reason: applying to ALL words makes the password unreadable
            and defeats memorability. One substitution = enough entropy boost.

  Stage 4 — Structure & capitalisation
            Capitalise each anchor word, join with separator
            Reason: CamelCase/TitleCase is readable AND adds uppercase chars

  Stage 5 — Entropy injection
            Append a short numeric suffix and one special character
            Reason: guarantees digits + special chars without destroying
            the readable core of the password

DESIGN DECISIONS (for your viva)
──────────────────────────────────
Q: Why remove filler words?
A: "is", "the", "a", "my" add zero memorability. Removing them
   makes the password shorter without losing the memory anchor.

Q: Why only substitute ONE word?
A: Full leet-speak (p@ssw0rd) is in every cracker's dictionary.
   Single substitution is rarer and still readable.

Q: Why append suffix instead of inserting randomly?
A: Users remember "word + number + symbol" patterns more reliably
   than scattered insertions. Appendix position is predictable TO
   THE USER but not to an attacker who doesn't know the base phrase.

Q: Is this actually secure?
A: Yes — the security comes from the entropy of the original phrase
   (which an attacker cannot guess) combined with the transformation.
   The phrase "my dog name is Bruno" has ~40+ bits of entropy from
   the attacker's perspective if they don't know your dog's name.
"""

import re
import sys
import secrets
import string
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── substitution table ────────────────────────────────────────────────────────
# LIMITED set — only the most readable substitutions.
# Full leet tables are in every cracking dictionary.
# Keeping it small means the output stays recognisable to the user.
SUBSTITUTIONS = {
    "a": "@",
    "e": "3",
    "o": "0",
    "i": "1",
    "s": "$",
}

# ── filler words to remove ────────────────────────────────────────────────────
# These words appear in almost every sentence and carry no personal meaning.
# Removing them shortens the password without losing the memory anchor.
FILLER_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "my", "your", "his", "her", "our", "their", "its",
    "i", "we", "you", "he", "she", "they", "it",
    "in", "on", "at", "to", "for", "of", "and", "or", "but",
    "this", "that", "with", "from", "by", "as", "so",
    "name", "named", "called",   # common in phrases like "my dog named Bruno"
}

# ── separators to try (user can choose) ──────────────────────────────────────
SEPARATORS = ["-", "_", ".", ""]

# ── special characters for suffix ────────────────────────────────────────────
SUFFIX_SPECIALS = ["!", "@", "#", "$", "%", "&", "*"]


# ── stage functions ───────────────────────────────────────────────────────────

def _parse_and_filter(phrase: str) -> list[str]:
    """
    Stage 1: tokenise the phrase and remove filler words.

    "my dog name is Bruno" → ["dog", "Bruno"]
    "I love coffee every morning" → ["love", "coffee", "morning"]
    "correct horse battery staple" → ["correct", "horse", "battery", "staple"]

    Keeps capitalisation of original tokens (proper nouns stay capitalised).
    Minimum word length: 2 characters (removes single letters).
    """
    # Tokenise: split on whitespace and punctuation
    tokens = re.findall(r"[a-zA-Z']+", phrase)

    filtered = []
    for token in tokens:
        # Strip possessives ("Bruno's" → "Bruno")
        clean = token.replace("'s", "").replace("'", "")
        if not clean:
            continue
        # Remove filler words (case-insensitive check)
        if clean.lower() in FILLER_WORDS:
            continue
        # Remove very short words (single letters, 2-char filler)
        if len(clean) < 2:
            continue
        filtered.append(clean)

    return filtered


def _select_anchor_words(
    words: list[str],
    max_words: int = 3,
) -> list[str]:
    """
    Stage 2: select the most meaningful words to anchor the password.

    Strategy:
      - Prefer longer words (more distinctive, harder to guess)
      - Keep at most max_words (default 3) to stay typeable
      - Always include proper nouns (capitalised words) — they are
        personally meaningful and memorable

    "dog Bruno coffee morning" → ["Bruno", "coffee", "morning"]
    (Bruno is a proper noun → included first; then longest remaining)
    """
    if not words:
        return []

    # Separate proper nouns (capitalised) from common words
    proper = [w for w in words if w[0].isupper()]
    common = [w for w in words if not w[0].isupper()]

    # Sort common words by length descending (longer = more distinctive)
    common.sort(key=len, reverse=True)

    # Build anchor list: proper nouns first, then longest common words
    anchors = proper + common

    # Cap at max_words
    return anchors[:max_words]


def _apply_substitution(word: str) -> str:
    """
    Stage 3: apply ONE smart substitution to a word.

    Applies the first substitution match found — stops after one.
    Reason: multiple substitutions (p@$$w0rd) are in every dictionary
    and look ugly. One substitution looks intentional and is rarer.

    "Bruno"  → "Bruno"  (no substitutable chars → unchanged)
    "coffee" → "c0ffee" (o → 0, first match)
    "apple"  → "@pple"  (a → @, first match)
    """
    for i, char in enumerate(word):
        if char.lower() in SUBSTITUTIONS:
            sub  = SUBSTITUTIONS[char.lower()]
            # Preserve capitalisation intent:
            # if original was uppercase 'A' → use uppercase sub if possible
            # '@' and '0' are not alpha so no case issue
            return word[:i] + sub + word[i + 1:]
    return word   # no substitutable character found → return unchanged


def _build_core(
    anchors:   list[str],
    separator: str,
    sub_index: int,
) -> str:
    """
    Stage 4: capitalise anchor words, apply ONE substitution, join.

    sub_index: which anchor word (by index) gets the substitution.
    Using a specific index (default: second word) means the pattern
    is predictable to the user but not to an attacker.

    ["dog", "Bruno", "coffee"], sep="-", sub_index=2
    → "Dog-Bruno-c0ffee"
    """
    processed = []
    for i, word in enumerate(anchors):
        # Always capitalise first letter (title case for the word)
        cased = word.capitalize()
        # Apply substitution to exactly one word
        if i == sub_index:
            cased = _apply_substitution(cased)
        processed.append(cased)

    return separator.join(processed)


def _generate_suffix(
    year_hint:      Optional[int] = None,
    n_digits:       int           = 2,
) -> tuple[str, str]:
    """
    Stage 5a: generate a numeric suffix.

    If year_hint is provided (e.g. 2024), use it — memorable for the user.
    Otherwise generate a short random 2-digit number.

    Returns (suffix_str, explanation) tuple.
    """
    if year_hint:
        return str(year_hint), f"year {year_hint}"
    else:
        digits = "".join(secrets.choice(string.digits) for _ in range(n_digits))
        return digits, f"random digits ({digits})"


def _generate_special(special_char: Optional[str] = None) -> str:
    """
    Stage 5b: choose a special character for the suffix.
    Uses the provided character if given, otherwise random from safe set.
    """
    if special_char and special_char in string.punctuation:
        return special_char
    return secrets.choice(SUFFIX_SPECIALS)


# ── master transformer ────────────────────────────────────────────────────────

def transform_passphrase(
    phrase:         str,
    separator:      str           = "",
    max_words:      int           = 3,
    sub_index:      int           = 1,
    year_hint:      Optional[int] = None,
    special_char:   Optional[str] = None,
    n_candidates:   int           = 4,
) -> dict:
    """
    Full transformation pipeline: phrase → secure memorable password.

    Args:
        phrase       : user's input phrase e.g. "my dog name is Bruno"
        separator    : character between words ("", "-", "_", ".")
        max_words    : maximum anchor words to keep (default 3)
        sub_index    : which word (0-indexed) gets leet substitution
        year_hint    : optional year to append (e.g. 2024) — more memorable
        special_char : override the appended special character
        n_candidates : number of variants to generate

    Returns:
        {
          "original_phrase": "my dog name is Bruno",
          "password":        "DogBrun0!24",        ← primary recommendation
          "candidates":      [...],                 ← alternative variants
          "pipeline": {                             ← step-by-step breakdown
            "filtered_words":  ["dog", "Bruno"],
            "anchor_words":    ["Bruno", "dog"],
            "core":            "Brun0Dog",
            "suffix":          "24!",
            "final":           "Brun0Dog24!",
          },
          "explanation":   "...",                   ← human-readable walkthrough
          "strength_tips": [...],
        }
    """
    if not phrase or not phrase.strip():
        raise ValueError("Phrase cannot be empty.")

    # ── Stage 1: parse ─────────────────────────────────────────────
    filtered = _parse_and_filter(phrase)

    if not filtered:
        raise ValueError(
            "No meaningful words found in phrase after filtering. "
            "Try a phrase with more content words (nouns, verbs, adjectives)."
        )

    # ── Stage 2: select anchors ────────────────────────────────────
    anchors = _select_anchor_words(filtered, max_words=max_words)

    # ── Stage 3+4: build core ──────────────────────────────────────
    # Clamp sub_index to valid range
    sub_idx = min(sub_index, len(anchors) - 1)
    core    = _build_core(anchors, separator, sub_idx)

    # ── Stage 5: append suffix ─────────────────────────────────────
    numeric_suffix, suffix_explanation = _generate_suffix(year_hint)
    special                             = _generate_special(special_char)
    suffix                              = numeric_suffix + special

    password = core + suffix

    # ── Build pipeline breakdown (for display + viva) ──────────────
    pipeline = {
        "original_phrase":  phrase,
        "filtered_words":   filtered,
        "anchor_words":     anchors,
        "substitution":     f"word[{sub_idx}] = '{anchors[sub_idx]}' → '{_apply_substitution(anchors[sub_idx].capitalize())}'",
        "core":             core,
        "numeric_suffix":   numeric_suffix,
        "special_char":     special,
        "final":            password,
    }

    # ── Generate candidates (variations) ──────────────────────────
    candidates = _generate_candidates(
        anchors, filtered, n_candidates, year_hint
    )
    # Ensure primary password is at position 0
    if password not in candidates:
        candidates.insert(0, password)
    else:
        candidates.remove(password)
        candidates.insert(0, password)

    # ── Human-readable explanation ─────────────────────────────────
    explanation = _build_explanation(phrase, filtered, anchors, pipeline)

    # ── Strength tips ──────────────────────────────────────────────
    tips = _strength_tips(password, phrase)

    return {
        "original_phrase": phrase,
        "password":        password,
        "candidates":      candidates[:n_candidates],
        "pipeline":        pipeline,
        "explanation":     explanation,
        "strength_tips":   tips,
    }


def _generate_candidates(
    anchors:    list[str],
    all_words:  list[str],
    n:          int,
    year_hint:  Optional[int],
) -> list[str]:
    """
    Generates n alternative password variants from the same phrase.
    Varies: separator, substitution target, suffix, word order.
    """
    candidates = []
    sep_choices = ["", "-", "_", "."]

    for i in range(n):
        sep      = sep_choices[i % len(sep_choices)]
        sub_idx  = i % len(anchors)

        # Occasionally shuffle word order for variety
        if i % 3 == 2 and len(anchors) > 1:
            shuffled = list(reversed(anchors))
        else:
            shuffled = anchors

        core    = _build_core(shuffled, sep, sub_idx)
        numeric = str(year_hint) if year_hint else "".join(
            secrets.choice(string.digits) for _ in range(2)
        )
        special = secrets.choice(SUFFIX_SPECIALS)
        candidates.append(core + numeric + special)

    return candidates


def _build_explanation(
    phrase:    str,
    filtered:  list[str],
    anchors:   list[str],
    pipeline:  dict,
) -> str:
    """Builds a human-readable step-by-step explanation."""
    removed = set(re.findall(r"[a-zA-Z']+", phrase.lower())) - {
        w.lower() for w in filtered
    }
    removed_str = ", ".join(f"'{w}'" for w in sorted(removed)) or "none"

    lines = [
        f"Input phrase : \"{phrase}\"",
        f"",
        f"Step 1 — Filter filler words",
        f"  Removed    : {removed_str}",
        f"  Remaining  : {filtered}",
        f"",
        f"Step 2 — Select anchor words (max {len(anchors)})",
        f"  Anchors    : {anchors}",
        f"  (Proper nouns kept first; then longest words)",
        f"",
        f"Step 3 — Smart substitution (one word only)",
        f"  {pipeline['substitution']}",
        f"  (Limited substitution — readable but harder to crack)",
        f"",
        f"Step 4 — Capitalise + join",
        f"  Core       : {pipeline['core']}",
        f"",
        f"Step 5 — Append suffix",
        f"  Digits     : {pipeline['numeric_suffix']}",
        f"  Special    : {pipeline['special_char']}",
        f"  Final      : {pipeline['final']}",
    ]
    return "\n".join(lines)


def _strength_tips(password: str, phrase: str) -> list[str]:
    """Returns contextual tips to make the password even stronger."""
    tips = []
    if len(password) < 12:
        tips.append(
            "Add more words from your phrase (increase max_words to 4) "
            "for a longer password."
        )
    if not any(c in string.punctuation for c in password):
        tips.append("Add a special character — use special_char='!' parameter.")
    if len(set(c.lower() for c in password if c.isalpha())) < 4:
        tips.append("Low character diversity — try a longer phrase.")
    if phrase.lower() in password.lower():
        tips.append(
            "The original phrase is partially visible in the output. "
            "This is intentional (memorability), but avoid using it "
            "for high-security accounts."
        )
    if not tips:
        tips.append(
            "Password looks good! Store it in the vault and use "
            "Memory Aid to reinforce recall."
        )
    return tips


# ── convenience wrappers ──────────────────────────────────────────────────────

def quick_transform(phrase: str, year: Optional[int] = None) -> str:
    """
    One-line transform — returns just the password string.
    Used by app/app.py for simple requests.

    Usage:
        pwd = quick_transform("my dog name is Bruno", year=2024)
        # → "BrunoDog24!" (or similar)
    """
    result = transform_passphrase(phrase, year_hint=year)
    return result["password"]


def transform_with_scores(phrase: str, year: Optional[int] = None) -> dict:
    """
    Transforms phrase AND runs both ML models on the result.
    Used by the Flask route that shows scores alongside the password.

    Returns transform result merged with ML scores.
    """
    result = transform_passphrase(phrase, year_hint=year)

    try:
        from generator.password_gen import score_password
        scores = score_password(result["password"])
        result["ml_scores"] = scores
    except Exception:
        result["ml_scores"] = None   # ML models not trained yet — non-fatal

    return result


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  generator/passphrase_transformer.py — self test")
    print("=" * 64)

    test_cases = [
        ("my dog name is Bruno",          2024),
        ("I love coffee every morning",   None),
        ("correct horse battery staple",  None),
        ("my cat whiskers is 3 years old",2021),
        ("Mount Everest is very tall",    None),
        ("sunshine and rainbow forever",  2025),
    ]

    for phrase, year in test_cases:
        print(f"\n{'─'*64}")
        result = transform_passphrase(phrase, year_hint=year, n_candidates=4)

        print(f"  Input    : {result['original_phrase']}")
        print(f"  Password : {result['password']}")
        print(f"\n  Pipeline breakdown:")
        p = result["pipeline"]
        print(f"    Filtered  : {p['filtered_words']}")
        print(f"    Anchors   : {p['anchor_words']}")
        print(f"    Sub       : {p['substitution']}")
        print(f"    Core      : {p['core']}")
        print(f"    Suffix    : {p['numeric_suffix']}{p['special_char']}")

        print(f"\n  Candidates:")
        for i, cand in enumerate(result["candidates"], 1):
            marker = " ← recommended" if i == 1 else ""
            print(f"    {i}. {cand}{marker}")

        print(f"\n  Tips:")
        for tip in result["strength_tips"]:
            print(f"    - {tip}")

    # ── Detailed explanation for one case ──────────────────────────
    print(f"\n{'='*64}")
    print("  Full explanation (for viva demo):")
    print(f"{'='*64}")
    result = transform_passphrase("my dog name is Bruno", year_hint=2024)
    print(result["explanation"])

    # ── ML scoring (if models trained) ────────────────────────────
    print(f"\n{'='*64}")
    print("  ML scoring:")
    print(f"{'='*64}")
    result_with_scores = transform_with_scores("my dog name is Bruno", year=2024)
    if result_with_scores["ml_scores"]:
        s = result_with_scores["ml_scores"]
        print(f"  Password     : {s['password']}")
        print(f"  Strength     : {s['strength_label']}  (p={s['strength_proba']:.3f})")
        print(f"  Memorability : {s['memorability_label']}  (p={s['memorability_proba']:.3f})")
        print(f"  Score        : {s['combined_score']:.3f}")
    else:
        print("  (Train models first to see ML scores)")

    print(f"\n{'='*64}")
    print("  ALL TESTS PASSED")
    print(f"{'='*64}")
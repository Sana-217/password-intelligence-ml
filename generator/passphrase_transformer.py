"""
passphrase_transformer.py
─────────────────────────────────────────────────────────────────────────────
PassGuard — Enhanced Passphrase Transformer
Converts a user's personal phrase into a secure, memorable password through
a 5-stage pipeline with a pool of 5 enhancement techniques.

Enhancement techniques (randomly 1 or 2 are selected per run):
  T1 — Acronym Injection    : initials of anchors → real word → inject
  T2 — Phonetic Mirroring   : anchor → pronounceable sound-alike
  T3 — Visual Shape Sub     : shape-based char replacements (B→8, G→9 …)
  T4 — Reverse Anchor       : reverse the non-primary anchor word
  T5 — Leet Substitution    : classic a→@, e→3, o→0, i→1, s→$ (original)

Author : PassGuard Team (Sana, Mahankali Bhavana, K Sulochana Preethi)
Guide  : Ms. B. Saritha, Associate Professor, Dept. of CSE, MVSREC
"""

import re
import secrets
import random
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Words removed in Stage 1 — semantically empty, add no personal meaning
FILLER_WORDS = {
    # articles
    "a", "an", "the",
    # personal pronouns
    "i", "my", "your", "his", "her", "its", "our", "their",
    "we", "you", "he", "she", "they", "it", "me", "him", "us", "them",
    # prepositions
    "in", "on", "at", "to", "for", "of", "from", "by", "as", "with",
    "into", "onto", "upon", "about", "above", "below", "between",
    # conjunctions
    "and", "or", "but", "so", "yet", "nor", "although", "because",
    # common verbs
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "do", "does", "did",
    # noise words
    "name", "named", "called", "known", "this", "that", "these", "those",
    "very", "really", "just", "also", "too", "only", "even",
}

# T5 — Leet substitution table
LEET_TABLE = str.maketrans({
    'a': '@', 'A': '@',
    'e': '3', 'E': '3',
    'o': '0', 'O': '0',
    'i': '1', 'I': '1',
    's': '$', 'S': '$',
})

# T3 — Visual shape substitution (different chars from leet)
VISUAL_TABLE = str.maketrans({
    'b': '8', 'B': '8',
    'g': '9', 'G': '9',
    'z': '2', 'Z': '2',
    't': '7', 'T': '7',
    'l': '1', 'L': '1',
    'q': '9', 'Q': '9',
    'c': '(',  'C': '(',
})

# T2 — Phonetic mirror dictionary
# word → sound-alike with better security properties
PHONETIC_MIRRORS = {
    "dog":    "dawg",
    "cat":    "kat",
    "love":   "luv",
    "fire":   "fyre",
    "night":  "nite",
    "light":  "lite",
    "right":  "rite",
    "write":  "rite",
    "phone":  "fone",
    "photo":  "foto",
    "cool":   "kool",
    "crazy":  "krazy",
    "queen":  "kween",
    "quick":  "kwik",
    "sure":   "shur",
    "star":   "staar",
    "blue":   "bluu",
    "true":   "tru",
    "you":    "yu",
    "home":   "hohm",
    "know":   "noe",
    "new":    "nu",
    "great":  "gr8",
    "mate":   "m8",
    "late":   "l8",
    "wait":   "w8",
    "hate":   "h8",
    "cake":   "kayk",
    "lake":   "layk",
    "make":   "mayk",
    "take":   "tayk",
    "black":  "blak",
    "back":   "bak",
    "track":  "trak",
    "friend": "frend",
    "money":  "munny",
    "funny":  "funni",
    "sunny":  "sunni",
    "honey":  "hunni",
    "coffee": "kofi",
    "city":   "siti",
    "happy":  "hapi",
    "baby":   "babi",
    "lady":   "laydee",
    "ready":  "reddy",
    "lucky":  "lukki",
    "rocky":  "rokki",
    "tiger":  "tyger",
    "flower": "flowr",
    "power":  "powr",
    "tower":  "towr",
    "shower": "showr",
    "sweet":  "swit",
    "street": "strit",
    "dream":  "dreem",
    "team":   "teem",
    "real":   "reel",
    "deal":   "deel",
    "feel":   "fiel",
    "storm":  "storrm",
    "born":   "borrn",
    "gold":   "golld",
    "bold":   "bolld",
    "cold":   "kolld",
    "old":    "olld",
    "world":  "wurld",
    "girl":   "gurl",
    "bird":   "burd",
    "first":  "furst",
    "work":   "wurk",
    "word":   "wurd",
}

# T1 — Acronym word bank: maps letters to memorable words
# Each letter has multiple options so we can form real compound words
ACRONYM_WORDS = {
    'a': ['Apple', 'Arrow', 'Atom', 'Ace', 'Arc'],
    'b': ['Bike', 'Bolt', 'Blaze', 'Base', 'Byte'],
    'c': ['Cloud', 'Core', 'Cruz', 'Cube', 'Chip'],
    'd': ['Day', 'Dart', 'Dawn', 'Deep', 'Dusk'],
    'e': ['Eagle', 'Edge', 'Echo', 'Ember', 'Eon'],
    'f': ['Fire', 'Flame', 'Flux', 'Forge', 'Frost'],
    'g': ['Gold', 'Gate', 'Gear', 'Glow', 'Grid'],
    'h': ['Hawk', 'Haze', 'Heat', 'Hero', 'Hub'],
    'i': ['Ice', 'Iron', 'Iris', 'Ink', 'Ion'],
    'j': ['Jade', 'Jet', 'Jump', 'Just', 'Joy'],
    'k': ['Key', 'Kite', 'Knox', 'Keen', 'King'],
    'l': ['Lake', 'Leaf', 'Lens', 'Lion', 'Lux'],
    'm': ['Moon', 'Mist', 'Mark', 'Maze', 'Mint'],
    'n': ['Nova', 'Node', 'Neon', 'Noon', 'Nord'],
    'o': ['Orbit', 'Oak', 'Opal', 'Onyx', 'Ore'],
    'p': ['Peak', 'Pixel', 'Pulse', 'Pine', 'Pond'],
    'q': ['Quest', 'Quad', 'Quill', 'Quick', 'Quiz'],
    'r': ['Rock', 'Rain', 'Ray', 'Reef', 'Root'],
    's': ['Star', 'Storm', 'Sky', 'Stone', 'Soul'],
    't': ['Tree', 'Tide', 'Torch', 'Trail', 'Tune'],
    'u': ['Ultra', 'Unit', 'Ural', 'Urge', 'Unix'],
    'v': ['Vault', 'Volt', 'Veil', 'Vibe', 'Vex'],
    'w': ['Wave', 'Wind', 'Wild', 'Wire', 'Wolf'],
    'x': ['Xray', 'Xeno', 'Xcel', 'Xmas', 'Xact'],
    'y': ['Year', 'Yell', 'Yogi', 'Yard', 'York'],
    'z': ['Zero', 'Zone', 'Zinc', 'Zeal', 'Zoom'],
}

# Special characters pool for Stage 5 entropy injection
SPECIAL_CHARS = list('!@#$%&*')


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — Filter filler words
# ─────────────────────────────────────────────────────────────────────────────

def _filter_fillers(phrase: str) -> list[str]:
    """
    Tokenise the input phrase and remove semantically empty filler words.
    Preserves the original capitalisation of each token for proper noun
    detection in Stage 2.

    Returns a list of retained tokens in original case.
    """
    # Split on whitespace and punctuation except apostrophes
    tokens = re.split(r"[\s\-_,.:;!?\"()\[\]{}]+", phrase.strip())
    tokens = [t for t in tokens if t]  # remove empty strings

    retained = []
    for token in tokens:
        if token.lower() not in FILLER_WORDS:
            retained.append(token)

    # If ALL words were filtered (very short phrases), return original tokens
    if not retained:
        retained = tokens

    return retained


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — Select anchor words
# ─────────────────────────────────────────────────────────────────────────────

def _select_anchors(tokens: list[str], max_anchors: int = 3) -> list[str]:
    """
    Select up to max_anchors anchor words from the retained tokens.

    Selection priority:
      1. Proper nouns (first character is uppercase in original phrase)
      2. Longer words (more distinctive, more entropy)
      3. Earlier position (user likely put important words first)

    Returns a list of anchor words in [primary, secondary, ...] order.
    The primary anchor is always index 0.
    """
    if not tokens:
        return []

    # Separate proper nouns from common words
    proper_nouns = [t for t in tokens if t[0].isupper()]
    common_words = [t for t in tokens if not t[0].isupper()]

    # Sort common words by length descending
    common_words.sort(key=len, reverse=True)

    # Build anchor list: proper nouns first, then common words
    anchors = proper_nouns + common_words

    # Cap at max_anchors
    return anchors[:max_anchors]


# ─────────────────────────────────────────────────────────────────────────────
#  TECHNIQUE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_leet(word: str) -> str:
    """T5: Classic leet substitution — a→@, e→3, o→0, i→1, s→$"""
    return word.translate(LEET_TABLE)


def _apply_visual_shape(word: str) -> str:
    """T3: Visual shape substitution — b→8, g→9, z→2, t→7, l→1"""
    return word.translate(VISUAL_TABLE)


def _apply_phonetic_mirror(word: str) -> str:
    """
    T2: Replace word with its phonetic sound-alike if one exists.
    Returns the mirrored word if found, otherwise returns original word.
    The sound-alike preserves the phonological loop rehearsability
    (Baddeley 1986) while reducing dictionary predictability.
    """
    lower = word.lower()
    if lower in PHONETIC_MIRRORS:
        mirrored = PHONETIC_MIRRORS[lower]
        # Preserve capitalisation style of original
        if word[0].isupper():
            return mirrored.capitalize()
        return mirrored
    # No mirror found — return original unchanged
    return word


def _apply_reverse(word: str) -> str:
    """
    T4: Reverse the characters of the word.
    Skips palindromes (reversing adds nothing).
    'Dog' → 'goD'
    """
    if word.lower() == word.lower()[::-1]:  # palindrome check
        return word
    reversed_word = word[::-1]
    # Capitalise first letter if original was capitalised
    if word[0].isupper():
        return reversed_word[0].upper() + reversed_word[1:].lower()
    return reversed_word.lower()


def _apply_acronym_inject(anchors: list[str]) -> str:
    """
    T1: Take the first letter of each anchor word, then for each letter
    pick a word from ACRONYM_WORDS, and concatenate them into a new token.

    'Bruno' + 'Dog' → initials 'B', 'D'
    → B: 'Bolt', D: 'Day' → inject 'BoltDay' into the password

    The generated word is pronounceable (activates phonological loop)
    and derived from the user's anchor initials (memorable connection).
    Returns the generated acronym token as a capitalised compound word.
    """
    initials = [anchor[0].lower() for anchor in anchors if anchor]
    # Cap at 3 initials to keep the token manageable
    initials = initials[:3]

    parts = []
    for letter in initials:
        word_options = ACRONYM_WORDS.get(letter, ['Key'])
        # Use secrets.choice for unpredictability
        parts.append(secrets.choice(word_options))

    return ''.join(parts)  # e.g. "BoltDay" or "BoltDayMoon"


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — Apply randomly selected techniques
# ─────────────────────────────────────────────────────────────────────────────

# All available techniques
TECHNIQUE_POOL = ['T1_acronym', 'T2_phonetic', 'T3_visual', 'T4_reverse', 'T5_leet']

# Conflict rules: these pairs should NOT both target the same word
CONFLICTS = {
    ('T3_visual', 'T5_leet'),   # both do character substitution
    ('T4_reverse', 'T2_phonetic'),  # both change the word structure significantly
}


def _pick_techniques() -> list[str]:
    """
    Randomly select 1 or 2 techniques from the pool.
    2 techniques selected with 70% probability, 1 with 30%.
    Ensures conflicting technique pairs are not both selected.
    """
    n = random.choices([1, 2], weights=[30, 70])[0]

    selected = random.sample(TECHNIQUE_POOL, n)

    # Check for conflicts — if conflicting pair selected, drop one
    if len(selected) == 2:
        pair = tuple(sorted(selected))
        if pair in CONFLICTS or (pair[1], pair[0]) in CONFLICTS:
            # Keep the first one, drop the second
            selected = [selected[0]]

    return selected


def _apply_techniques(anchors: list[str], techniques: list[str]) -> tuple[list[str], str, list[str]]:
    """
    Apply the selected techniques to the anchor words.

    Rules:
    - T1 (acronym inject) generates a NEW token — injected alongside anchors
    - T2, T3, T4, T5 modify existing anchor words
    - When 2 techniques both modify words, apply each to a DIFFERENT anchor
    - Primary anchor (index 0) is always modified first if applicable

    Returns:
        modified_anchors  : list of (possibly modified) anchor words
        acronym_token     : the T1 generated word (empty string if T1 not used)
        applied_log       : human-readable description of what was applied
    """
    if not anchors:
        return anchors, '', []

    modified = list(anchors)  # copy so we don't mutate original
    acronym_token = ''
    applied_log = []

    # Track which anchor index each technique targets
    word_techniques = [t for t in techniques if t != 'T1_acronym']
    has_acronym = 'T1_acronym' in techniques

    # Assign word techniques to anchor indices
    # First technique → anchor index 0 (primary)
    # Second technique → anchor index 1 (secondary, if exists)
    for i, tech in enumerate(word_techniques):
        target_idx = min(i, len(modified) - 1)
        original = modified[target_idx]

        if tech == 'T5_leet':
            modified[target_idx] = _apply_leet(original)
            applied_log.append(
                f"T5 Leet substitution on '{original}' → '{modified[target_idx]}'"
            )

        elif tech == 'T3_visual':
            result = _apply_visual_shape(original)
            if result != original:  # only log if something actually changed
                modified[target_idx] = result
                applied_log.append(
                    f"T3 Visual shape on '{original}' → '{modified[target_idx]}'"
                )

        elif tech == 'T2_phonetic':
            result = _apply_phonetic_mirror(original)
            if result != original:
                modified[target_idx] = result
                applied_log.append(
                    f"T2 Phonetic mirror on '{original}' → '{modified[target_idx]}'"
                )

        elif tech == 'T4_reverse':
            # Prefer to reverse the non-primary (last) anchor
            rev_idx = len(modified) - 1 if len(modified) > 1 else 0
            original_rev = modified[rev_idx]
            result = _apply_reverse(original_rev)
            if result != original_rev:
                modified[rev_idx] = result
                applied_log.append(
                    f"T4 Reverse on '{original_rev}' → '{modified[rev_idx]}'"
                )

    # Handle T1 acronym injection
    if has_acronym:
        acronym_token = _apply_acronym_inject(anchors)  # use ORIGINAL anchors for initials
        applied_log.append(
            f"T1 Acronym inject: initials of {[a[0] for a in anchors[:3]]} → '{acronym_token}'"
        )

    return modified, acronym_token, applied_log


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 — Capitalise and join
# ─────────────────────────────────────────────────────────────────────────────

def _capitalise_and_join(
    modified_anchors: list[str],
    acronym_token: str,
    use_interleave: bool = False
) -> str:
    """
    Stage 4: Capitalise each anchor word and join them.

    If use_interleave is True AND exactly 2 anchors of similar length exist,
    interleave their characters: 'Bruno' + 'Dog' → 'BDrunog'

    The acronym token (if generated by T1) is inserted between the
    first and second anchor as a separator/bridge word.

    Returns the core password string (before entropy injection).
    """
    # Capitalise all anchors
    caps = [w.capitalize() for w in modified_anchors]

    if use_interleave and len(caps) == 2:
        a, b = caps[0], caps[1]
        # Only interleave if lengths are within 3 chars of each other
        if abs(len(a) - len(b)) <= 3:
            interleaved = ''.join(
                c for pair in zip(a, b) for c in pair
            )
            # Append remaining chars from the longer word
            min_len = min(len(a), len(b))
            interleaved += a[min_len:] + b[min_len:]
            core = interleaved
            if acronym_token:
                core = acronym_token + core
            return core

    # Standard join
    if acronym_token and len(caps) >= 2:
        # Insert acronym token between first and second anchor
        core = caps[0] + acronym_token + ''.join(caps[1:])
    else:
        core = ''.join(caps)

    return core


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 5 — Entropy injection
# ─────────────────────────────────────────────────────────────────────────────

def _inject_entropy(core: str, year_hint: Optional[str] = None) -> str:
    """
    Stage 5: Append a numeric suffix and a special character.

    If year_hint is provided (e.g. "2024"), use it as the numeric suffix
    — personally meaningful and easy to remember.
    If not, generate 2 random digits using the CSPRNG.

    A special character is selected from SPECIAL_CHARS using the CSPRNG.
    """
    # Accept both int (2024) and str ("2024") — app.py passes int
    year_str = str(year_hint).strip() if year_hint is not None else ""
    if year_str.isdigit():
        suffix = year_str
    else:
        # 2 random digits — secrets.randbelow(90)+10 guarantees 2 digits
        suffix = str(secrets.randbelow(90) + 10)

    special = secrets.choice(SPECIAL_CHARS)
    return core + suffix + special


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRANSFORM FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def transform_passphrase(
    phrase: str,
    year_hint: Optional[str] = None,
    force_techniques: Optional[list[str]] = None,
    use_interleave: bool = False,
    n_variants: int = 5,
) -> dict:
    """
    Transform a user's personal phrase into a secure memorable password.

    Pipeline:
        Stage 1  — Filter filler words (always)
        Stage 2  — Select anchor words (always)
        Stage 3  — Apply 1-2 randomly selected techniques from pool of 5
        Stage 4  — Capitalise + join (with optional interleave)
        Stage 5  — Entropy injection (year + special char)

    Parameters:
        phrase           : natural language phrase from user
        year_hint        : optional year string (e.g. "2024") for suffix
        force_techniques : override random selection (for testing/reproducibility)
        use_interleave   : enable character interleaving in Stage 4
        n_variants       : number of alternative variants to generate

    Returns a dict with:
        password         : the primary recommended password
        techniques_used  : list of technique names applied
        pipeline_log     : step-by-step breakdown for UI display
        variants         : list of alternative passwords with their techniques
        anchors_original : original anchor words (before modification)
        anchors_modified : anchor words after technique application
    """
    if not phrase or not phrase.strip():
        raise ValueError("Phrase cannot be empty")

    # ── Stage 1: Filter ──────────────────────────────────────────────
    original_tokens = re.split(r"[\s\-_,.:;!?\"()\[\]{}]+", phrase.strip())
    original_tokens = [t for t in original_tokens if t]

    filtered_tokens = _filter_fillers(phrase)

    removed = [t for t in original_tokens if t.lower() in FILLER_WORDS]

    # ── Stage 2: Select anchors ───────────────────────────────────────
    anchors = _select_anchors(filtered_tokens)

    if not anchors:
        # Fallback: use the longest word from original
        anchors = sorted(original_tokens, key=len, reverse=True)[:2]

    # ── Stage 3: Apply techniques ─────────────────────────────────────
    if force_techniques:
        techniques = force_techniques
    else:
        techniques = _pick_techniques()

    modified_anchors, acronym_token, tech_log = _apply_techniques(
        anchors, techniques
    )

    # ── Stage 4: Capitalise + join ────────────────────────────────────
    # Decide interleave: enabled if flag set AND 2 anchors of similar length
    do_interleave = (
        use_interleave
        and len(modified_anchors) == 2
        and abs(len(modified_anchors[0]) - len(modified_anchors[1])) <= 3
    )

    core = _capitalise_and_join(modified_anchors, acronym_token, do_interleave)

    # ── Stage 5: Entropy injection ────────────────────────────────────
    password = _inject_entropy(core, year_hint)

    # ── Build pipeline log for UI ─────────────────────────────────────
    pipeline_log = [
        {
            "stage": "Stage 1 — Filter Filler Words",
            "input": phrase,
            "removed": removed if removed else ["(none removed)"],
            "retained": filtered_tokens,
            "output": " ".join(filtered_tokens),
        },
        {
            "stage": "Stage 2 — Select Anchor Words",
            "input": filtered_tokens,
            "output": anchors,
            "note": f"Selected {len(anchors)} anchor(s). Proper nouns prioritised."
        },
        {
            "stage": "Stage 3 — Enhancement Techniques Applied",
            "techniques_selected": techniques,
            "details": tech_log if tech_log else ["No word-level changes (T1 only)"],
            "anchors_before": anchors,
            "anchors_after": modified_anchors,
            "acronym_token": acronym_token if acronym_token else "(none)",
        },
        {
            "stage": "Stage 4 — Capitalise and Join",
            "input": modified_anchors,
            "interleave_used": do_interleave,
            "output": core,
        },
        {
            "stage": "Stage 5 — Entropy Injection",
            "core": core,
            "suffix": str(year_hint) if year_hint is not None else "random 2-digit",
            "special_char": password[-1],
            "output": password,
        },
    ]

    # ── Generate n_variants alternative passwords ─────────────────────
    variants = _generate_variants(
        anchors=anchors,
        year_hint=year_hint,
        primary_techniques=techniques,
        n=n_variants - 1,  # -1 because primary password is already one variant
    )

    # ── Return full result ────────────────────────────────────────────
    return {
        "password":          password,
        "techniques_used":   techniques,
        "pipeline_log":      pipeline_log,
        "variants":          variants,
        "anchors_original":  anchors,
        "anchors_modified":  modified_anchors,
        "acronym_token":     acronym_token,
        "interleave_used":   do_interleave,
        "core":              core,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  VARIANT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _generate_variants(
    anchors: list[str],
    year_hint: Optional[str],
    primary_techniques: list[str],
    n: int = 4,
) -> list[dict]:
    """
    Generate n alternative password variants using different technique
    combinations, ensuring no variant uses the exact same technique set
    as the primary password.

    Each variant is returned as:
        { 'password': str, 'techniques': list[str] }
    """
    variants = []
    seen_technique_sets = {tuple(sorted(primary_techniques))}
    attempts = 0

    while len(variants) < n and attempts < 30:
        attempts += 1
        techniques = _pick_techniques()
        tech_key = tuple(sorted(techniques))

        if tech_key in seen_technique_sets:
            continue  # skip duplicate technique combination

        seen_technique_sets.add(tech_key)

        try:
            modified, acronym_token, _ = _apply_techniques(anchors, techniques)
            core = _capitalise_and_join(modified, acronym_token)
            password = _inject_entropy(core, year_hint)
            variants.append({
                "password":   password,
                "techniques": techniques,
                "core":       core,
            })
        except Exception:
            continue  # skip any variant that errors out

    return variants


# ─────────────────────────────────────────────────────────────────────────────
#  TECHNIQUE DESCRIPTIONS (for UI display)
# ─────────────────────────────────────────────────────────────────────────────

TECHNIQUE_DESCRIPTIONS = {
    'T1_acronym':  "Acronym Injection — first letters of anchor words → pronounceable compound word injected as bridge token",
    'T2_phonetic': "Phonetic Mirroring — anchor word replaced with sound-alike (memorability preserved, dictionary predictability reduced)",
    'T3_visual':   "Visual Shape Substitution — shape-based replacements (B→8, G→9, Z→2, T→7) on one anchor word",
    'T4_reverse':  "Reverse Anchor — non-primary anchor word reversed (adds pattern unpredictability while remaining reconstructible)",
    'T5_leet':     "Leet Substitution — classic character substitutions (a→@, e→3, o→0, i→1, s→$) on one anchor word",
}


def get_technique_description(technique_id: str) -> str:
    """Return a human-readable description of a technique for UI display."""
    return TECHNIQUE_DESCRIPTIONS.get(technique_id, "Unknown technique")


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("PassGuard Enhanced Passphrase Transformer — Quick Test")
    print("=" * 65)

    test_phrases = [
        ("my dog name is Bruno",    "2024"),
        ("i love coffee every morning", "2023"),
        ("my favourite city is Mumbai", "2019"),
        ("sister birthday is March",    None),
        ("Rocky is my best friend",     "2022"),
    ]

    for phrase, year in test_phrases:
        print(f"\nPhrase : \"{phrase}\"")
        print(f"Year   : {year or 'none'}")

        result = transform_passphrase(phrase, year_hint=year)

        print(f"Anchors        : {result['anchors_original']}")
        print(f"Techniques     : {result['techniques_used']}")
        for line in result['pipeline_log']:
            stage = line['stage']
            out   = line.get('output', '')
            print(f"  {stage}")
            if out:
                print(f"    → {out}")
            if line.get('details'):
                for d in line['details']:
                    print(f"    • {d}")

        print(f"\n★ Password : {result['password']}")
        print(f"\n  Variants:")
        for v in result['variants']:
            print(f"    {v['password']:30s}  [{', '.join(v['techniques'])}]")
        print("-" * 65)


# ─────────────────────────────────────────────────────────────────────────────
#  COMPATIBILITY WRAPPER — used by app/app.py
# ─────────────────────────────────────────────────────────────────────────────

def transform_with_scores(
    phrase: str,
    year_hint: Optional[str] = None,
    year: Optional[str] = None,        # alias — app.py calls with 'year='
    scorer=None,
) -> dict:
    """
    Wrapper around transform_passphrase() that also scores the password
    using the ML scorer if one is provided.

    This function is imported by app/app.py via:
        from generator.passphrase_transformer import transform_with_scores

    Parameters:
        phrase     : natural language phrase from user
        year_hint  : optional year string (e.g. "2024")
        scorer     : optional ML scorer object with a .score(password) method
                     If None, scores default to 0.0

    Returns a dict compatible with the Flask /transform route:
        password         : the primary recommended password
        pipeline_log     : step-by-step breakdown for UI display
        techniques_used  : list of technique IDs applied
        variants         : list of alternative passwords
        anchors_original : original anchor words
        anchors_modified : anchor words after technique application
        strength         : strength label string (if scorer provided)
        memorability     : memorability label string (if scorer provided)
        strength_score   : float 0.0-1.0 (if scorer provided)
        memorability_score : float 0.0-1.0 (if scorer provided)
        combined_score   : float 0.0-1.0 (if scorer provided)
        scored_variants  : variants list with scores added (if scorer provided)
    """
    # Support both year= and year_hint= — app.py passes year as int
    if year is not None and year_hint is None:
        year_hint = year  # pass as-is, _inject_entropy handles int or str

    # Run the full transformer pipeline
    result = transform_passphrase(phrase, year_hint=year_hint)

    # Default scores
    result['strength']           = 'unknown'
    result['memorability']       = 'unknown'
    result['strength_score']     = 0.0
    result['memorability_score'] = 0.0
    result['combined_score']     = 0.0
    result['scored_variants']    = result['variants']

    # If scorer is provided, score the primary password and all variants
    if scorer is not None:
        try:
            primary_scores = scorer.score(result['password'])
            result['strength']           = primary_scores.get('strength', 'unknown')
            result['memorability']       = primary_scores.get('memorability', 'unknown')
            result['strength_score']     = primary_scores.get('strength_score', 0.0)
            result['memorability_score'] = primary_scores.get('memorability_score', 0.0)
            result['combined_score']     = primary_scores.get('combined_score', 0.0)

            # Score each variant too
            scored_variants = []
            for v in result['variants']:
                try:
                    v_scores = scorer.score(v['password'])
                    v['strength']           = v_scores.get('strength', 'unknown')
                    v['memorability']       = v_scores.get('memorability', 'unknown')
                    v['strength_score']     = v_scores.get('strength_score', 0.0)
                    v['memorability_score'] = v_scores.get('memorability_score', 0.0)
                    v['combined_score']     = v_scores.get('combined_score', 0.0)
                except Exception:
                    v['strength']           = 'unknown'
                    v['memorability']       = 'unknown'
                    v['combined_score']     = 0.0
                scored_variants.append(v)

            # Re-sort variants by combined score descending
            scored_variants.sort(
                key=lambda x: x.get('combined_score', 0.0), reverse=True
            )
            result['scored_variants'] = scored_variants

        except Exception as e:
            # Scorer failed — return result without scores rather than crashing
            result['scorer_error'] = str(e)

    # ── Add keys that app.py /transform route expects ────────────────
    # app.py does: result["candidates"], result["pipeline"],
    #              result["explanation"], result["strength_tips"], result["ml_scores"]

    # candidates — JS expects a FLAT LIST of password strings
    result["candidates"] = [
        v["password"] for v in result.get("variants", [])
    ]

    # pipeline — JS expects a FLAT DICT with exact keys
    log = result.get("pipeline_log", [])
    original_phrase = log[0].get("input", "")         if len(log) > 0 else ""
    filtered_words  = log[0].get("retained", [])      if len(log) > 0 else []
    anchor_words    = log[1].get("output", [])         if len(log) > 1 else []
    core            = log[3].get("output", "")         if len(log) > 3 else ""
    final_pwd       = log[4].get("output", result.get("password","")) if len(log) > 4 else result.get("password","")
    tech_details    = log[2].get("details", [])        if len(log) > 2 else []
    substitution    = "; ".join(tech_details) if tech_details else "Standard pipeline"
    suffix_info     = str(log[4].get("suffix",""))     if len(log) > 4 else ""
    special_char    = str(log[4].get("special_char","")) if len(log) > 4 else ""
    result["pipeline"] = {
        "original_phrase": original_phrase,
        "filtered_words":  filtered_words if isinstance(filtered_words, list) else [str(filtered_words)],
        "anchor_words":    anchor_words   if isinstance(anchor_words, list)   else [str(anchor_words)],
        "substitution":    substitution,
        "core":            core,
        "numeric_suffix":  suffix_info,
        "special_char":    special_char,
        "final":           final_pwd,
    }

    # explanation — human readable summary of what was done
    techniques_used = result.get("techniques_used", [])
    tech_names = {
        "T1_acronym":  "Acronym Injection",
        "T2_phonetic": "Phonetic Mirroring",
        "T3_visual":   "Visual Shape Substitution",
        "T4_reverse":  "Reverse Anchor",
        "T5_leet":     "Leet Substitution",
    }
    applied_names = [tech_names.get(t, t) for t in techniques_used]
    result["explanation"] = (
        f"Applied {len(applied_names)} enhancement technique(s): "
        + ", ".join(applied_names)
        + ". Password anchored to your personal phrase for easy recall."
        if applied_names else
        "Password generated from your personal phrase."
    )

    # strength_tips — actionable tips for the UI
    result["strength_tips"] = [
        "Your phrase is the memory anchor — you can always reconstruct this password.",
        f"Techniques applied: {', '.join(applied_names) if applied_names else 'standard pipeline'}.",
        "Store this in the vault below for secure zero-knowledge storage.",
    ]

    # ml_scores — dict of scores (populated if scorer ran successfully)
    result["ml_scores"] = {
        "strength":           result.get("strength", "unknown"),
        "memorability":       result.get("memorability", "unknown"),
        "strength_score":     result.get("strength_score", 0.0),
        "memorability_score": result.get("memorability_score", 0.0),
        "combined_score":     result.get("combined_score", 0.0),
    }

    return result
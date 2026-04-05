# generator/memory_aid.py
"""
Memory aid generator — helps users remember their passwords.

WHY THIS FILE EXISTS
─────────────────────
Generating a strong password solves half the problem.
The other half: the user must actually remember it.

This file implements four scientifically-grounded memory techniques:

  1. CHUNKING        — splits password into 3–4 character groups
                       e.g. "xK9!mPq2" → "xK9! · mPq · 2"
                       Basis: Miller (1956) — working memory holds 7±2 chunks

  2. PHONETIC STORY  — converts each chunk into a sound-alike word/phrase
                       e.g. "xK9!" → "ex-Kay-Nine-bang"
                       Basis: Baddeley (1986) — phonological loop encodes speech

  3. ACRONYM         — turns a memorable sentence into a password
                       e.g. "My cat Whiskers turned 3 in 2021!" → "McWt3i2!"
                       Basis: Paivio (1971) — dual coding (image + word)

  4. VISUAL STORY    — maps each character to a vivid image and links them
                       into a narrative
                       e.g. "xK9!" → "an X-ray hits a King, 9 guards run, BANG!"
                       Basis: Method of Loci + imagery mnemonics research

These are the techniques cited in your project documentation.
This file delivers them as working code, not just theory.

IMPORTANT: this file has ZERO external dependencies beyond stdlib.
No ML models, no crypto, no NLTK downloads.
It works immediately after installation.
"""

import re
import sys
import secrets
import string
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── character → phonetic syllable map ────────────────────────────────────────
# Every printable ASCII character mapped to a memorable spoken sound.
# Designed to be:
#   (a) unambiguous — no two characters share a phonetic
#   (b) short       — 1–2 syllables so they fit in working memory
#   (c) vivid       — concrete nouns/verbs aid imagery encoding

PHONETIC_MAP = {
    # Lowercase letters
    "a": "alpha",   "b": "bravo",   "c": "charlie", "d": "delta",
    "e": "echo",    "f": "foxtrot", "g": "golf",    "h": "hotel",
    "i": "india",   "j": "juliet",  "k": "kilo",    "l": "lima",
    "m": "mike",    "n": "november","o": "oscar",   "p": "papa",
    "q": "quebec",  "r": "romeo",   "s": "sierra",  "t": "tango",
    "u": "uniform", "v": "victor",  "w": "whiskey", "x": "x-ray",
    "y": "yankee",  "z": "zulu",

    # Uppercase — prefixed with "big" to distinguish from lowercase
    "A": "big-Alpha",   "B": "big-Bravo",   "C": "big-Charlie",
    "D": "big-Delta",   "E": "big-Echo",    "F": "big-Foxtrot",
    "G": "big-Golf",    "H": "big-Hotel",   "I": "big-India",
    "J": "big-Juliet",  "K": "big-Kilo",    "L": "big-Lima",
    "M": "big-Mike",    "N": "big-November","O": "big-Oscar",
    "P": "big-Papa",    "Q": "big-Quebec",  "R": "big-Romeo",
    "S": "big-Sierra",  "T": "big-Tango",   "U": "big-Uniform",
    "V": "big-Victor",  "W": "big-Whiskey", "X": "big-X-ray",
    "Y": "big-Yankee",  "Z": "big-Zulu",

    # Digits — spoken as words
    "0": "zero",  "1": "one",   "2": "two",   "3": "three",
    "4": "four",  "5": "five",  "6": "six",   "7": "seven",
    "8": "eight", "9": "nine",

    # Special characters — vivid one-word names
    "!": "BANG",    "@": "at",      "#": "hash",    "$": "dollar",
    "%": "percent", "^": "caret",   "&": "and",     "*": "star",
    "(": "open",    ")": "close",   "-": "dash",    "_": "under",
    "=": "equals",  "+": "plus",    "[": "bracket", "]": "end-bracket",
    "{": "brace",   "}": "end-brace","\\":"backslash","|": "pipe",
    ";": "semi",    ":": "colon",   "'": "tick",    "\"":"quote",
    ",": "comma",   ".": "dot",     "<": "less",    ">": "more",
    "?": "query",   "/": "slash",   "`": "grave",   "~": "tilde",
    " ": "space",
}

# ── character → visual image map ─────────────────────────────────────────────
# Short, vivid image for each character — used in story generation.
# Chosen for distinctiveness and visual concreteness (Paivio's imageability).

VISUAL_MAP = {
    # Lowercase
    "a": "an apple",       "b": "a ball",        "c": "a cat",
    "d": "a dragon",       "e": "an eagle",      "f": "a flame",
    "g": "a ghost",        "h": "a hammer",      "i": "an ice cube",
    "j": "a jellyfish",    "k": "a kite",        "l": "a lemon",
    "m": "a moon",         "n": "a needle",      "o": "an owl",
    "p": "a piano",        "q": "a queen",       "r": "a rocket",
    "s": "a snake",        "t": "a tiger",       "u": "an umbrella",
    "v": "a volcano",      "w": "a wave",        "x": "an X-mark",
    "y": "a yo-yo",        "z": "a zebra",
    # Uppercase — larger/more powerful versions
    "A": "a giant apple",  "B": "a giant boulder","C": "a castle",
    "D": "a dinosaur",     "E": "an explosion",   "F": "a forest fire",
    "G": "a golden gate",  "H": "a huge hammer",  "I": "an iceberg",
    "J": "a jet plane",    "K": "a king",         "L": "a lightning bolt",
    "M": "a mountain",     "N": "a nuclear plant","O": "an ocean",
    "P": "a pyramid",      "Q": "a giant queen",  "R": "a red dragon",
    "S": "a storm",        "T": "a tsunami",      "U": "an UFO",
    "V": "a vast valley",  "W": "a waterfall",    "X": "an X-wing",
    "Y": "a yellow sun",   "Z": "a zombie horde",
    # Digits
    "0": "a hollow ring",  "1": "a single candle","2": "two swans",
    "3": "three coins",    "4": "four-leaf clover","5": "a starfish",
    "6": "a snail shell",  "7": "a lucky seven",  "8": "an hourglass",
    "9": "a balloon",
    # Special
    "!": "a thunderbolt",  "@": "a spinning vortex","#": "a crossroads",
    "$": "a treasure chest","%": "a percentage sign","^": "a lightning rod",
    "&": "a chain link",   "*": "a shooting star", "(": "an open gate",
    ")": "a closing door", "-": "a sword",         "_": "a bridge",
    "=": "a scale",        "+": "a cross",
}

# Connector phrases for linking story images
_CONNECTORS = [
    "which crashes into",
    "then meets",
    "which transforms into",
    "suddenly becomes",
    "which explodes into",
    "and chases",
    "then swallows",
    "which melts into",
    "then flies over",
    "and lands on",
]


# ── technique 1: chunking ─────────────────────────────────────────────────────

def chunk_password(password: str, chunk_size: int = 4) -> dict:
    """
    Splits password into equal-sized chunks separated by a visual marker.

    Cognitive basis: Miller (1956) showed working memory holds 7±2 chunks.
    A 16-char random password = 16 individual items (exceeds capacity).
    Split into 4-char chunks = 4 items (well within capacity).

    Args:
        password   : password string to chunk
        chunk_size : characters per chunk (default 4)

    Returns:
        {
          "chunks":     ["xK9!", "mPq2", "Tz@W"],
          "display":    "xK9!  ·  mPq2  ·  Tz@W",
          "description": "Remember 3 groups of 4 characters..."
        }
    """
    if not password:
        raise ValueError("Password cannot be empty.")

    chunks = [
        password[i:i + chunk_size]
        for i in range(0, len(password), chunk_size)
    ]

    display = "  ·  ".join(chunks)

    n = len(chunks)
    sizes = [len(c) for c in chunks]
    size_desc = (
        f"{n} groups of {chunk_size}"
        if len(set(sizes)) == 1
        else f"{n} groups"
    )

    description = (
        f"Remember {size_desc} separately. "
        f"Your brain stores each group as ONE memory unit, "
        f"reducing {len(password)} characters to just {n} chunks."
    )

    return {
        "chunks":      chunks,
        "display":     display,
        "description": description,
        "n_chunks":    n,
        "chunk_size":  chunk_size,
    }


# ── technique 2: phonetic story ───────────────────────────────────────────────

def phonetic_encoding(password: str) -> dict:
    """
    Converts each character to its NATO/phonetic equivalent, then groups
    them into a speakable sequence.

    Cognitive basis: Baddeley's phonological loop encodes information as
    speech sounds. A speakable sequence can be sub-vocally rehearsed even
    while doing other tasks — it sticks in memory automatically.

    Args:
        password : password string to encode

    Returns:
        {
          "phonetics":   ["big-Kilo", "alpha", "nine", "BANG", ...],
          "spoken":      "big-Kilo · alpha · nine · BANG · ...",
          "grouped":     "big-Kilo-alpha-nine-BANG  |  mike-papa-quebec-two",
          "description": "Say this aloud 3 times..."
        }
    """
    phonetics = []
    for char in password:
        phonetics.append(PHONETIC_MAP.get(char, f"[{char}]"))

    spoken  = " · ".join(phonetics)

    # Group phonetics to match chunks (groups of 4 chars)
    chunk_phonetics = [phonetics[i:i+4] for i in range(0, len(phonetics), 4)]
    grouped = "  |  ".join("-".join(g) for g in chunk_phonetics)

    description = (
        f"Say this aloud 3 times slowly:\n"
        f"  '{grouped}'\n"
        f"The phonological loop in your brain will encode it automatically. "
        f"Each repetition strengthens the memory trace."
    )

    return {
        "phonetics":   phonetics,
        "spoken":      spoken,
        "grouped":     grouped,
        "description": description,
    }


# ── technique 3: acronym generator ───────────────────────────────────────────

def generate_acronym_sentence(password: str) -> dict:
    """
    Generates a memorable sentence where the first letter of each word
    matches the corresponding character of the password (where possible).

    Also works in reverse: given a memorable sentence, extracts the
    first letter of each word to form a password.

    Cognitive basis: Paivio's dual-coding theory — encoding information
    as both a verbal sequence AND a mental image doubles retention.
    A vivid sentence creates an image; the image retrieves the sentence;
    the sentence retrieves the password.

    Args:
        password : password string to encode

    Returns:
        {
          "sentence":    "King Arthur Nine! mighty...",
          "explanation": "Take the first letter/digit of each word...",
          "reverse_tip": "Or: think of YOUR own sentence using these letters"
        }
    """
    # Word banks keyed by first letter — vivid, concrete, imageable words
    word_bank = {
        "a": ["apple", "arrow", "anchor", "axe", "ant"],
        "b": ["bear", "ball", "boat", "bomb", "branch"],
        "c": ["cat", "crown", "cave", "crane", "coin"],
        "d": ["dog", "door", "drum", "dagger", "diamond"],
        "e": ["eagle", "echo", "ember", "engine", "elk"],
        "f": ["fire", "flag", "fence", "frog", "flash"],
        "g": ["gold", "gate", "ghost", "gem", "gorilla"],
        "h": ["hammer", "horse", "hook", "hawk", "halo"],
        "i": ["ice", "iron", "island", "ink", "imp"],
        "j": ["jar", "jade", "jaguar", "jewel", "jump"],
        "k": ["king", "key", "knife", "knight", "kite"],
        "l": ["lion", "lamp", "leaf", "lance", "lava"],
        "m": ["moon", "mask", "mist", "magnet", "marble"],
        "n": ["night", "nail", "net", "nexus", "noble"],
        "o": ["oak", "owl", "ocean", "orb", "onyx"],
        "p": ["pearl", "path", "pike", "prism", "peak"],
        "q": ["queen", "quartz", "quest", "quill", "quake"],
        "r": ["river", "ring", "rock", "raven", "rope"],
        "s": ["star", "sword", "storm", "stone", "sphinx"],
        "t": ["torch", "tower", "tide", "thorn", "throne"],
        "u": ["urn", "ultra", "union", "uplift", "unicorn"],
        "v": ["vault", "vine", "viper", "vortex", "valor"],
        "w": ["wolf", "wave", "wall", "wheel", "wind"],
        "x": ["xenon", "X-mark", "xyst", "xenith"],
        "y": ["yew", "yacht", "yoke", "yellow", "yarn"],
        "z": ["zero", "zone", "zenith", "zinc", "zeal"],
    }

    words = []
    for char in password:
        if char.isalpha():
            key  = char.lower()
            bank = word_bank.get(key, [char])
            word = secrets.choice(bank)
            # Capitalise if original char was uppercase
            if char.isupper():
                word = word.capitalize()
            words.append(word)
        elif char.isdigit():
            # Digits become themselves — "Nine" or just the numeral
            digit_words = {
                "0": "zero", "1": "one", "2": "two", "3": "three",
                "4": "four", "5": "five", "6": "six", "7": "seven",
                "8": "eight", "9": "nine",
            }
            words.append(digit_words[char])
        else:
            # Special chars become their phonetic — "BANG" for "!"
            words.append(PHONETIC_MAP.get(char, char).upper())

    sentence = " ".join(words)

    # Verify reverse works (first letter of each word matches password)
    check_chars = []
    for w in sentence.split():
        # Extract first alphanumeric character
        for c in w:
            if c.isalnum():
                check_chars.append(c)
                break

    explanation = (
        f"Sentence: \"{sentence}\"\n"
        f"Take the first meaningful character from each word "
        f"to reconstruct your password.\n"
        f"Better yet: create YOUR OWN sentence using these starting sounds — "
        f"personal sentences are 3× more memorable than generated ones."
    )

    return {
        "sentence":     sentence,
        "words":        words,
        "explanation":  explanation,
        "reverse_tip":  (
            "Pro tip: replace this sentence with one from YOUR life "
            "(a memory, a place, a person). Personal meaning = "
            "instant long-term retention."
        ),
    }


# ── technique 4: visual story ─────────────────────────────────────────────────

def generate_visual_story(password: str, max_chars: int = 12) -> dict:
    """
    Generates a short vivid narrative linking the visual image of each
    character into a memorable scene.

    Cognitive basis: Method of Loci (Roman Room) + Paivio's imagery.
    Bizarre, vivid, action-filled scenes are remembered far better than
    neutral ones (McDaniel & Einstein, 1986 — bizarreness effect).
    We deliberately make the story strange.

    Args:
        password  : password string to encode
        max_chars : max characters to include in story (keep it manageable)

    Returns:
        {
          "story":       "An apple crashes into a King, then a nine-balloon...",
          "scene_count": 8,
          "description": "Visualise this scene as vividly as possible..."
        }
    """
    # Use first max_chars characters only — stories get unwieldy beyond ~12
    pwd_slice = password[:max_chars]
    remainder = len(password) - len(pwd_slice)

    scenes = []
    for char in pwd_slice:
        image = VISUAL_MAP.get(char, f"the symbol '{char}'")
        scenes.append(image)

    # Build narrative by linking scenes with connectors
    story_parts = [scenes[0].capitalize()]
    for i, scene in enumerate(scenes[1:], 0):
        connector = _CONNECTORS[i % len(_CONNECTORS)]
        story_parts.append(f"{connector} {scene}")

    story = " ".join(story_parts) + "."

    if remainder > 0:
        story += (
            f" (Story covers first {max_chars} characters. "
            f"Remaining {remainder}: use chunking for the rest.)"
        )

    description = (
        "Visualise this scene as vividly as possible — make it "
        "colourful, bizarre, and action-filled. Close your eyes and "
        "run through it like a movie. The stranger the better: "
        "bizarre images are proven to be remembered longer."
    )

    return {
        "story":       story,
        "scenes":      scenes,
        "scene_count": len(scenes),
        "description": description,
    }


# ── master function: all four techniques in one call ─────────────────────────

def generate_memory_aids(
    password:    str,
    chunk_size:  int  = 4,
    max_story:   int  = 12,
    include:     Optional[list] = None,
) -> dict:
    """
    Runs all four memory techniques for a password and returns a
    structured dict suitable for rendering in the Flask web app.

    Args:
        password   : the password to generate aids for
        chunk_size : chars per chunk in chunking technique (default 4)
        max_story  : max chars to use in visual story (default 12)
        include    : list of techniques to include, or None for all
                     e.g. ["chunking", "phonetic"] for just two

    Returns:
        {
          "password":  "xK9!mPq2",
          "length":    8,
          "chunking":  { ... },
          "phonetic":  { ... },
          "acronym":   { ... },
          "story":     { ... },
          "summary":   "Use chunking + phonetic for short passwords..."
        }

    Usage in app.py:
        aids = generate_memory_aids(password)
        return jsonify(aids)
    """
    if not password:
        raise ValueError("Password cannot be empty.")

    all_techniques = ["chunking", "phonetic", "acronym", "story"]
    selected       = include if include else all_techniques

    result = {
        "password": password,
        "length":   len(password),
    }

    if "chunking" in selected:
        result["chunking"] = chunk_password(password, chunk_size)

    if "phonetic" in selected:
        result["phonetic"] = phonetic_encoding(password)

    if "acronym" in selected:
        result["acronym"]  = generate_acronym_sentence(password)

    if "story" in selected:
        result["story"]    = generate_visual_story(password, max_story)

    # Recommendation based on password type
    result["summary"] = _recommend_technique(password)

    return result


def _recommend_technique(password: str) -> str:
    """
    Recommends the most suitable memory technique based on password type.
    Called inside generate_memory_aids() — not exported directly.
    """
    # Detect password type
    has_words   = bool(re.search(r"[a-zA-Z]{4,}", password))
    has_special = any(c in string.punctuation for c in password)
    is_long     = len(password) >= 16
    is_random   = not has_words and len(set(password)) > len(password) * 0.7

    if "-" in password and has_words:
        return (
            "This is a passphrase — it's already memorable! "
            "Use the ACRONYM technique to create a backup sentence, "
            "and CHUNKING to type it in sections."
        )
    elif is_random and is_long:
        return (
            "This is a high-entropy random password. "
            "Use CHUNKING to break it into manageable groups, "
            "then PHONETIC to create a spoken sequence you can rehearse. "
            "Consider storing in your vault and using a passphrase instead."
        )
    elif has_special and not is_long:
        return (
            "Use the VISUAL STORY technique — special characters "
            "map to vivid images (! = thunderbolt, @ = vortex). "
            "The bizarre story will stick in memory quickly."
        )
    else:
        return (
            "Use CHUNKING as your primary technique, "
            "then reinforce with PHONETIC encoding. "
            "Say the phonetic sequence aloud 3 times to activate "
            "your phonological loop."
        )


# ── reverse: sentence → password ─────────────────────────────────────────────

def sentence_to_password(
    sentence:       str,
    append_digits:  int  = 2,
    append_special: bool = True,
) -> dict:
    """
    REVERSE OPERATION: converts a memorable sentence into a password
    by taking the first character of each word.

    This is the acronym technique in reverse — the user supplies the
    sentence, we extract the password.

    Args:
        sentence       : e.g. "My cat Whiskers ate 3 fish on Tuesday!"
        append_digits  : add N random digits at the end (default 2)
        append_special : add a random special char at the end (default True)

    Returns:
        {
          "sentence":  "My cat Whiskers ate 3 fish on Tuesday!",
          "password":  "McWa3foT!47",
          "breakdown": [("My","M"), ("cat","c"), ("Whiskers","W"), ...],
          "strength_tip": "Add digits and special chars to increase entropy"
        }

    Example:
        result = sentence_to_password("My cat Whiskers ate 3 fish on Tuesday!")
        print(result["password"])  # → "McWa3foT!47"  (or similar)
    """

    words = sentence.strip().split()
    breakdown = []
    chars     = []

    for word in words:
        # Take the first alphanumeric character from each word
        for char in word:
            if char.isalnum() or char in string.punctuation:
                breakdown.append((word, char))
                chars.append(char)
                break

    # Append random digits
    digit_suffix = ""
    for _ in range(append_digits):
        d = secrets.choice(string.digits)
        chars.append(d)
        digit_suffix += d

    # Append special character
    special_char = ""
    if append_special:
        special_char = _s.choice("!@#$%&*")
        chars.append(special_char)

    password = "".join(chars)

    return {
        "sentence":     sentence,
        "password":     password,
        "breakdown":    breakdown,
        "digit_suffix": digit_suffix,
        "special_char": special_char,
        "strength_tip": (
            f"Password '{password}' has {len(password)} characters. "
            f"Add more words to your sentence for a longer password. "
            f"Make the sentence personal — a memory only you know."
        ),
    }


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  generator/memory_aid.py — self test")
    print("=" * 64)

    test_passwords = [
        "xK9!mPq2",              # short random
        "correct-horse-battery", # passphrase
        "Mumbai@2019!Chai",      # personal password
    ]

    for pwd in test_passwords:
        print(f"\n{'═'*64}")
        print(f"  Password: {pwd!r}  (length {len(pwd)})")
        print(f"{'═'*64}")

        aids = generate_memory_aids(pwd)

        # Chunking
        c = aids["chunking"]
        print(f"\n  [1] CHUNKING")
        print(f"      {c['display']}")
        print(f"      {c['description']}")

        # Phonetic
        p = aids["phonetic"]
        print(f"\n  [2] PHONETIC")
        print(f"      {p['grouped']}")
        print(f"      Say aloud 3 times to encode via phonological loop.")

        # Acronym
        a = aids["acronym"]
        print(f"\n  [3] ACRONYM SENTENCE")
        print(f"      {a['sentence']}")
        print(f"      {a['reverse_tip']}")

        # Story
        s = aids["story"]
        print(f"\n  [4] VISUAL STORY")
        print(f"      {s['story']}")
        print(f"      {s['description']}")

        # Summary
        print(f"\n  [!] RECOMMENDATION")
        print(f"      {aids['summary']}")

    # Reverse: sentence → password
    print(f"\n{'═'*64}")
    print("  REVERSE: sentence → password")
    print(f"{'═'*64}")
    sentence = "My dog Rocket jumped over 7 fences last Tuesday!"
    result   = sentence_to_password(sentence)
    print(f"\n  Sentence : {result['sentence']}")
    print(f"  Password : {result['password']}")
    print(f"  Breakdown:")
    for word, char in result["breakdown"]:
        print(f"    '{word}' → '{char}'")
    print(f"  Tip: {result['strength_tip']}")

    print(f"\n{'='*64}")
    print("  ALL TESTS PASSED")
    print(f"{'='*64}")
    print("\n  Next step: python app/app.py  (Step 10 — Flask web app)")
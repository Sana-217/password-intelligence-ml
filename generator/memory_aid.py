# generator/memory_aid.py
"""
Memory aid generator — helps users remember their passwords.

WHY THIS FILE EXISTS
─────────────────────
Generating a strong password solves half the problem.
The other half: the user must actually remember it.

This file implements four scientifically-grounded memory techniques:

  1. CHUNKING        — splits password into 3–4 character groups
                       Basis: Miller (1956) — working memory holds 7±2 chunks

  2. PHONETIC STORY  — converts each chunk into a sound-alike word/phrase
                       Basis: Baddeley (1986) — phonological loop encodes speech

  3. ACRONYM         — turns a memorable sentence into a password
                       Basis: Paivio (1971) — dual coding (image + word)

  4. VISUAL STORY    — word-aware: splits CamelCase, recognises real words,
                       maps them to vivid related scenes, links into a story
                       Basis: Method of Loci + McDaniel & Einstein (1986)
                       bizarreness effect

IMPORTANT: this file has ZERO external dependencies beyond stdlib.
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
PHONETIC_MAP = {
    "a": "alpha",   "b": "bravo",   "c": "charlie", "d": "delta",
    "e": "echo",    "f": "foxtrot", "g": "golf",    "h": "hotel",
    "i": "india",   "j": "juliet",  "k": "kilo",    "l": "lima",
    "m": "mike",    "n": "november","o": "oscar",   "p": "papa",
    "q": "quebec",  "r": "romeo",   "s": "sierra",  "t": "tango",
    "u": "uniform", "v": "victor",  "w": "whiskey", "x": "x-ray",
    "y": "yankee",  "z": "zulu",
    "A": "big-Alpha",   "B": "big-Bravo",   "C": "big-Charlie",
    "D": "big-Delta",   "E": "big-Echo",    "F": "big-Foxtrot",
    "G": "big-Golf",    "H": "big-Hotel",   "I": "big-India",
    "J": "big-Juliet",  "K": "big-Kilo",    "L": "big-Lima",
    "M": "big-Mike",    "N": "big-November","O": "big-Oscar",
    "P": "big-Papa",    "Q": "big-Quebec",  "R": "big-Romeo",
    "S": "big-Sierra",  "T": "big-Tango",   "U": "big-Uniform",
    "V": "big-Victor",  "W": "big-Whiskey", "X": "big-X-ray",
    "Y": "big-Yankee",  "Z": "big-Zulu",
    "0": "zero",  "1": "one",   "2": "two",   "3": "three",
    "4": "four",  "5": "five",  "6": "six",   "7": "seven",
    "8": "eight", "9": "nine",
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

# ── character → visual image map (for single chars) ──────────────────────────
VISUAL_MAP = {
    "a": "an apple",       "b": "a ball",        "c": "a cat",
    "d": "a dragon",       "e": "an eagle",      "f": "a flame",
    "g": "a ghost",        "h": "a hammer",      "i": "an ice cube",
    "j": "a jellyfish",    "k": "a kite",        "l": "a lemon",
    "m": "a moon",         "n": "a needle",      "o": "an owl",
    "p": "a piano",        "q": "a queen",       "r": "a rocket",
    "s": "a snake",        "t": "a tiger",       "u": "an umbrella",
    "v": "a volcano",      "w": "a wave",        "x": "an X-mark",
    "y": "a yo-yo",        "z": "a zebra",
    "A": "a giant apple",  "B": "a giant boulder","C": "a castle",
    "D": "a dinosaur",     "E": "an explosion",   "F": "a forest fire",
    "G": "a golden gate",  "H": "a huge hammer",  "I": "an iceberg",
    "J": "a jet plane",    "K": "a king",         "L": "a lightning bolt",
    "M": "a mountain",     "N": "a nuclear plant","O": "an ocean",
    "P": "a pyramid",      "Q": "a giant queen",  "R": "a red dragon",
    "S": "a storm",        "T": "a tsunami",      "U": "an UFO",
    "V": "a vast valley",  "W": "a waterfall",    "X": "an X-wing",
    "Y": "a yellow sun",   "Z": "a zombie horde",
    "0": "a hollow ring",  "1": "a single candle","2": "two swans",
    "3": "three coins",    "4": "four-leaf clover","5": "a starfish",
    "6": "a snail shell",  "7": "a lucky seven",  "8": "an hourglass",
    "9": "a balloon",
    "!": "a thunderbolt",  "@": "a spinning vortex","#": "a crossroads",
    "$": "a treasure chest","%": "a whirlwind",   "^": "a lightning rod",
    "&": "a chain link",   "*": "a shooting star", "(": "an open gate",
    ")": "a closing door", "-": "a sword",         "_": "a bridge",
    "=": "a balance scale","+": "a glowing cross",
}

# ── word image bank ───────────────────────────────────────────────────────────
# Maps recognisable password words to vivid, specific visual scenes.
WORD_IMAGES = {
    "landed":  "a plane crash-landing on a runway",
    "landing": "a spacecraft touching down",
    "land":    "a vast open landscape stretching to the horizon",
    "dream":   "a swirling luminous dream cloud",
    "dreamed": "a dreamer floating weightlessly in clouds",
    "dr":      "a doctor rushing out in a white coat",
    "boj":     "a mysterious figure named Boj",
    "boy":     "a young boy sprinting at full speed",
    "bruno":   "Bruno — a loyal dog wagging its tail wildly",
    "dog":     "a huge barking dog",
    "dawg":    "a cool dog wearing sunglasses",
    "correct": "a giant glowing green checkmark",
    "horse":   "a galloping white horse at full speed",
    "battery": "a massive crackling power battery",
    "mumbai":  "the Mumbai skyline blazing at night",
    "chai":    "a steaming giant cup of chai",
    "coffee":  "a steaming coffee mug the size of a house",
    "fire":    "a raging wildfire consuming everything",
    "king":    "a golden-crowned king on a massive throne",
    "queen":   "a jewelled queen raising her sceptre",
    "rocky":   "a boxer punching the air in triumph",
    "rock":    "a massive rolling boulder",
    "storm":   "a raging thunderstorm splitting the sky",
    "love":    "a giant glowing red heart pulsing",
    "star":    "a blazing star exploding into light",
    "moon":    "a full moon glowing over a dark ocean",
    "sun":     "a blinding golden sun filling the sky",
    "tiger":   "a snarling tiger mid-leap",
    "wolf":    "a howling wolf on a mountain cliff",
    "dragon":  "a fire-breathing dragon circling a castle",
    "ice":     "a frozen glacier cracking with a boom",
    "snake":   "a giant coiled cobra rising to strike",
    "angel":   "a glowing angel with wings spread wide",
    "devil":   "a red devil laughing with a pitchfork",
    "dark":    "an endless void of absolute darkness",
    "black":   "a black hole consuming everything",
    "white":   "a blinding white flash erasing everything",
    "gold":    "a mountain of shining gold coins",
    "power":   "a bolt of raw electrical power shattering a wall",
    "super":   "a superhero bursting through a concrete wall",
    "shadow":  "a creeping shadow with no owner",
    "thunder": "a thunderclap shattering the sky open",
    "magic":   "an explosion of colourful magic sparks",
    "ninja":   "a shadow ninja vanishing into black smoke",
    "warrior": "an armoured warrior raising a flaming sword",
    "world":   "the entire spinning earth viewed from space",
    "city":    "a towering neon city skyline at midnight",
    "ocean":   "an endless roaring ocean under a storm",
    "river":   "a rushing river carving through stone",
    "sky":     "an infinite open sky at sunrise",
    "night":   "a pitch-black starry night",
    "rose":    "a single blood-red rose",
    "flower":  "a blooming flower field stretching endlessly",
    "rainbow": "a vivid double rainbow arching overhead",
    "sword":   "a flaming sword raised high",
    "shield":  "an unbreakable golden shield",
    "crown":   "a heavy jewelled crown",
    "castle":  "a massive castle with a raised drawbridge",
    "forest":  "a dense enchanted forest glowing green",
    "blaze":   "a blaze consuming everything in orange flame",
    "frost":   "frost crystals spreading across a window in seconds",
    "nova":    "a supernova bursting with blinding light",
    "bolt":    "a lightning bolt striking the ground",
    "day":     "a bright sunny day with golden light",
    "dawn":    "the first light of dawn breaking over mountains",
    "dusk":    "a deep orange dusk settling over a city",
    "morning": "a crisp morning with mist rising",
    "evening": "a golden evening sky at sunset",
    "summer":  "a blazing summer day at the beach",
    "winter":  "a snowstorm burying everything in white",
    "spring":  "flowers bursting out of frozen ground",
    "river":   "a raging river breaking its banks",
    "lake":    "a perfectly still mirror-like lake",
    "friend":  "two friends laughing uncontrollably",
    "family":  "a warm family reunion with open arms",
    "sister":  "sisters hugging tightly",
    "brother": "brothers play-fighting",
    "mother":  "a mother cradling a glowing child",
    "father":  "a father lifting a child to the sky",
    "home":    "a warm glowing home in darkness",
    "heart":   "a giant pulsing neon heart",
    "smile":   "a huge beaming smile lighting up a face",
    "happy":   "a jumping figure radiating joy",
    "lucky":   "a four-leaf clover glowing gold",
    "sunny":   "a bright sunny meadow",
    "pizza":   "a giant melting pizza slice",
    "game":    "a glowing game controller in action",
    "music":   "musical notes dancing in the air",
    "dance":   "a dancer spinning in golden light",
    "monkey":  "a monkey swinging through jungle vines",
    "blood":   "a crimson river rushing fast",
    "bridge":  "a bridge suspended over an enormous chasm",
    "cave":    "a dark cave with glowing eyes inside",
    "tower":   "a crumbling stone tower in a storm",
    "flag":    "a flag whipping in a fierce wind",
    "cosmic":  "a cosmic explosion of galaxies colliding",
    "pilot":   "a pilot in full gear gripping the controls",
    "master":  "a wise old master meditating in silence",
    "prince":  "a young prince on horseback",
    "mars":    "the red surface of Mars at sunset",
    "earth":   "the blue Earth floating in space",
    "space":   "infinite black space filled with stars",
    "storm":   "a storm raging with lightning and howling wind",
    "flash":   "a blinding camera flash",
    "spark":   "a shower of golden sparks",
    "energy":  "crackling neon energy arcing in all directions",
    "cyber":   "a neon-lit cyberpunk cityscape",
    "crypto":  "glowing encrypted data streams",
    "tech":    "a circuit board lighting up in sequence",
    "code":    "lines of green code cascading down a screen",
    "hack":    "a hacker typing furiously in the dark",
    "secure":  "a locked vault sealed with light",
    "vault":   "a massive bank vault door swinging open",
    "pass":    "a glowing golden pass card",
    "guard":   "a stoic guard standing at a fortress gate",
}

# Story connectors
_WORD_ACTIONS = [
    "crashes into",   "leaps over",    "explodes beside",
    "spins around",   "swallows",      "transforms into",
    "chases",         "lands on",      "collides with",
    "melts into",     "flies past",    "devours",
    "launches at",    "circles around","crashes through",
]

_STORY_CONNECTORS = [
    "then", "suddenly", "next",
    "meanwhile", "instantly", "moments later",
]

# Story connector phrases for vivid linking
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
    Basis: Miller (1956) — working memory holds 7±2 chunks.
    """
    if not password:
        raise ValueError("Password cannot be empty.")

    chunks  = [password[i:i+chunk_size] for i in range(0, len(password), chunk_size)]
    display = "  ·  ".join(chunks)
    n       = len(chunks)
    sizes   = [len(c) for c in chunks]
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
    Converts each character to its NATO/phonetic equivalent.
    Basis: Baddeley (1986) — phonological loop encodes speakable sequences.
    """
    phonetics       = [PHONETIC_MAP.get(char, f"[{char}]") for char in password]
    spoken          = " · ".join(phonetics)
    chunk_phonetics = [phonetics[i:i+4] for i in range(0, len(phonetics), 4)]
    grouped         = "  |  ".join("-".join(g) for g in chunk_phonetics)
    description = (
        f"Say this aloud 3 times slowly:\n"
        f"  '{grouped}'\n"
        f"The phonological loop in your brain will encode it automatically."
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
    Generates a memorable sentence — one word per PASSWORD SEGMENT,
    not one word per character.

    How it works:
      1. Split password into meaningful segments using CamelCase split
         e.g. "LandedDreamJob95*" → ["Landed","Dream","Job","9","5","*"]
      2. For each WORD segment: pick a sentence word starting with the
         same first letter
         e.g. "Landed" → first letter "L" → "Lions"
              "Dream"  → first letter "D" → "Dive"
              "Job"    → first letter "J" → "Joyfully"
      3. For digits: write the digit word ("nine", "five")
      4. For special chars: write their phonetic ("BANG", "STAR")
      5. Assemble into a sentence that actually corresponds to the
         password structure — not 17 random words for 17 characters

    Result for "LandedDreamJob95*":
      "Lions Dive Joyfully, nine five, BANG!"

    Basis: Paivio (1971) — dual-coding (verbal + visual encoding).
    """
    # Word banks keyed by first letter
    word_bank = {
        "a": ["Amazing","Adventurous","Ancient","Agile","Awesome"],
        "b": ["Brave","Brilliant","Bold","Beautiful","Blazing"],
        "c": ["Courageous","Creative","Clever","Calm","Cosmic"],
        "d": ["Daring","Dynamic","Deep","Determined","Dazzling"],
        "e": ["Epic","Energetic","Elegant","Endless","Electric"],
        "f": ["Fierce","Flying","Fabulous","Fast","Fearless"],
        "g": ["Glowing","Grand","Golden","Graceful","Giant"],
        "h": ["Heroic","Humble","Hidden","Huge","Heavenly"],
        "i": ["Incredible","Infinite","Intense","Invisible","Icy"],
        "j": ["Joyful","Jumping","Just","Jolly","Jade"],
        "k": ["Keen","Kind","Kingly","Knightly","Known"],
        "l": ["Legendary","Luminous","Loyal","Lively","Lasting"],
        "m": ["Mighty","Magical","Massive","Mysterious","Majestic"],
        "n": ["Noble","Nimble","Natural","Neon","Noted"],
        "o": ["Outstanding","Orbital","Obscure","Oblique","Ominous"],
        "p": ["Powerful","Proud","Pure","Patient","Precise"],
        "q": ["Quick","Quiet","Quantum","Quirky","Questing"],
        "r": ["Radiant","Restless","Raging","Royal","Rising"],
        "s": ["Strong","Stellar","Swift","Serene","Supreme"],
        "t": ["Tremendous","Tall","Tenacious","Timeless","Tidal"],
        "u": ["Unstoppable","Unique","Ultra","Unbroken","Upward"],
        "v": ["Valiant","Vast","Vivid","Vigilant","Volcanic"],
        "w": ["Wild","Wise","Wonderful","Wandering","Winged"],
        "x": ["Xenial","X-factor","Xtra","X-ray","Xenith"],
        "y": ["Young","Youthful","Yearning","Yellow","Yielding"],
        "z": ["Zealous","Zesty","Zero-gravity","Zoned","Zenith"],
    }

    # Noun/verb banks for building complete sentence parts
    noun_bank = {
        "a": ["aces","arrows","atoms","axes","anchors"],
        "b": ["bears","bolts","beasts","blades","birds"],
        "c": ["cats","comets","claws","clouds","crests"],
        "d": ["dogs","dragons","dunes","darts","depths"],
        "e": ["eagles","embers","eons","edges","empires"],
        "f": ["flames","foxes","fists","flashes","forces"],
        "g": ["giants","gates","ghosts","gems","galaxies"],
        "h": ["hawks","hammers","heroes","horns","heights"],
        "i": ["ice","iron","islands","infernos","impulses"],
        "j": ["jaguars","jewels","jets","journeys","jolts"],
        "k": ["kings","knights","knives","kites","kingdoms"],
        "l": ["lions","lasers","lances","leaps","legends"],
        "m": ["mountains","moons","mists","minds","metals"],
        "n": ["ninjas","needles","nova","nexus","nights"],
        "o": ["owls","oceans","orbits","odes","onslaughts"],
        "p": ["phoenixes","peaks","powers","paths","pillars"],
        "q": ["queens","quests","quarks","quills","quakes"],
        "r": ["ravens","rockets","rivers","rings","realms"],
        "s": ["storms","stars","swords","shadows","spirits"],
        "t": ["tigers","tides","torches","titans","towers"],
        "u": ["unicorns","unfurls","unions","upheavals","urges"],
        "v": ["volcanoes","vipers","vaults","voids","visions"],
        "w": ["wolves","waves","warriors","winds","worlds"],
        "x": ["xenoliths","x-rays","xenons","xcellence","xplosions"],
        "y": ["yaks","yells","yields","yarns","yearnings"],
        "z": ["zebras","zones","zeniths","zigzags","zealots"],
    }

    digit_words = {
        "0": "zero", "1": "one",   "2": "two",   "3": "three", "4": "four",
        "5": "five", "6": "six",   "7": "seven",  "8": "eight", "9": "nine",
    }

    # Step 1: split password into word segments (CamelCase aware)
    segments = _split_camel_password(password)

    # Step 2: for each segment, generate ONE sentence component
    sentence_parts = []
    breakdown      = []  # for explanation display

    for seg in segments:
        if seg.isalpha() and len(seg) >= 2:
            # Word segment — generate "Adjective Noun" starting with same letter
            first_letter = seg[0].lower()
            adj  = secrets.choice(word_bank.get(first_letter, ["Amazing"]))
            noun = secrets.choice(noun_bank.get(first_letter, ["arrows"]))
            part = f"{adj} {noun}"
            sentence_parts.append(part)
            breakdown.append((seg, f"→ '{part}' (starts with '{seg[0]}')"))

        elif seg.isalpha() and len(seg) == 1:
            # Single letter — just one word
            first_letter = seg.lower()
            word = secrets.choice(word_bank.get(first_letter, [seg]))
            if seg.isupper():
                word = word.capitalize()
            sentence_parts.append(word)
            breakdown.append((seg, f"→ '{word}'"))

        elif seg.isdigit():
            # Digit — write digit word
            dword = digit_words.get(seg, seg)
            sentence_parts.append(dword)
            breakdown.append((seg, f"→ '{dword}'"))

        else:
            # Special character — phonetic name in caps
            phon = PHONETIC_MAP.get(seg, seg).upper()
            sentence_parts.append(phon)
            breakdown.append((seg, f"→ '{phon}'"))

    # Step 3: join into a readable sentence with punctuation
    # Add commas between word parts, exclamation at end
    if len(sentence_parts) <= 3:
        sentence = " ".join(sentence_parts) + "!"
    else:
        # Group: word segments together, then digits, then special
        word_parts    = [p for seg, p in zip(segments, sentence_parts) if seg.isalpha()]
        nonword_parts = [p for seg, p in zip(segments, sentence_parts) if not seg.isalpha()]
        if nonword_parts:
            sentence = " ".join(word_parts) + ", " + " ".join(nonword_parts) + "!"
        else:
            sentence = " ".join(sentence_parts) + "!"

    # Capitalise first letter
    sentence = sentence[0].upper() + sentence[1:]

    explanation = (
        f'Sentence: "{sentence}"\n\n'
        f"How it maps to your password:\n" +
        "\n".join(f"  '{seg}' {desc}" for seg, desc in breakdown) +
        f"\n\nBetter yet: create YOUR OWN sentence where each part "
        f"starts with the same letter as each password segment. "
        f"Personal sentences are 3× more memorable."
    )
    return {
        "sentence":    sentence,
        "words":       sentence_parts,
        "breakdown":   breakdown,
        "explanation": explanation,
        "reverse_tip": (
            "Pro tip: replace this sentence with one from YOUR life "
            "(a memory, a place, a person). Personal meaning = "
            "instant long-term retention."
        ),
    }


# ── VISUAL STORY HELPERS ──────────────────────────────────────────────────────

def _split_camel_password(text: str) -> list:
    """
    Split a CamelCase or mixed password into meaningful tokens.

    Examples:
        "LandedDr3@mBoj95*" → ["Landed","Dr","3","@","m","Boj","9","5","*"]
        "BrunoD0g2024!"     → ["Bruno","D","0","g","2","0","2","4","!"]
        "correct-horse"     → ["correct","-","horse"]
    """
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isalpha():
            j = i + 1
            while j < len(text) and text[j].isalpha():
                # Split at CamelCase boundary (lowercase → uppercase transition)
                if text[j].isupper() and j > i + 1:
                    break
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            tokens.append(c)
            i += 1
    return tokens


def _get_token_image(token: str) -> str:
    """
    Return the best vivid image for a password token.
    Priority order:
      1. Word image bank (specific vivid scenes for known words)
      2. Single alphabetic character → letter image
      3. Single digit or special → VISUAL_MAP
      4. Unknown multi-char token → named mysterious figure
    """
    lower = token.lower()

    # 1. Word image bank — specific vivid scenes
    if lower in WORD_IMAGES:
        return WORD_IMAGES[lower]

    # 2. Single alphabetic character
    if len(token) == 1 and token.isalpha():
        return VISUAL_MAP.get(token, f"the letter '{token}'")

    # 3. Single digit or special character
    if len(token) == 1:
        return VISUAL_MAP.get(token, f"the symbol '{token}'")

    # 4. Multi-char non-dictionary token — treat as a named character
    return f"a mysterious figure called '{token}'"


# ── technique 4: visual story (WORD-AWARE) ───────────────────────────────────

def generate_visual_story(password: str, max_chars: int = 14) -> dict:
    """
    Word-aware visual story generator.

    Unlike the old character-by-character approach, this function:
      1. Splits the password on CamelCase boundaries first
      2. Recognises whole words (Bruno, Dog, Landed, Dream...)
         and maps them to related vivid scenes
      3. Falls back to character images only for digits and special chars
      4. Links all scenes into a bizarre memorable narrative

    This means the story is ACTUALLY RELATED to the password content.

    Example:
        "LandedDr3@mBoj95*"
        → "A plane crash-landing on a runway then it crashes into
           a doctor in a white coat suddenly it leaps over three coins
           next it explodes beside a spinning vortex..."

    Basis: Method of Loci + Paivio (1971) imagery encoding +
           McDaniel & Einstein (1986) bizarreness effect.
    """
    if not password:
        return {
            "story":       "",
            "scenes":      [],
            "scene_count": 0,
            "description": "",
        }

    pwd_slice = password[:max_chars]
    remainder = len(password) - len(pwd_slice)

    # Step 1: split into meaningful tokens
    tokens = _split_camel_password(pwd_slice)

    # Step 2: convert each token to its vivid image
    scenes = []
    for token in tokens:
        image = _get_token_image(token)
        scenes.append({"segment": token, "image": image})

    if not scenes:
        return {
            "story":       "No story generated.",
            "scenes":      [],
            "scene_count": 0,
            "description": "",
        }

    # Step 3: build the narrative
    story_parts = [scenes[0]["image"].capitalize()]
    for idx, scene in enumerate(scenes[1:]):
        action    = _WORD_ACTIONS[idx % len(_WORD_ACTIONS)]
        connector = _STORY_CONNECTORS[idx % len(_STORY_CONNECTORS)]
        story_parts.append(f"{connector} it {action} {scene['image']}")

    story = " ".join(story_parts) + "."

    if remainder > 0:
        story += (
            f" (Story covers first {max_chars} characters. "
            f"Use chunking for the remaining {remainder}.)"
        )

    description = (
        "Visualise this scene as vividly as possible — in colour, "
        "with motion and exaggerated scale. The stranger the better: "
        "bizarre images are proven to be remembered far longer "
        "(McDaniel & Einstein, 1986)."
    )

    return {
        "story":       story,
        "scenes":      scenes,
        "scene_count": len(scenes),
        "description": description,
    }


# ── master function ───────────────────────────────────────────────────────────

def generate_memory_aids(
    password:   str,
    chunk_size: int           = 4,
    max_story:  int           = 14,
    include:    Optional[list] = None,
) -> dict:
    """
    Runs all four memory techniques for a password.
    Returns a structured dict suitable for rendering in Flask.
    """
    if not password:
        raise ValueError("Password cannot be empty.")

    all_techniques = ["chunking", "phonetic", "acronym", "story"]
    selected       = include if include else all_techniques

    result = {"password": password, "length": len(password)}

    if "chunking" in selected:
        result["chunking"] = chunk_password(password, chunk_size)
    if "phonetic" in selected:
        result["phonetic"] = phonetic_encoding(password)
    if "acronym"  in selected:
        result["acronym"]  = generate_acronym_sentence(password)
    if "story"    in selected:
        result["story"]    = generate_visual_story(password, max_story)

    result["summary"] = _recommend_technique(password)
    return result


def _recommend_technique(password: str) -> str:
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
            "then PHONETIC to create a spoken sequence you can rehearse."
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
    Converts a memorable sentence into a password by taking the
    first character of each word.
    """
    words     = sentence.strip().split()
    breakdown = []
    chars     = []

    for word in words:
        for char in word:
            if char.isalnum() or char in string.punctuation:
                breakdown.append((word, char))
                chars.append(char)
                break

    digit_suffix = ""
    for _ in range(append_digits):
        d = secrets.choice(string.digits)
        chars.append(d)
        digit_suffix += d

    special_char = ""
    if append_special:
        special_char = secrets.choice("!@#$%&*")
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
    test_passwords = [
        "LandedDr3@mBoj95*",
        "BrunoD0g2024!",
        "correct-horse-battery",
        "Mumbai@2019!Chai",
        "Rocky8Star2019*",
    ]

    for pwd in test_passwords:
        print(f"\n{'='*60}")
        print(f"Password: {pwd}")
        print(f"{'='*60}")
        aids = generate_memory_aids(pwd)

        print(f"\n[1] CHUNKING")
        print(f"    {aids['chunking']['display']}")

        print(f"\n[2] PHONETIC")
        print(f"    {aids['phonetic']['grouped']}")

        print(f"\n[3] ACRONYM")
        print(f"    {aids['acronym']['sentence']}")

        print(f"\n[4] VISUAL STORY")
        r = aids['story']
        print(f"    Tokens : {[s['segment'] for s in r['scenes']]}")
        print(f"    Story  : {r['story']}")

        print(f"\n[!] Recommendation: {aids['summary']}")
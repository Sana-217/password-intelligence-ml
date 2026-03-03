
import random
import re
import joblib
import pandas as pd
from preprocessing.feature_extraction import extract_features

# Load models
memorability_model = joblib.load("models/memorability_model.pkl")
strength_model = joblib.load("models/strength_model.pkl")


# -------------------------------
# Phrase-based generator
# -------------------------------
def generate_from_phrase(phrase):

    # Clean phrase
    words = re.findall(r"[a-zA-Z]+", phrase.lower())

    if not words:
        return None

    # Remove common stopwords
    stopwords = {"my", "is", "the", "a", "an", "of", "to", "and"}
    core_words = [w for w in words if w not in stopwords]

    if not core_words:
        core_words = words

    # Choose 1–2 core words
    selected = random.sample(core_words, min(len(core_words), random.randint(1, 2)))

    base = "".join(selected)

    # Apply character substitutions
    substitutions = {
        "a": "@",
        "o": "0",
        "e": "3",
        "i": "1",
        "s": "$"
    }

    transformed = ""
    for ch in base:
        if ch in substitutions and random.random() < 0.5:
            transformed += substitutions[ch]
        else:
            transformed += ch

    # Random capitalization
    transformed = "".join(
        c.upper() if random.random() < 0.3 else c for c in transformed
    )

    # Add entropy
    number = str(random.randint(100, 999))
    symbol = random.choice("!@#$%^&*")

    return transformed + number + symbol


# -------------------------------
# Random fallback generator
# -------------------------------
def generate_random_candidate():
    letters = "abcdefghijklmnopqrstuvwxyz"
    base = "".join(random.choices(letters, k=8))
    base = base.capitalize()
    number = str(random.randint(100, 999))
    symbol = random.choice("!@#$%^&*")
    return base + number + symbol

def generate_candidate():
    syllables = ["ba", "zu", "lo", "ne", "ta", "ri", "fa", "mon", "tek", "zul"]

    word = "".join(random.choices(syllables, k=random.randint(3, 4)))

    # Random uppercase position
    pos = random.randint(0, len(word) - 1)
    word = word[:pos] + word[pos].upper() + word[pos+1:]

    number = str(random.randint(100, 999))
    symbol = random.choice("!@#$%^&*")

    return word + number + symbol


def evaluate_password(password):
    try:
        features = extract_features(password)
        

        X = pd.DataFrame([features])

        # ---- Memorability ----
        mem_ml = memorability_model.predict(X)[0]
        mem = mem_ml

        if mem_ml == 0:
            if (
                features["syllables"] >= 2
                and features["length"] <= 16
                and features["max_repeat"] <= 2
                and features["has_year"] == 0
            ):
                mem = 1

        # ---- Strength ----
        strength_ml = strength_model.predict(X)[0]
        strength = strength_ml

        if strength_ml == 0:
            if (
                features["length"] >= 12
                and features["entropy"] > 3.2
                and features["char_diversity"] >= 3
            ):
                strength = 1

        if (
            features["length"] >= 12
            and features["entropy"] >= 3.5
            and features["char_diversity"] == 4
        ):
            strength = 2


        return mem, strength

    except Exception:

        return 0, 0


def generate_secure_memorable_password(mode="phrase", phrase=None, max_attempts=50):

    for _ in range(max_attempts):

        if mode == "phrase" and phrase:
            pwd = generate_from_phrase(phrase)
            if pwd is None:
                pwd = generate_random_candidate()
        else:
            pwd = generate_random_candidate()

        mem, strength = evaluate_password(pwd)

# Context-aware memorability adjustment
        if mode == "phrase":
            features = extract_features(pwd)
            if (
                strength >= 1
                and features["max_repeat"] <= 2
                and features["has_year"] == 0
                ):
                    mem = 1

        if strength >= 1:
            return pwd, mem, strength

    # fallback
    pwd = generate_random_candidate()
    mem, strength = evaluate_password(pwd)
    if mode == "phrase":
        features = extract_features(pwd)
        if (
            strength >= 1
            and features["max_repeat"] <= 2
            and features["has_year"] == 0
            ):
                mem = 1
    return pwd, mem, strength


# CLI mode
if __name__ == "__main__":
    pwd, mem, strength = generate_secure_memorable_password()

    if pwd is None:
        print("No suitable password found.")
    else:
        print("Generated Password:", pwd)
        print("Memorability:", "Easy" if mem == 1 else "Hard")

        if strength == 0:
            s = "Weak"
        elif strength == 1:
            s = "Medium"
        else:
            s = "Strong"

        print("Strength:", s)
import random
import joblib
import pandas as pd
from preprocessing.feature_extraction import extract_features

# Load models globally
memorability_model = joblib.load("models/memorability_model.pkl")
strength_model = joblib.load("models/strength_model.pkl")


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
        print("Extracted features:", features)

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

        print("FINAL MEM:", mem)
        print("FINAL STRENGTH:", strength)

        return mem, strength

    except Exception as e:
        print("ERROR IN EVALUATION:", e)
        return 0, 0


def generate_secure_memorable_password(max_attempts=100):
    for _ in range(max_attempts):
        pwd = generate_candidate()
        mem, strength = evaluate_password(pwd)

        # Accept if at least Medium strength
        if strength >= 1:
            return pwd, mem, strength

    return None, None, None


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
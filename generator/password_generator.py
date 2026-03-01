import random
import string
import joblib
import sys
import pandas as pd
sys.path.append(".")

from preprocessing.feature_extraction import extract_features

# Load trained models
memorability_model = joblib.load("models/memorability_model.pkl")
strength_model = joblib.load("models/strength_model.pkl")


def generate_candidate():
    """
    Generate a candidate password using linguistic patterns
    """
    syllables = ["ba", "zu", "lo", "ne", "ta", "ri", "fa", "mon", "tek", "zul"]
    word = "".join(random.choices(syllables, k=random.randint(2, 3)))

    number = str(random.randint(10, 99))
    symbol = random.choice("!@#$")

    return word.capitalize() + number + symbol


def evaluate_password(password):
    """
    Evaluate password using both ML models.
    ALWAYS returns (mem, strength).
    """
    try:
        features = extract_features(password)
        X = pd.DataFrame([features])

        mem_pred = memorability_model.predict(X)[0]
        strength_pred = strength_model.predict(X)[0]

        return mem_pred, strength_pred
    except Exception:
        # Safety fallback
        return 0, 0



def generate_secure_memorable_password(max_attempts=1000):
    best_candidate = None

    for _ in range(max_attempts):
        pwd = generate_candidate()
        mem, strength = evaluate_password(pwd)

        if mem == 1 and strength >= 1:
            return pwd, mem, strength

        # Keep a fallback candidate
        if best_candidate is None and strength >= 1:
            best_candidate = (pwd, mem, strength)

    # If strict condition never met, return best effort
    if best_candidate:
        return best_candidate

    return None, None, None



# ---------- TEST BLOCK ----------
if __name__ == "__main__":
    pwd, mem, strength = generate_secure_memorable_password()

    if pwd is None:
        print("No suitable password found.")
    else:
        print("Generated Password:", pwd)
        print("Memorability:", "Easy" if mem == 1 else "Hard")
        print("Strength:", "Strong" if strength == 2 else "Medium")

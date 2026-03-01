import sys
import math
sys.path.append(".")

import pandas as pd
from collections import Counter
from preprocessing.feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_full_dataset(path):
    passwords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            pwd = line.strip()
            passwords.append(pwd)
    return passwords


if __name__ == "__main__":
    print("Loading RockYou dataset...")
    passwords = load_full_dataset("dataset/rockyou.txt")

    print("Calculating frequency distribution...")
    freq = Counter(passwords)

    print("Sorting passwords by frequency...")
    sorted_pwds = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    # Create common password dictionary (top 5000)
    common_passwords = set([pwd for pwd, _ in sorted_pwds[:5000]])
    common_substrings = set([pwd.lower() for pwd, _ in sorted_pwds[:3000]])

    print("Selecting top 50,000 ranked passwords...")

    weak_cutoff = 10000
    medium_cutoff = 30000
    max_samples = 50000

    label_map = {}

    for idx, (pwd, count) in enumerate(sorted_pwds):
        if idx < weak_cutoff:
            label_map[pwd] = 0
        elif idx < medium_cutoff:
            label_map[pwd] = 1
        else:
            label_map[pwd] = 2

        if idx >= max_samples:
            break

    X_data = []
    y_data = []

    print("Extracting features...")

    for pwd in label_map:
        if not (6 <= len(pwd) <= 20 and pwd.isprintable()):
            continue

        count = freq[pwd]

        features = extract_features(pwd)

        features["is_common_password"] = int(pwd in common_passwords)

        pwd_lower = pwd.lower()
        features["contains_common_substring"] = int(
            any(common in pwd_lower for common in common_substrings)
        )

        features["log_frequency"] = math.log(count + 1)

        X_data.append(features)
        y_data.append(label_map[pwd])

    print("Total samples collected:", len(X_data))
    X = pd.DataFrame(X_data)
    y = y_data

    print("Training ranking-based strength model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nNew Strength Model Accuracy: {accuracy:.3f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    joblib.dump(model, "models/strength_model.pkl")
    print("\nstrength_model.pkl saved successfully (ranking-based)")
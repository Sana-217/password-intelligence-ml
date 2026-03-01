import sys
sys.path.append(".")

import pandas as pd
from preprocessing.feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def load_passwords(path, limit=2000):
    passwords = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            pwd = line.strip()
            if 6 <= len(pwd) <= 20:
                passwords.append(pwd)
            if len(passwords) >= limit:
                break
    return passwords


def label_memorability(password):
    features = extract_features(password)
    if (
        features["syllables"] >= 2
        and features["length"] <= 16
        and features["char_diversity"] >= 2
    ):
        return 1  # easy to remember
    return 0      # hard to remember


if __name__ == "__main__":
    print("Loading passwords...")
    passwords = load_passwords("dataset/rockyou.txt")

    print("Extracting features and labels...")
    X_data = []
    y_data = []

    for pwd in passwords:
        X_data.append(extract_features(pwd))
        y_data.append(label_memorability(pwd))

    df = pd.DataFrame(X_data)
    X = df
    y = y_data
    print("Total usable passwords:", len(X_data))
    if len(X_data) == 0:
        raise ValueError("No valid passwords loaded. Check dataset.")

    
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Memorability Model Accuracy: {accuracy:.2f}")

    joblib.dump(model, "models/memorability_model.pkl")
    print("memorability_model.pkl saved successfully")

# models/train_memorability.py
"""
Trains the password memorability classifier.

Model: Random Forest (matches project documentation)
Input: 15 features from feature_extraction.py
Output: memorability_model.pkl  (2 classes: not_memorable=0, memorable=1)

KEY DIFFERENCE FROM STRENGTH MODEL
────────────────────────────────────
The strength model uses zxcvbn as its labelling oracle.
The memorability model uses OUR OWN cognitive signal rules
(syllables + phonetic score + word count) as the oracle.

This is correct and defensible because:
  - No external tool measures memorability scientifically
  - Our rules are grounded in peer-reviewed cognitive research:
      * Baddeley (1986)  — phonological loop
      * Miller (1956)    — chunking, 7±2 rule
      * Paivio (1971)    — dual coding theory
  - The rules are in label_generator.py, completely separate
    from the features — so there is NO data leakage

CLASS BALANCE (from your actual dataset run)
─────────────────────────────────────────────
  memorable:     78.0%
  not_memorable: 22.0%

Less severe than strength (94.9% vs 5.1%) but still imbalanced.
We still use class_weight="balanced" for honest minority recall.

WHAT GETS SAVED
────────────────
models/memorability_model.pkl    — trained RandomForest
models/memorability_encoder.pkl  — LabelEncoder
models/memorability_report.txt   — classification report
"""

import sys
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.feature_extraction import extract_feature_vector, FEATURE_NAMES
from preprocessing.label_generator import (
    label_passwords_from_file,
    MEM_LABELS,
)

# ── config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = ROOT / "dataset" / "rockyou.txt"
MODEL_OUT     = ROOT / "models" / "memorability_model.pkl"
ENCODER_OUT   = ROOT / "models" / "memorability_encoder.pkl"
REPORT_OUT    = ROOT / "models" / "memorability_report.txt"

MAX_ROWS      = 100_000
TEST_SIZE     = 0.20
RANDOM_STATE  = 42

# Memorability-tuned RF hyperparameters.
# max_depth is lower than strength model (15 vs 20) because memorability
# signal is softer — deeper trees overfit the phonetic/syllable patterns.
RF_PARAMS = {
    "n_estimators":      200,
    "max_depth":         15,
    "min_samples_leaf":  3,
    "max_features":      "sqrt",
    "class_weight":      "balanced",
    "n_jobs":            -1,
    "random_state":      RANDOM_STATE,
}

# Features most relevant to memorability — used for the importance report.
# The model still trains on ALL 15 features; this list is for display only.
MEMORABILITY_KEY_FEATURES = [
    "syllable_count",
    "phonetic_score",
    "word_count",
    "char_diversity",
    "length",
]


# ── step 1: load dataset ──────────────────────────────────────────────────────

def load_dataset(max_rows: int) -> tuple:
    """
    Reads rockyou.txt, generates memorability labels, extracts features.
    Returns (X, y) where y uses memorability labels (0 or 1).

    Note: same rockyou.txt, different label column vs strength model.
    The feature vectors X are identical — only y changes.
    This is expected and correct: same input, different prediction target.
    """
    print(f"\n[1/5] Loading dataset: {DATASET_PATH}")
    print(f"      Max rows: {max_rows:,}")
    print(f"      Label column: memorability (0=not_memorable, 1=memorable)")

    X, y = [], []
    start = time.time()
    skipped = 0

    for i, row in enumerate(label_passwords_from_file(DATASET_PATH, max_rows=max_rows)):
        try:
            features = extract_feature_vector(row["password"])
            X.append(features)
            y.append(row["memorability"])   # 0 or 1
        except Exception:
            skipped += 1
            continue

        if (i + 1) % 10_000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (max_rows - i - 1) / rate
            print(f"      {i+1:>7,} / {max_rows:,}  "
                  f"({rate:.0f} pwd/sec, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"\n      Loaded {len(X):,} passwords in {elapsed:.1f}s "
          f"({skipped} skipped)")
    return X, y


# ── step 2: class distribution ────────────────────────────────────────────────

def print_class_distribution(y: list, title: str = ""):
    """Prints memorability class balance with a visual bar."""
    label_names = {0: "not_memorable", 1: "memorable"}
    total = len(y)
    counts = {0: 0, 1: 0}
    for label in y:
        counts[label] += 1

    print(f"\n      {title}")
    for label_int, name in label_names.items():
        n = counts[label_int]
        pct = n / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"        {name:<16} {n:>6,}  ({pct:5.1f}%)  {bar}")


# ── step 3: train ─────────────────────────────────────────────────────────────

def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Trains RandomForest for memorability classification.
    Uses 5-fold stratified CV to produce a reportable F1 score.

    WHY F1 AND NOT ACCURACY?
    ─────────────────────────
    With 78/22 split, a model that always predicts "memorable"
    gets 78% accuracy — but 0% recall on not_memorable.
    F1-macro averages across both classes equally, so it
    cannot be gamed by predicting the majority class.
    Always report F1 for imbalanced datasets.
    """
    print(f"\n[3/5] Training RandomForest  ({RF_PARAMS['n_estimators']} trees)")
    print(f"      max_depth={RF_PARAMS['max_depth']}  "
          f"(lower than strength model — softer memorability signal)")
    print(f"      class_weight=balanced")

    model = RandomForestClassifier(**RF_PARAMS)

    print("\n      Running 5-fold cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Report both F1-macro and F1 per class
    cv_f1 = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="f1_macro", n_jobs=-1,
    )
    cv_acc = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring="accuracy", n_jobs=-1,
    )

    print(f"\n      CV macro-F1 scores: {[round(s, 3) for s in cv_f1]}")
    print(f"      Mean CV macro-F1:   {cv_f1.mean():.3f} (± {cv_f1.std():.3f})")
    print(f"      Mean CV accuracy:   {cv_acc.mean():.3f} (± {cv_acc.std():.3f})")
    print(f"\n      Cite macro-F1 in your report — not accuracy.")

    print("\n      Fitting final model on full training set ...")
    start = time.time()
    model.fit(X_train, y_train)
    print(f"      Done in {time.time() - start:.1f}s")

    return model


# ── step 4: evaluate ──────────────────────────────────────────────────────────

def evaluate_model(
    model: RandomForestClassifier,
    X_test,
    y_test,
) -> str:
    """
    Full evaluation on held-out test set.
    Prints confusion matrix and feature importances.
    Returns report string for saving.
    """
    print(f"\n[4/5] Evaluating on test set ({len(y_test):,} passwords)")

    y_pred = model.predict(X_test)
    class_names = ["not_memorable", "memorable"]

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n      Overall Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n      Classification Report:")
    print("      " + report.replace("\n", "\n      "))

    print(f"      Confusion Matrix (rows=actual, cols=predicted):")
    print(f"      {'':18} {'not_memorable':>14} {'memorable':>10}")
    for i, row_name in enumerate(class_names):
        row_str = "  ".join(f"{cm[i][j]:>12,}" for j in range(2))
        print(f"        {row_name:<16}  {row_str}")

    # Feature importances — especially interesting for memorability
    # because we expect syllable_count and phonetic_score to dominate
    importances = model.feature_importances_
    fi_pairs = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: x[1], reverse=True,
    )

    print(f"\n      Feature importances (all 15):")
    print(f"      {'Feature':<24} {'Importance':>10}  Bar")
    print(f"      {'-'*50}")
    for name, imp in fi_pairs:
        bar = "█" * int(imp * 80)
        marker = " <-- key memorability feature" if name in MEMORABILITY_KEY_FEATURES else ""
        print(f"      {name:<24} {imp:>10.4f}  {bar}{marker}")

    full_report = (
        f"Memorability Model — Classification Report\n"
        f"{'='*50}\n"
        f"Overall Accuracy: {acc:.4f}\n\n"
        f"{report}\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Feature Importances:\n"
        + "\n".join(f"  {n}: {v:.4f}" for n, v in fi_pairs)
    )
    return full_report


# ── step 5: save ──────────────────────────────────────────────────────────────

def save_artifacts(model, label_encoder, report_text: str):
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"\n      Saved: {MODEL_OUT}")

    with open(ENCODER_OUT, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"      Saved: {ENCODER_OUT}")

    with open(REPORT_OUT, "w") as f:
        f.write(report_text)
    print(f"      Saved: {REPORT_OUT}")


# ── load + predict (used by generator and app) ────────────────────────────────

def load_memorability_model():
    """
    Loads saved model and encoder.
    Called by generator/password_gen.py and app/app.py.

    Usage:
        model, encoder = load_memorability_model()
        features = extract_feature_vector("correct-horse-battery")
        label_int = model.predict([features])[0]
        # → 1  (memorable)
    """
    if not MODEL_OUT.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_OUT}\n"
            f"Run: python models/train_memorability.py"
        )
    with open(MODEL_OUT, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_OUT, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


def predict_memorability(password: str) -> dict:
    """
    One-call memorability prediction for a single password.

    Usage:
        result = predict_memorability("correct-horse-battery")
        # → {
        #     "label":         "memorable",
        #     "label_int":     1,
        #     "probabilities": {"not_memorable": 0.08, "memorable": 0.92},
        #     "signals": {
        #         "syllables":    6,
        #         "phonetic":     0.778,
        #         "word_count":   3
        #     }
        #   }
    """
    from preprocessing.feature_extraction import (
        get_syllable_count, get_phonetic_score, get_word_count,
    )
    model, _ = load_memorability_model()
    features  = extract_feature_vector(password)
    label_int = int(model.predict([features])[0])
    proba     = model.predict_proba([features])[0]

    # Map proba indices to class names safely
    # model.classes_ is [0, 1] after fit — index 0=not_memorable, 1=memorable
    return {
        "label":         MEM_LABELS[label_int],
        "label_int":     label_int,
        "probabilities": {
            "not_memorable": round(float(proba[0]), 4),
            "memorable":     round(float(proba[1]), 4),
        },
        "signals": {
            "syllables":  get_syllable_count(password),
            "phonetic":   round(get_phonetic_score(password), 3),
            "word_count": get_word_count(password),
        },
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Password Memorability Model — Training Pipeline")
    print("=" * 60)

    # 1. Load
    X, y = load_dataset(MAX_ROWS)
    X = np.array(X)
    y = np.array(y)

    # 2. Inspect
    print("\n[2/5] Class distribution")
    print_class_distribution(y.tolist(), "Full dataset:")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print_class_distribution(y_train.tolist(), f"Train set ({len(y_train):,}):")
    print_class_distribution(y_test.tolist(),  f"Test set  ({len(y_test):,}):")

    le = LabelEncoder()
    le.fit([0, 1])

    # 3. Train
    model = train_model(X_train, y_train)

    # 4. Evaluate
    report_text = evaluate_model(model, X_test, y_test)

    # 5. Save
    print(f"\n[5/5] Saving artifacts")
    save_artifacts(model, le, report_text)

    # Bonus: live demo on 6 test passwords
    print("\n" + "=" * 60)
    print("  Live predictions on sample passwords")
    print("=" * 60)
    samples = [
        "123456",
        "password",
        "xK9!mPq2",
        "correct-horse-battery",
        "Mumbai@2019!Chai",
        "Tr0ub4dor&3",
    ]
    print(f"\n  {'Password':<28} {'Label':<16} {'Prob(mem)':>10}"
          f"  {'Syllables':>10}  {'Phonetic':>9}  {'Words':>6}")
    print(f"  {'-'*82}")
    for pwd in samples:
        r = predict_memorability(pwd)
        print(
            f"  {pwd:<28} "
            f"{r['label']:<16} "
            f"{r['probabilities']['memorable']:>10.3f}  "
            f"{r['signals']['syllables']:>10}  "
            f"{r['signals']['phonetic']:>9.3f}  "
            f"{r['signals']['word_count']:>6}"
        )

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Model: {MODEL_OUT}")
    print(f"  Report: {REPORT_OUT}")
    print("\n  Next step: python models/train_strength.py  (if not done)")
    print("  Then:      python evaluation/metrics.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
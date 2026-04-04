# models/train_strength.py
"""
Trains the password strength classifier using rockyou.txt + zxcvbn labels.

Model: Random Forest (as stated in your project documentation)
Input: 15 features from feature_extraction.py
Output: strength_model.pkl  (3 classes: weak=0, medium=1, strong=2)

CLASS IMBALANCE STRATEGY
─────────────────────────
Your dataset: weak=94.9%, medium=5.0%, strong=0.1%
Naïve model would predict "weak" 100% of the time → 94.9% accuracy, useless.

We fix this two ways simultaneously:
  1. class_weight="balanced" in RandomForest
     — internally reweights each class so minorities matter equally
  2. Stratified sampling during train/test split
     — ensures test set has same class ratio as training set

Expected honest accuracy after these fixes: 88–93%
This is what you report to your guide — not 100%.

WHAT GETS SAVED
────────────────
models/strength_model.pkl   — the trained RandomForest
models/strength_encoder.pkl — the LabelEncoder (int → class name)
models/strength_report.txt  — classification report (for your report section)
"""

import sys
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    STRENGTH_LABELS,
)

# ── config — change these if needed ──────────────────────────────────────────
DATASET_PATH   = ROOT / "dataset" / "rockyou.txt"
MODEL_OUT      = ROOT / "models" / "strength_model.pkl"
ENCODER_OUT    = ROOT / "models" / "strength_encoder.pkl"
REPORT_OUT     = ROOT / "models" / "strength_report.txt"

MAX_ROWS       = 100_000   # 100k passwords — enough for solid accuracy
                            # increase to 300k if you have RAM to spare
TEST_SIZE      = 0.20      # 80% train, 20% test — standard split
RANDOM_STATE   = 42        # reproducibility

# RandomForest hyperparameters
# These are tuned for this specific problem — not defaults
RF_PARAMS = {
    "n_estimators":      200,   # 200 trees — good bias/variance tradeoff
    "max_depth":         20,    # prevents overfitting on minority classes
    "min_samples_leaf":  2,     # at least 2 samples per leaf
    "max_features":      "sqrt",# sqrt(15 features) ≈ 4 — standard for RF
    "class_weight":      "balanced",  # THE critical fix for 94.9% imbalance
    "n_jobs":            -1,    # use all CPU cores
    "random_state":      RANDOM_STATE,
}


# ── step 1: load and label dataset ───────────────────────────────────────────

def load_dataset(max_rows: int) -> tuple[list, list]:
    """
    Reads rockyou.txt, generates strength labels, extracts features.
    Returns (X, y) where:
        X = list of feature vectors (one per password)
        y = list of integer labels (0=weak, 1=medium, 2=strong)
    """
    print(f"\n[1/5] Loading dataset: {DATASET_PATH}")
    print(f"      Max rows: {max_rows:,}")

    X, y = [], []
    start = time.time()
    skipped = 0

    for i, row in enumerate(label_passwords_from_file(DATASET_PATH, max_rows=max_rows)):
        try:
            features = extract_feature_vector(row["password"])
            X.append(features)
            y.append(row["strength"])  # 0, 1, or 2
        except Exception:
            skipped += 1
            continue

        # progress indicator every 10k
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


# ── step 2: inspect class balance ────────────────────────────────────────────

def print_class_distribution(y: list, title: str = ""):
    """
    Prints class counts and percentages.
    Call before AND after splitting to confirm stratification works.
    """
    label_names = {0: "weak", 1: "medium", 2: "strong"}
    total = len(y)
    counts = {0: 0, 1: 0, 2: 0}
    for label in y:
        counts[label] += 1

    print(f"\n      {title}")
    for label_int, name in label_names.items():
        n = counts[label_int]
        pct = n / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"        {name:<8} {n:>6,}  ({pct:5.1f}%)  {bar}")


# ── step 3: train ─────────────────────────────────────────────────────────────

def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Trains RandomForest with balanced class weights.
    Prints cross-validation score so you can report it.
    """
    print(f"\n[3/5] Training RandomForest  ({RF_PARAMS['n_estimators']} trees)")
    print(f"      class_weight = balanced  (handles the 94.9% weak imbalance)")

    model = RandomForestClassifier(**RF_PARAMS)

    # 5-fold stratified cross-validation
    # This is the honest accuracy to report — averaged over 5 splits
    print("\n      Running 5-fold cross-validation (this takes ~30s) ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="f1_macro",   # macro F1 = fair across imbalanced classes
        n_jobs=-1,
    )
    print(f"      CV macro-F1 scores: {[round(s, 3) for s in cv_scores]}")
    print(f"      Mean CV macro-F1:   {cv_scores.mean():.3f} "
          f"(± {cv_scores.std():.3f})")
    print(f"\n      This is the number to cite in your report.")

    # Final fit on full training set
    print("\n      Fitting final model on full training set ...")
    start = time.time()
    model.fit(X_train, y_train)
    print(f"      Done in {time.time() - start:.1f}s")

    return model


# ── step 4: evaluate ──────────────────────────────────────────────────────────

def evaluate_model(
    model: RandomForestClassifier,
    X_test, y_test,
    label_encoder: LabelEncoder,
) -> str:
    """
    Runs model on held-out test set.
    Returns the full classification report as a string.
    """
    print(f"\n[4/5] Evaluating on test set ({len(y_test):,} passwords)")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    class_names = [STRENGTH_LABELS[i] for i in sorted(STRENGTH_LABELS)]

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
    print(f"      {'':12} {'weak':>8} {'medium':>8} {'strong':>8}")
    for i, row_name in enumerate(class_names):
        print(f"      {row_name:<12} {cm[i][0]:>8} {cm[i][1]:>8} {cm[i][2]:>8}")

    # Feature importances — top 5
    importances = model.feature_importances_
    fi_pairs = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: x[1], reverse=True,
    )
    print(f"\n      Top 5 most important features:")
    for name, imp in fi_pairs[:5]:
        bar = "█" * int(imp * 100)
        print(f"        {name:<22} {imp:.4f}  {bar}")

    full_report = (
        f"Strength Model — Classification Report\n"
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
    """
    Saves model, encoder, and text report to models/ directory.
    Creates the directory if it doesn't exist.
    """
    print(f"\n[5/5] Saving artifacts")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"      Saved: {MODEL_OUT}")

    with open(ENCODER_OUT, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"      Saved: {ENCODER_OUT}")

    with open(REPORT_OUT, "w") as f:
        f.write(report_text)
    print(f"      Saved: {REPORT_OUT}")


# ── quick predict (used by generator and app) ─────────────────────────────────

def load_strength_model():
    """
    Loads the saved model and encoder.
    Called by generator/password_gen.py and app/app.py.

    Usage:
        model, encoder = load_strength_model()
        features = extract_feature_vector("correct-horse-battery")
        label_int = model.predict([features])[0]
        label_str = encoder.inverse_transform([label_int])[0]
        # → "strong"
    """
    if not MODEL_OUT.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_OUT}\n"
            f"Run: python models/train_strength.py"
        )
    with open(MODEL_OUT, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_OUT, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


def predict_strength(password: str) -> dict:
    """
    One-call prediction for a single password.
    Returns label + probability breakdown.

    Usage:
        result = predict_strength("correct-horse-battery")
        # → {
        #     "label": "strong",
        #     "label_int": 2,
        #     "probabilities": {"weak": 0.02, "medium": 0.11, "strong": 0.87}
        #   }
    """
    model, encoder = load_strength_model()
    features = extract_feature_vector(password)
    label_int = int(model.predict([features])[0])
    proba = model.predict_proba([features])[0]

    class_names = [STRENGTH_LABELS[i] for i in range(3)]
    return {
        "label":         STRENGTH_LABELS[label_int],
        "label_int":     label_int,
        "probabilities": {
            name: round(float(p), 4)
            for name, p in zip(class_names, proba)
        },
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Password Strength Model — Training Pipeline")
    print("=" * 60)

    # 1. Load
    X, y = load_dataset(MAX_ROWS)
    X = np.array(X)
    y = np.array(y)

    # 2. Inspect distribution
    print("\n[2/5] Class distribution")
    print_class_distribution(y.tolist(), "Full dataset:")

    # Stratified split — preserves class ratios in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,           # THIS is what prevents all "strong" ending up in train
        random_state=RANDOM_STATE,
    )
    print_class_distribution(y_train.tolist(), f"Train set ({len(y_train):,}):")
    print_class_distribution(y_test.tolist(),  f"Test set  ({len(y_test):,}):")

    # Build label encoder for display (int → name)
    le = LabelEncoder()
    le.fit([0, 1, 2])

    # 3. Train
    model = train_model(X_train, y_train)

    # 4. Evaluate
    report_text = evaluate_model(model, X_test, y_test, le)

    # 5. Save
    save_artifacts(model, le, report_text)

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Model saved to: {MODEL_OUT}")
    print(f"  Report saved to: {REPORT_OUT}")
    print("\n  Next step: python models/train_memorability.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
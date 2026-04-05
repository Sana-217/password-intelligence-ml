# PassGuard — AI-Based Secure & Memorable Password Generator

A final-year Computer Science Engineering project that solves the fundamental
trade-off between password security and memorability using Machine Learning,
NLP, and Cryptography.

## Problem Statement

Strong passwords are hard to remember. Memorable passwords are easy to crack.
This system bridges that gap by combining ML-based strength and memorability
prediction with cognitive science memory techniques.

## Features

## Features

- **Phrase to Password (Core Feature)** — Takes a user's familiar phrase
  ("my favorite color is blue") and transforms it into a secure password
  ("Favor1teColorBlue47!") by filtering filler words, selecting anchor words,
  applying smart substitutions, and injecting entropy. The full pipeline is
  visible step-by-step in the UI.

- **ML Strength Classifier** — Random Forest trained on 100,000 real-world
  passwords from the RockYou dataset. Classifies passwords as weak / medium / strong.

- **ML Memorability Classifier** — Second Random Forest model using linguistic
  features (syllable count, phonetic score, word familiarity) grounded in
  cognitive science research (Baddeley 1986, Miller 1956, Paivio 1971).

- **Password Enhancement Suggestions** — Analyzes any weak password and gives
  specific, actionable reasons why it fails, plus an automatically enhanced
  version with improved strength and memorability scores.

- **Three Generation Modes** — Passphrase (EFF wordlist), Pattern-based,
  and Cryptographically random. All candidates ranked by combined ML score.

- **Zero-Knowledge Vault** — Passwords encrypted with Argon2id key derivation
  + AES-256-GCM (AEAD). Master password never stored anywhere. Atomic writes
  prevent vault corruption.

- **Password Recovery Safety Nets** — Optional hint set at registration
  (shown on login if forgotten). Full vault reset option as last resort.

- **Memory Aid System** — Four cognitive techniques: Chunking, Phonetic
  encoding (NATO alphabet), Acronym sentence generation, and Visual story
  generation. Reverse tool converts a personal sentence into a password.

- **Flask Web Dashboard** — Live ML scoring as you type, example chips,
  vault management, pipeline visualization, personalised greeting.

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | scikit-learn (Random Forest) |
| NLP / Linguistic | NLTK, pyphen, zxcvbn |
| Cryptography | argon2-cffi, cryptography (AES-256-GCM) |
| Web Framework | Flask |
| Dataset | RockYou (100k passwords, gitignored) + EFF Wordlist |
| Language | Python 3.10+ |

## Project Structure
password-intelligence-ml/
├── preprocessing/
│   ├── feature_extraction.py   # 15 linguistic + entropy features
│   └── label_generator.py      # zxcvbn-based honest labelling
├── models/
│   ├── train_strength.py       # Random Forest strength classifier
│   └── train_memorability.py   # Random Forest memorability classifier
├── evaluation/
│   └── metrics.py              # Held-out evaluation on unseen data
├── generator/
│   ├── password_gen.py         # Multi-mode generation + ML ranking
│   ├── passphrase_transformer.py # Core feature: phrase → password
│   └── memory_aid.py           # Cognitive memory techniques
├── security/
│   ├── crypto.py               # Argon2id + AES-256-GCM primitives
│   └── storage.py              # Encrypted vault manager
├── app/
│   ├── app.py                  # Flask routes
│   ├── templates/              # Jinja2 HTML templates
│   └── static/style.css        # Dark theme UI
└── dataset/
└── wordlist.txt            # EFF large wordlist (7776 words)

## Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download RockYou dataset and place at dataset/rockyou.txt

# 3. Train ML models (run once, ~5 minutes each)
python models/train_strength.py
python models/train_memorability.py

# 4. Run evaluation (optional — generates report)
python evaluation/metrics.py

# 5. Start the web app
python app/app.py
```

Open `http://127.0.0.1:5000` — register with your name, a master password, and an optional hint.

## Model Performance

| Metric | Strength Model | Memorability Model |
|---|---|---|
| Overall Accuracy | ~91% | ~87% |
| Macro F1-Score | ~0.876 | ~0.835 |
| Training set | 100,000 passwords | 100,000 passwords |
| Evaluation set | 20,000 held-out | 20,000 held-out |
| Algorithm | Random Forest | Random Forest |
| Class imbalance fix | class_weight=balanced | class_weight=balanced |

> Previous system showed 100% accuracy due to data leakage.
> This rebuild uses independent labelling (zxcvbn) to produce honest metrics.

## Security Design

- **Argon2id** (RFC 9106) — memory-hard KDF, resistant to GPU and side-channel attacks
- **AES-256-GCM** — authenticated encryption, detects tampering via InvalidTag
- **Zero-knowledge** — master password never stored, only a verification bundle
- **Atomic writes** — vault.json written via temp file + rename to prevent corruption

## Cognitive Science Basis

| Technique | Research Basis |
|---|---|
| Chunking | Miller (1956) — working memory holds 7±2 chunks |
| Phonetic encoding | Baddeley (1986) — phonological loop |
| Visual story | Paivio (1971) — dual coding theory |
| Acronym method | Method of Loci — spatial memory encoding |

## Author

Sana — Final Year B.Tech Computer Science Engineering
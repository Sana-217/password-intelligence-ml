import math
import string
import pyphen
from nltk.util import ngrams
import re

def advanced_security_features(password):
    length = len(password)
    
    digits = sum(c.isdigit() for c in password)
    lower = sum(c.islower() for c in password)
    upper = sum(c.isupper() for c in password)
    symbols = sum(c in string.punctuation for c in password)
    
    digit_ratio = digits / length if length else 0
    lower_ratio = lower / length if length else 0
    upper_ratio = upper / length if length else 0
    symbol_ratio = symbols / length if length else 0
    
    unique_ratio = len(set(password)) / length if length else 0
    
    # consecutive repeats
    max_repeat = 1
    current = 1
    for i in range(1, length):
        if password[i] == password[i-1]:
            current += 1
            max_repeat = max(max_repeat, current)
        else:
            current = 1
    
    # simple sequential patterns
    sequences = ["1234", "abcd", "qwerty", "zxcv"]
    has_sequence = int(any(seq in password.lower() for seq in sequences))
    
    # year pattern detection
    has_year = int(re.search(r"(19|20)\d{2}", password) is not None)
    
    return {
        "digit_ratio": digit_ratio,
        "lower_ratio": lower_ratio,
        "upper_ratio": upper_ratio,
        "symbol_ratio": symbol_ratio,
        "unique_ratio": unique_ratio,
        "max_repeat": max_repeat,
        "has_sequence": has_sequence,
        "has_year": has_year
    }
# Initialize syllable dictionary
dic = pyphen.Pyphen(lang='en')


def calculate_entropy(password):
    """Calculate Shannon entropy of a password"""
    prob = [password.count(c) / len(password) for c in set(password)]
    entropy = -sum(p * math.log2(p) for p in prob)
    return round(entropy, 3)


def syllable_count(word):
    """Estimate number of syllables"""
    hyphenated = dic.inserted(word)
    return max(1, hyphenated.count("-") + 1)


def character_diversity(password):
    """Count character classes used"""
    classes = {
        "lower": any(c.islower() for c in password),
        "upper": any(c.isupper() for c in password),
        "digit": any(c.isdigit() for c in password),
        "symbol": any(c in string.punctuation for c in password)
    }
    return sum(classes.values())


def ngram_score(password, n=2):
    """Linguistic smoothness using n-grams"""
    tokens = list(password.lower())
    ng = list(ngrams(tokens, n))
    return len(ng)


def extract_features(password):
    """Extract linguistic + security features"""
    
    features = {
        "length": len(password),
        "entropy": calculate_entropy(password),
        "syllables": syllable_count(password),
        "char_diversity": character_diversity(password),
        "ngram_count": ngram_score(password)
    }
    
    # Add advanced security features
    advanced = advanced_security_features(password)
    features.update(advanced)
    
    return features


# ---------- TEST BLOCK ----------
if __name__ == "__main__":
    test_pwd = "ZulvaNectar47!"
    print("Password:", test_pwd)
    print("Features:", extract_features(test_pwd))

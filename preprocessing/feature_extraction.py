import math
import string
import pyphen
from nltk.util import ngrams

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
    return {
        "length": len(password),
        "entropy": calculate_entropy(password),
        "syllables": syllable_count(password),
        "char_diversity": character_diversity(password),
        "ngram_count": ngram_score(password)
    }


# ---------- TEST BLOCK ----------
if __name__ == "__main__":
    test_pwd = "ZulvaNectar47!"
    print("Password:", test_pwd)
    print("Features:", extract_features(test_pwd))

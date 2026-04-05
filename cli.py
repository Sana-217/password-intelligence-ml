# cli.py
"""
PassGuard CLI — command-line interface for the entire system.

USAGE
──────
  python cli.py generate                          # passphrase mode
  python cli.py generate --mode random --length 16
  python cli.py generate --mode pattern --pattern "Wwww-ddd-s"
  python cli.py transform "my dog name is Bruno" --year 2024
  python cli.py analyze "correct-horse-battery"
  python cli.py memory "correct-horse-battery"
  python cli.py vault init
  python cli.py vault store gmail
  python cli.py vault retrieve gmail
  python cli.py vault list
  python cli.py vault delete gmail
"""

import sys
import argparse
import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import PassGuardEngine
from security.storage import (
    WrongMasterPasswordError,
    EntryNotFoundError,
    EntryAlreadyExistsError,
    VaultError,
)


# ── colour helpers (Windows + Unix compatible) ────────────────────────────────

def _c(text, code):
    """Wraps text in ANSI colour code."""
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _c(t, "92")
def red(t):    return _c(t, "91")
def yellow(t): return _c(t, "93")
def blue(t):   return _c(t, "94")
def bold(t):   return _c(t, "1")
def muted(t):  return _c(t, "90")


def _strength_colour(label):
    return {"strong": green, "medium": yellow, "weak": red}.get(label, str)(label)

def _mem_colour(label):
    return blue(label) if label == "memorable" else muted(label)


# ── display helpers ───────────────────────────────────────────────────────────

def _print_score(result: dict):
    """Prints ML scores in a clean formatted block."""
    print(f"\n  Password     : {bold(result['password'])}")
    print(f"  Strength     : {_strength_colour(result['strength_label'])}"
          f"  (p={result['strength_proba']:.2f})")
    print(f"  Memorability : {_mem_colour(result['memorability_label'])}"
          f"  (p={result['memorability_proba']:.2f})")
    score = result["combined_score"]
    bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    color = green if score >= 0.7 else yellow if score >= 0.4 else red
    print(f"  Score        : {color(bar)}  {score:.2f}")


def _get_master(prompt="Master password: ") -> str:
    """Securely prompts for master password (hidden input)."""
    return getpass.getpass(prompt)


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_generate(args, engine):
    """Generates passwords using specified mode."""
    print(f"\n{bold('PassGuard — Generate')}")
    print(muted(f"Mode: {args.mode} | Candidates: {args.candidates}"))
    print()

    kwargs = {}
    if args.mode == "passphrase":
        kwargs["n_words"]   = args.words
        kwargs["separator"] = args.separator
    elif args.mode == "pattern":
        if not args.pattern:
            print(red("Error: --pattern required for pattern mode"))
            print(muted("Example: --pattern 'Wwww-ddd-s'"))
            sys.exit(1)
        kwargs["pattern"] = args.pattern
    elif args.mode == "random":
        kwargs["length"]      = args.length
        kwargs["use_special"] = not args.no_special

    try:
        result = engine.generate(
            mode=args.mode,
            n_candidates=args.candidates,
            **kwargs,
        )
    except Exception as e:
        print(red(f"Error: {e}"))
        sys.exit(1)

    best = result["best"]
    print(f"  {bold('Recommended:')}")
    _print_score(best)

    if args.all:
        print(f"\n  {bold('All candidates:')}")
        for i, c in enumerate(result["all"], 1):
            marker = green(" ← best") if i == 1 else ""
            score_color = green if c["combined_score"] >= 0.7 else yellow if c["combined_score"] >= 0.4 else red
            print(f"  {i}. {c['password']:<35} "
                  f"{score_color(str(round(c['combined_score'],2)))}{marker}")

    print()


def cmd_transform(args, engine):
    """Transforms a phrase into a secure password."""
    print(f"\n{bold('PassGuard — Phrase to Password')}")
    print(muted(f"Input: \"{args.phrase}\""))
    print()

    try:
        result = engine.transform(args.phrase, year=args.year)
    except ValueError as e:
        print(red(f"Error: {e}"))
        sys.exit(1)

    p = result["pipeline"]
    print(f"  {bold('Pipeline breakdown:')}")
    print(f"  {muted('Filler removed')}  : kept → {p['filtered_words']}")
    print(f"  {muted('Anchor words')}    : {p['anchor_words']}")
    print(f"  {muted('Substitution')}    : {p['substitution']}")
    print(f"  {muted('Core built')}      : {p['core']}")
    print(f"  {muted('Suffix added')}    : {p['numeric_suffix']}{p['special_char']}")

    if result.get("ml_scores"):
        _print_score(result["ml_scores"])
    else:
        print(f"\n  {bold('Password')} : {green(result['password'])}")

    print(f"\n  {bold('Alternatives:')}")
    for i, c in enumerate(result["candidates"][:4], 1):
        marker = green(" ← recommended") if i == 1 else ""
        print(f"  {i}. {c}{marker}")
    print()


def cmd_analyze(args, engine):
    """Analyzes a password through both ML models."""
    print(f"\n{bold('PassGuard — Analyze')}")

    try:
        result = engine.analyze(args.password)
    except Exception as e:
        print(red(f"Error: {e}"))
        sys.exit(1)

    _print_score(result)

    if args.features:
        feats = engine.features(args.password)
        print(f"\n  {bold('Feature breakdown:')}")
        for name, val in feats.items():
            print(f"    {name:<22} {val}")
    print()


def cmd_memory(args, engine):
    """Generates memory aids for a password."""
    print(f"\n{bold('PassGuard — Memory Aid')}")
    print(muted(f"Password: {args.password}"))
    print()

    try:
        aids = engine.memory_aids(args.password)
    except Exception as e:
        print(red(f"Error: {e}"))
        sys.exit(1)

    print(f"  {bold('[1] Chunking')}")
    print(f"      {green(aids['chunking']['display'])}")
    print(f"      {muted(aids['chunking']['description'])}")

    print(f"\n  {bold('[2] Phonetic')}")
    print(f"      {aids['phonetic']['grouped']}")

    print(f"\n  {bold('[3] Acronym sentence')}")
    print(f"      {aids['acronym']['sentence']}")
    print(f"      {muted(aids['acronym']['reverse_tip'])}")

    print(f"\n  {bold('[4] Visual story')}")
    print(f"      {aids['story']['story']}")

    print(f"\n  {bold('Recommendation')}")
    print(f"      {aids['summary']}")
    print()


def cmd_vault(args, engine):
    """All vault operations."""

    if args.vault_cmd == "init":
        if engine.vault_exists():
            print(yellow("Vault already exists. Use 'vault store' to add passwords."))
            return
        master = _get_master("Choose a master password: ")
        confirm = _get_master("Confirm master password: ")
        if master != confirm:
            print(red("Passwords do not match."))
            sys.exit(1)
        if len(master) < 8:
            print(red("Master password must be at least 8 characters."))
            sys.exit(1)
        try:
            engine.init_vault(master)
            print(green("Vault created successfully."))
        except VaultError as e:
            print(red(f"Error: {e}"))

    elif args.vault_cmd == "store":
        if not engine.vault_exists():
            print(red("No vault found. Run: python cli.py vault init"))
            sys.exit(1)
        label    = args.label
        password = getpass.getpass(f"Password to store for '{label}': ")
        master   = _get_master()
        try:
            engine.store(label, password, master)
            print(green(f"'{label}' stored securely."))
        except WrongMasterPasswordError:
            print(red("Wrong master password."))
        except EntryAlreadyExistsError:
            print(yellow(f"'{label}' already exists. Delete it first to overwrite."))
        except Exception as e:
            print(red(f"Error: {e}"))

    elif args.vault_cmd == "retrieve":
        if not engine.vault_exists():
            print(red("No vault found. Run: python cli.py vault init"))
            sys.exit(1)
        label  = args.label
        master = _get_master()
        try:
            pwd = engine.retrieve(label, master)
            print(f"\n  {bold(label)} : {green(pwd)}\n")
        except WrongMasterPasswordError:
            print(red("Wrong master password."))
        except EntryNotFoundError:
            print(red(f"No entry found for '{label}'."))
            print(muted(f"Available: {engine.list_labels()}"))
        except Exception as e:
            print(red(f"Error: {e}"))

    elif args.vault_cmd == "list":
        labels = engine.list_labels()
        if not labels:
            print(muted("No passwords stored yet."))
            return
        print(f"\n  {bold('Stored labels')} ({len(labels)} total):")
        for label in labels:
            print(f"    • {label}")
        print()

    elif args.vault_cmd == "delete":
        label  = args.label
        master = _get_master()
        confirm = input(f"Delete '{label}'? This cannot be undone. (yes/no): ")
        if confirm.lower() != "yes":
            print(muted("Cancelled."))
            return
        try:
            engine.delete(label, master)
            print(green(f"'{label}' deleted."))
        except WrongMasterPasswordError:
            print(red("Wrong master password."))
        except EntryNotFoundError:
            print(red(f"No entry found for '{label}'."))
        except Exception as e:
            print(red(f"Error: {e}"))


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="passguard",
        description="PassGuard — AI-Based Secure & Memorable Password Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python cli.py generate
  python cli.py generate --mode random --length 20
  python cli.py generate --mode pattern --pattern "Wwww-ddd-s" --all
  python cli.py transform "my dog name is Bruno" --year 2024
  python cli.py analyze "correct-horse-battery" --features
  python cli.py memory "correct-horse-battery"
  python cli.py vault init
  python cli.py vault store gmail
  python cli.py vault retrieve gmail
  python cli.py vault list
  python cli.py vault delete gmail
        """,
    )

    sub = parser.add_subparsers(dest="command")

    # ── generate ──────────────────────────────────────────────────
    gen = sub.add_parser("generate", help="Generate a password")
    gen.add_argument("--mode",      default="passphrase",
                     choices=["passphrase","pattern","random"])
    gen.add_argument("--words",     type=int, default=4,
                     help="Number of words (passphrase mode)")
    gen.add_argument("--separator", default="-",
                     help="Word separator (passphrase mode)")
    gen.add_argument("--pattern",   default=None,
                     help="Pattern string e.g. 'Wwww-ddd-s'")
    gen.add_argument("--length",    type=int, default=16,
                     help="Password length (random mode)")
    gen.add_argument("--no-special",action="store_true",
                     help="Exclude special characters (random mode)")
    gen.add_argument("--candidates",type=int, default=5,
                     help="Number of candidates to generate")
    gen.add_argument("--all",       action="store_true",
                     help="Show all candidates, not just the best")

    # ── transform ─────────────────────────────────────────────────
    tr = sub.add_parser("transform", help="Transform a phrase into a password")
    tr.add_argument("phrase", help="Your memorable phrase")
    tr.add_argument("--year", type=int, default=None,
                    help="Year to append e.g. 2024")

    # ── analyze ───────────────────────────────────────────────────
    an = sub.add_parser("analyze", help="Analyze a password")
    an.add_argument("password", help="Password to analyze")
    an.add_argument("--features", action="store_true",
                    help="Show all 15 feature values")

    # ── memory ────────────────────────────────────────────────────
    mem = sub.add_parser("memory", help="Generate memory aids for a password")
    mem.add_argument("password", help="Password to generate aids for")

    # ── vault ─────────────────────────────────────────────────────
    vault = sub.add_parser("vault", help="Manage encrypted password vault")
    vsub  = vault.add_subparsers(dest="vault_cmd")

    vsub.add_parser("init",     help="Create a new vault")
    vsub.add_parser("list",     help="List all stored labels")

    vs = vsub.add_parser("store",    help="Store a password")
    vs.add_argument("label", help="Label e.g. gmail")

    vr = vsub.add_parser("retrieve", help="Retrieve a password")
    vr.add_argument("label", help="Label to retrieve")

    vd = vsub.add_parser("delete",   help="Delete a vault entry")
    vd.add_argument("label", help="Label to delete")

    return parser


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Enable ANSI colours on Windows
    if sys.platform == "win32":
        import os
        os.system("color")

    parser = build_parser()
    args   = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    engine = PassGuardEngine()

    if args.command == "generate":
        cmd_generate(args, engine)
    elif args.command == "transform":
        cmd_transform(args, engine)
    elif args.command == "analyze":
        cmd_analyze(args, engine)
    elif args.command == "memory":
        cmd_memory(args, engine)
    elif args.command == "vault":
        if not args.vault_cmd:
            print("Available vault commands: init, store, retrieve, list, delete")
            sys.exit(0)
        cmd_vault(args, engine)


if __name__ == "__main__":
    main()
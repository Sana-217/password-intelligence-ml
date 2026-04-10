# app/app.py
"""
Flask web application — the user-facing interface for the entire project.

ROUTES
───────
  GET  /                  → index (landing page)
  GET  /register          → registration form
  POST /register          → create vault + redirect to dashboard
  GET  /login             → login form
  POST /login             → verify master password + redirect to dashboard
  GET  /dashboard         → main dashboard (requires login)
  POST /generate          → generate password (AJAX JSON)
  POST /analyze           → analyze a password (AJAX JSON)
  POST /store             → store password in vault (AJAX JSON)
  POST /retrieve          → retrieve password from vault (AJAX JSON)
  POST /delete            → delete vault entry (AJAX JSON)
  GET  /memory-aid        → memory aid page for a password
  GET  /logout            → clear session + redirect to login

SESSION
────────
  session["master"]    → master password (in-memory only, never written to disk)
  session["logged_in"] → True when authenticated

SECURITY NOTES
───────────────
  - Master password lives in Flask session (server-side, signed cookie)
  - Session expires when browser closes (SESSION_PERMANENT = False)
  - All vault routes check session["logged_in"] before executing
  - Wrong master password → WrongMasterPasswordError → 401 response
"""

import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from functools import wraps
from generator.passphrase_transformer import transform_with_scores
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, flash,
)
import os
from datetime import timedelta
from generator.password_gen  import generate_best, score_password
from generator.memory_aid    import generate_memory_aids, sentence_to_password
from security.storage        import (
    VaultManager,
    initialise_vault,
    vault_exists,
    WrongMasterPasswordError,
    EntryNotFoundError,
    EntryAlreadyExistsError,
    VaultError,
)
from preprocessing.feature_extraction import (
    get_length, get_shannon_entropy, get_charset_size,
    get_word_count, get_syllable_count, get_has_keyboard_walk,
    get_has_repeated_chars, get_is_common_password,
    get_special_char_ratio, get_uppercase_ratio, get_digit_ratio,
)
import secrets
import string
from security.spaced_repetition import (
    get_due_today, get_stats, get_progress_label
)
from security.spaced_repetition import get_all_entries, get_progress_label
from security.spaced_repetition import record_result, get_progress_label
import hmac
from security.spaced_repetition import (
    get_entry, _save, _load, _days_from_now
)
from datetime import date, timedelta
from security.spaced_repetition import (
    add_to_schedule, remove_from_schedule,
)
# ── app factory ───────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "change-this-in-production-use-secrets-module"


app.config["SESSION_PERMANENT"]        = False
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)

@app.before_request
def clear_stale_session():
    """Force logout if session has no master password loaded."""
    protected = ["dashboard", "generate", "analyze", "store",
                 "retrieve", "delete", "memory_aid", "transform", "enhance"]
    if request.endpoint in protected:
        if not session.get("logged_in") or not session.get("master"):
            session.clear()



# ── login required decorator ──────────────────────────────────────────────────

def login_required(f):
    """Redirects to /login if user is not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_vault() -> VaultManager:
    """Returns an unlocked VaultManager using the session master password."""
    vm = VaultManager()
    vm.unlock(session["master"])
    return vm


def _json_error(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


def _json_ok(data: dict):
    return jsonify({"success": True, **data})


# ── routes: auth ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        if vault_exists():
            flash("A vault already exists. Please log in.", "info")
            return redirect(url_for("login"))
        return render_template("register.html")

    username = request.form.get("username", "").strip()
    master   = request.form.get("master_password", "").strip()
    confirm  = request.form.get("confirm_password", "").strip()

    if not username:
        flash("Please enter your name.", "danger")
        return render_template("register.html")
    if not master:
        flash("Master password cannot be empty.", "danger")
        return render_template("register.html")
    if len(master) < 8:
        flash("Master password must be at least 8 characters.", "danger")
        return render_template("register.html")
    if master != confirm:
        flash("Passwords do not match.", "danger")
        return render_template("register.html")

    try:
        initialise_vault(master)

        hint      = request.form.get("hint", "").strip()
        name_file = ROOT / "vault_meta.json"
        name_file.write_text(json.dumps({"username": username, "hint": hint}))

        session["logged_in"] = True
        session["master"]    = master
        session["username"]  = username
        flash(f"Welcome, {username}! Your vault has been created.", "success")
        return redirect(url_for("dashboard"))

    except VaultError as e:
        flash(str(e), "danger")
        return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if not vault_exists():
            flash("No vault found. Please register first.", "info")
            return redirect(url_for("register"))
        return render_template("login.html")

    master = request.form.get("master_password", "").strip()
    if not master:
        flash("Please enter your master password.", "danger")
        return render_template("login.html")

    try:
        vm = VaultManager()
        vm.unlock(master)
        vm.lock()
        session["logged_in"] = True
        session["master"]    = master

        # Load username if saved
        name_file = ROOT / "vault_meta.json"
        if name_file.exists():
            meta = json.loads(name_file.read_text())
            session["username"] = meta.get("username", "User")
        else:
            session["username"] = "User"

        return redirect(url_for("dashboard"))
    except WrongMasterPasswordError:
        flash("Incorrect master password.", "danger")
        return render_template("login.html")
    except Exception as e:
        flash(f"Error: {e}", "danger")
        return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


@app.route("/get-hint")
def get_hint():
    name_file = ROOT / "vault_meta.json"
    if name_file.exists():
        meta = json.loads(name_file.read_text())
        return jsonify({"hint": meta.get("hint", "")})
    return jsonify({"hint": ""})


@app.route("/reset-vault", methods=["POST"])
def reset_vault():
    vault_file = ROOT / "vault.json"
    meta_file  = ROOT / "vault_meta.json"
    try:
        if vault_file.exists():
            os.remove(vault_file)
        if meta_file.exists():
            os.remove(meta_file)
        session.clear()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
# ── routes: dashboard ─────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    try:
        vm     = _get_vault()
        labels = vm.list_labels()
        count  = len(labels)
        vm.lock()
    except Exception:
        labels = []
        count  = 0
    return render_template("dashboard.html", labels=labels, count=count)


# ── routes: generate (AJAX) ───────────────────────────────────────────────────

@app.route("/generate", methods=["POST"])
@login_required
def generate():
    """
    Accepts JSON or form data. Returns best generated password + ML scores.

    Request body:
        mode         : "passphrase" | "pattern" | "random"
        n_words      : int (passphrase only, default 4)
        separator    : str (passphrase only, default "-")
        pattern      : str (pattern only, e.g. "Wwww-ddd-s")
        length       : int (random only, default 16)
        n_candidates : int (default 5)

    Response:
        { success, password, strength_label, memorability_label,
          combined_score, strength_proba, memorability_proba, mode }
    """
    data = request.get_json(silent=True) or request.form

    mode         = data.get("mode", "passphrase")
    n_candidates = int(data.get("n_candidates", 5))

    try:
        if mode == "passphrase":
            result = generate_best(
                mode="passphrase",
                n_words=int(data.get("n_words", 4)),
                separator=data.get("separator", "-"),
                capitalise=data.get("capitalise", "true") != "false",
                n_candidates=n_candidates,
            )
        elif mode == "pattern":
            pattern = data.get("pattern", "")
            if not pattern:
                return _json_error("Pattern is required for pattern mode.")
            result = generate_best(
                mode="pattern",
                pattern=pattern,
                n_candidates=n_candidates,
            )
        elif mode == "random":
            result = generate_best(
                mode="random",
                length=int(data.get("length", 16)),
                use_uppercase=data.get("use_uppercase", "true") != "false",
                use_digits=data.get("use_digits", "true") != "false",
                use_special=data.get("use_special", "true") != "false",
                n_candidates=n_candidates,
            )
        else:
            return _json_error(f"Unknown mode: {mode}")

        best = result["best"]
        return _json_ok({
            "password":           best["password"],
            "strength_label":     best["strength_label"],
            "memorability_label": best["memorability_label"],
            "combined_score":     best["combined_score"],
            "strength_proba":     best["strength_proba"],
            "memorability_proba": best["memorability_proba"],
            "mode":               mode,
            "all_candidates":     [
                {
                    "password": c["password"],
                    "score":    c["combined_score"],
                    "strength": c["strength_label"],
                }
                for c in result["all"]
            ],
        })

    except Exception as e:
        return _json_error(str(e))

@app.route("/transform", methods=["POST"])
@login_required
def transform():
    data   = request.get_json(silent=True) or request.form
    phrase = data.get("phrase", "").strip()
    year   = data.get("year", None)

    if not phrase:
        return _json_error("Phrase cannot be empty.")

    try:
        result = transform_with_scores(
            phrase,
            year=int(year) if year else None,
        )
        return _json_ok({
            "password":    result["password"],
            "candidates":  result["candidates"],
            "pipeline":    result["pipeline"],
            "explanation": result["explanation"],
            "tips":        result["strength_tips"],
            "ml_scores":   result.get("ml_scores"),
        })
    except ValueError as e:
        return _json_error(str(e))
    except Exception as e:
        return _json_error(str(e))
# ── routes: analyze (AJAX) ────────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    """
    Scores a user-supplied password through both ML models.

    Request body:
        password : str

    Response:
        { success, password, strength_label, memorability_label,
          combined_score, strength_proba, memorability_proba }
    """
    data     = request.get_json(silent=True) or request.form
    password = data.get("password", "").strip()

    if not password:
        return _json_error("Password cannot be empty.")

    try:
        scores = score_password(password)
        return _json_ok({
            "password":           scores["password"],
            "strength_label":     scores["strength_label"],
            "memorability_label": scores["memorability_label"],
            "combined_score":     scores["combined_score"],
            "strength_proba":     scores["strength_proba"],
            "memorability_proba": scores["memorability_proba"],
        })
    except Exception as e:
        return _json_error(str(e))

@app.route("/enhance", methods=["POST"])
@login_required
def enhance():
    """
    Analyzes a weak/medium password and returns:
      1. Specific reasons why it's weak
      2. Actionable suggestions
      3. An enhanced version of the password
    """
    data     = request.get_json(silent=True) or request.form
    password = data.get("password", "").strip()

    if not password:
        return _json_error("Password cannot be empty.")

    try:
        
        

        suggestions = []
        enhanced    = password

        # ── diagnose and suggest ──────────────────────────────────
        if get_is_common_password(password):
            suggestions.append({
                "issue": "This is one of the most commonly used passwords",
                "fix":   "It appears in every attacker's dictionary — change it completely",
                "type":  "critical",
            })

        if get_length(password) < 10:
            suggestions.append({
                "issue": f"Too short — only {get_length(password)} characters",
                "fix":   "Add at least 4 more characters — length is the single biggest factor in security",
                "type":  "error",
            })

        if get_has_keyboard_walk(password):
            suggestions.append({
                "issue": "Contains a keyboard walk pattern (e.g. qwerty, 12345)",
                "fix":   "Replace the sequential pattern with random characters",
                "type":  "error",
            })

        if get_has_repeated_chars(password):
            suggestions.append({
                "issue": "Contains repeated characters (e.g. aaa, 111)",
                "fix":   "Replace repeated characters with varied ones",
                "type":  "warning",
            })

        if get_charset_size(password) < 36:
            if not any(c in string.ascii_uppercase for c in password):
                suggestions.append({
                    "issue": "No uppercase letters",
                    "fix":   "Capitalise at least one word or add an uppercase letter",
                    "type":  "warning",
                })
            if not any(c in string.digits for c in password):
                suggestions.append({
                    "issue": "No digits",
                    "fix":   "Add 2–3 numbers — avoid predictable ones like 123",
                    "type":  "warning",
                })

        if get_special_char_ratio(password) == 0:
            suggestions.append({
                "issue": "No special characters",
                "fix":   "Add at least one special character (!, @, #, $, &)",
                "type":  "warning",
            })

        if get_shannon_entropy(password) < 2.5:
            suggestions.append({
                "issue": "Very low entropy — characters are too repetitive",
                "fix":   "Use a wider variety of characters",
                "type":  "warning",
            })

        if get_word_count(password) == 0 and get_syllable_count(password) < 2:
            suggestions.append({
                "issue": "Not memorable — no recognisable words or syllables",
                "fix":   "Add a real word to create a memory anchor, e.g. append your pet's name",
                "type":  "info",
            })

        if not suggestions:
            suggestions.append({
                "issue": "Password looks reasonable",
                "fix":   "Consider using our Passphrase mode for something both stronger and more memorable",
                "type":  "info",
            })

        # ── build enhanced version ────────────────────────────────
        enhanced = _build_enhanced(password)

        # ── score the enhanced version ────────────────────────────
        enhanced_scores = score_password(enhanced)

        return _json_ok({
            "original":        password,
            "suggestions":     suggestions,
            "enhanced":        enhanced,
            "enhanced_scores": enhanced_scores,
        })

    except Exception as e:
        return _json_error(str(e))


def _build_enhanced(password: str) -> str:
    """
    Builds an improved version of a weak password by applying
    targeted enhancements. Preserves the original structure
    so the user still recognises it.
    """
    enhanced = password

    # 1. Capitalise first letter if all lowercase
    if enhanced == enhanced.lower():
        enhanced = enhanced.capitalize()

    # 2. Add a special character if missing
    if not any(c in string.punctuation for c in enhanced):
        enhanced += secrets.choice("!@#$&*")

    # 3. Add digits if missing
    if not any(c.isdigit() for c in enhanced):
        enhanced += str(secrets.randbelow(90) + 10)  # 10–99

    # 4. If still too short, append a random word fragment
    if len(enhanced) < 12:
        fragments = ["Sky", "Fox", "Blaze", "Nova", "Crypt", "Storm"]
        enhanced += secrets.choice(fragments)

    # 5. If still no uppercase mid-word, capitalise a random vowel position
    if not any(c.isupper() for c in enhanced[1:]):
        vowel_positions = [
            i for i, c in enumerate(enhanced)
            if c.lower() in "aeiou" and i > 0
        ]
        if vowel_positions:
            pos      = secrets.choice(vowel_positions)
            enhanced = enhanced[:pos] + enhanced[pos].upper() + enhanced[pos+1:]

    return enhanced
# ── routes: vault CRUD (AJAX) ─────────────────────────────────────────────────

@app.route("/store", methods=["POST"])
@login_required
def store():
    """Stores a password in the vault under a label."""
    data     = request.get_json(silent=True) or request.form
    label    = data.get("label", "").strip()
    password = data.get("password", "").strip()

    if not label:
        return _json_error("Label cannot be empty.")
    if not password:
        return _json_error("Password cannot be empty.")

    try:
        vm = _get_vault()
        vm.store(label, password)
        add_to_schedule(label)
        vm.lock()
        return _json_ok({"label": label, "message": f"'{label}' stored."})
    except EntryAlreadyExistsError:
        return _json_error(
            f"'{label}' already exists. Use update to overwrite.", 409
        )
    except Exception as e:
        return _json_error(str(e))


@app.route("/retrieve", methods=["POST"])
@login_required
def retrieve():
    """Retrieves and decrypts a stored password."""
    data  = request.get_json(silent=True) or request.form
    label = data.get("label", "").strip()

    if not label:
        return _json_error("Label cannot be empty.")

    try:
        vm       = _get_vault()
        password = vm.retrieve(label)
        vm.lock()
        return _json_ok({"label": label, "password": password})
    except EntryNotFoundError:
        return _json_error(f"No entry found for '{label}'.", 404)
    except Exception as e:
        return _json_error(str(e))


@app.route("/update", methods=["POST"])
@login_required
def update():
    """Updates an existing vault entry."""
    data         = request.get_json(silent=True) or request.form
    label        = data.get("label", "").strip()
    new_password = data.get("password", "").strip()

    if not label or not new_password:
        return _json_error("Label and password are required.")

    try:
        vm = _get_vault()
        vm.update(label, new_password)
        vm.lock()
        return _json_ok({"label": label, "message": f"'{label}' updated."})
    except EntryNotFoundError:
        return _json_error(f"No entry found for '{label}'.", 404)
    except Exception as e:
        return _json_error(str(e))


@app.route("/delete", methods=["POST"])
@login_required
def delete():
    """Deletes a vault entry."""
    data  = request.get_json(silent=True) or request.form
    label = data.get("label", "").strip()

    if not label:
        return _json_error("Label cannot be empty.")

    try:
        vm = _get_vault()
        vm.delete(label)
        remove_from_schedule(label)
        vm.lock()
        return _json_ok({"label": label, "message": f"'{label}' deleted."})
    except EntryNotFoundError:
        return _json_error(f"No entry found for '{label}'.", 404)
    except Exception as e:
        return _json_error(str(e))


@app.route("/list-labels")
@login_required
def list_labels_route():
    """Returns all stored labels as JSON."""
    try:
        vm     = _get_vault()
        labels = vm.list_labels()
        vm.lock()
        return _json_ok({"labels": labels})
    except Exception as e:
        return _json_error(str(e))


# ── routes: memory aid ────────────────────────────────────────────────────────

@app.route("/memory-aid", methods=["GET", "POST"])
@login_required
def memory_aid():
    password = request.args.get("password") or (
        (request.get_json(silent=True) or request.form).get("password", "")
    )
    aids = None
    if password:
        try:
            aids = generate_memory_aids(password.strip())
        except Exception as e:
            flash(str(e), "danger")
    return render_template("memory_aid.html", aids=aids, password=password)


@app.route("/sentence-to-password", methods=["POST"])
@login_required
def from_sentence():
    data     = request.get_json(silent=True) or request.form
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return _json_error("Sentence cannot be empty.")
    try:
        result = sentence_to_password(sentence)
        return _json_ok(result)
    except Exception as e:
        return _json_error(str(e))

@app.route("/logout-beacon", methods=["POST"])
def logout_beacon():
    return "", 204   # do nothing — session timeout handles cleanup

# ─────────────────────────────────────────────────────────────────────────────
# SPACED REPETITION ROUTES
# Add these routes to app/app.py
#
# Also add this import near the top of app.py:
#   from security.spaced_repetition import (
#       add_to_schedule, remove_from_schedule,
#       get_due_today, get_all_entries,
#       get_stats, record_result, get_progress_label,
#   )
#
# And modify the existing /store route to also call add_to_schedule(label)
# And modify the existing /delete route to also call remove_from_schedule(label)
# ─────────────────────────────────────────────────────────────────────────────


@app.route("/practice-due")
@login_required
def practice_due():
    """
    Returns passwords due for recall practice today.

    Response:
        {
          success: true,
          due:     [ { label, next_review, review_count, streak,
                       mastered, last_result, progress_label } ],
          stats:   { total, due, mastered, upcoming, streak }
        }
    """
    try:
       
        due_entries = get_due_today()
        stats       = get_stats()

        due_list = []
        for e in due_entries:
            due_list.append({
                "label":          e["label"],
                "next_review":    e.get("next_review"),
                "review_count":   e.get("review_count", 0),
                "streak":         e.get("streak", 0),
                "mastered":       e.get("mastered", False),
                "last_result":    e.get("last_result"),
                "progress_label": get_progress_label(e),
            })

        return _json_ok({ "due": due_list, "stats": stats })
    except Exception as e:
        return _json_error(str(e))


@app.route("/practice-all")
@login_required
def practice_all():
    """Returns ALL scheduled entries (for the full schedule view)."""
    try:
        
        entries = get_all_entries()
        result  = []
        for e in entries:
            result.append({
                "label":          e["label"],
                "next_review":    e.get("next_review"),
                "review_count":   e.get("review_count", 0),
                "streak":         e.get("streak", 0),
                "mastered":       e.get("mastered", False),
                "last_result":    e.get("last_result"),
                "progress_label": get_progress_label(e),
            })
        return _json_ok({ "entries": result })
    except Exception as e:
        return _json_error(str(e))


@app.route("/practice-check", methods=["POST"])
@login_required
def practice_check():
    """
    Verify a recall attempt for a given label.

    The user types the password from memory. This route:
      1. Retrieves the real password from the vault
      2. Compares with the user's attempt (constant-time comparison)
      3. Records the result in the review schedule
      4. Returns whether it was correct + next review date

    Request body:
        { label: str, attempt: str }

    Response:
        {
          success:      true,
          correct:      bool,
          label:        str,
          next_review:  str | null,
          review_count: int,
          streak:       int,
          mastered:     bool,
          message:      str,
          progress_label: str,
        }
    """
    data    = request.get_json(silent=True) or request.form
    label   = data.get("label",   "").strip()
    attempt = data.get("attempt", "").strip()

    if not label:
        return _json_error("Label is required.")
    if not attempt:
        return _json_error("Please type the password to check.")

    try:
        
        # Retrieve the real password from vault
        vm       = _get_vault()
        real_pwd = vm.retrieve(label)
        vm.lock()

        # Constant-time comparison to prevent timing attacks
        correct = hmac.compare_digest(attempt, real_pwd)

        # Record result in spaced repetition schedule
        entry = record_result(label, success=correct)

        if correct:
            if entry.get("mastered"):
                message = f"🎉 Perfect! '{label}' is now MASTERED — no more reviews needed!"
            else:
                days = (entry.get("next_review") or "")
                message = (
                    f"✅ Correct! Next review in "
                    f"{_days_until(entry.get('next_review'))} day(s)."
                )
        else:
            message = (
                f"❌ Incorrect. The password was: {real_pwd}\n"
                f"Review reset — practice again tomorrow."
            )

        return _json_ok({
            "correct":        correct,
            "label":          label,
            "next_review":    entry.get("next_review"),
            "review_count":   entry.get("review_count", 0),
            "streak":         entry.get("streak", 0),
            "mastered":       entry.get("mastered", False),
            "message":        message,
            "progress_label": get_progress_label(entry),
            "real_password":  real_pwd if not correct else None,
        })

    except Exception as e:
        return _json_error(str(e))


@app.route("/practice-skip", methods=["POST"])
@login_required
def practice_skip():
    """
    Skip a review — push it to tomorrow without marking pass or fail.
    Useful when the user knows the password but doesn't want to type it now.
    """
    data  = request.get_json(silent=True) or request.form
    label = data.get("label", "").strip()
    if not label:
        return _json_error("Label is required.")

    try:
        
        sched = _load()
        if label in sched:
            sched[label]["next_review"] = _days_from_now(1)
            _save(sched)
        return _json_ok({ "label": label, "message": "Skipped — review tomorrow." })
    except Exception as e:
        return _json_error(str(e))


# ── helper ─────────────────────────────────────────────────────────────────────

def _days_until(date_str):
    """Return number of days until a date string (YYYY-MM-DD)."""
    if not date_str:
        return 0
    
    try:
        delta = (date.fromisoformat(date_str) - date.today()).days
        return max(0, delta)
    except Exception:
        return 0
# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  PassGuard — AI Password Security System")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
    
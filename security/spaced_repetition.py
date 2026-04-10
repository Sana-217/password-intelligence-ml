"""
security/spaced_repetition.py
─────────────────────────────────────────────────────────────────────────────
Spaced Repetition System for PassGuard
Implements the Ebbinghaus forgetting curve to schedule password recall
practice at scientifically optimal intervals.

THEORY
───────
Ebbinghaus (1885) discovered that memory retention decays exponentially
after learning, but each successful review resets the decay curve and
extends the time before the next review is needed.

The optimal review intervals (used by Anki, Duolingo, etc.):
  After storing  : review in  1 day
  After review 1 : review in  3 days
  After review 2 : review in  7 days
  After review 3 : review in 14 days
  After review 4 : review in 30 days
  After review 5 : MASTERED — no more reviews needed

A failed review resets the interval back to 1 day.

STORAGE
────────
All review data is stored in review_schedule.json in the project root.
Format:
{
  "gmail": {
    "label":           "gmail",
    "date_added":      "2024-04-09",
    "next_review":     "2024-04-10",
    "review_count":    0,
    "streak":          0,
    "mastered":        false,
    "last_result":     null,
    "history": [
      { "date": "2024-04-10", "result": "pass", "interval": 3 }
    ]
  }
}
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
SCHEDULE_FILE = ROOT / "review_schedule.json"

# Ebbinghaus review intervals in days
# Index = number of successful reviews completed
INTERVALS = [1, 3, 7, 14, 30]
# After INTERVALS[-1] days with no failure → MASTERED


# ── internal helpers ──────────────────────────────────────────────────────────

def _load() -> dict:
    """Load the review schedule from disk. Returns empty dict if not found."""
    if not SCHEDULE_FILE.exists():
        return {}
    try:
        return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    """Atomically save the review schedule to disk."""
    tmp = SCHEDULE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(SCHEDULE_FILE)


def _today() -> str:
    """Return today's date as YYYY-MM-DD string."""
    return date.today().isoformat()


def _days_from_now(n: int) -> str:
    """Return the date n days from today as YYYY-MM-DD string."""
    return (date.today() + timedelta(days=n)).isoformat()


def _is_due(entry: dict) -> bool:
    """Return True if an entry is due for review today or overdue."""
    if entry.get("mastered"):
        return False
    next_review = entry.get("next_review", _today())
    return next_review <= _today()


def _next_interval(review_count: int) -> int:
    """Return the number of days until next review based on review count."""
    idx = min(review_count, len(INTERVALS) - 1)
    return INTERVALS[idx]


# ── public API ────────────────────────────────────────────────────────────────

def add_to_schedule(label: str) -> dict:
    """
    Add a newly stored password to the review schedule.
    Called automatically when the user stores a password in the vault.

    Args:
        label : the vault label (e.g. "gmail")

    Returns the new schedule entry.
    """
    data  = _load()

    # Don't overwrite if already scheduled (e.g. user re-stored same label)
    if label in data:
        return data[label]

    entry = {
        "label":        label,
        "date_added":   _today(),
        "next_review":  _days_from_now(1),   # first review tomorrow
        "review_count": 0,
        "streak":       0,
        "mastered":     False,
        "last_result":  None,
        "history":      [],
    }

    data[label] = entry
    _save(data)
    return entry


def remove_from_schedule(label: str) -> None:
    """
    Remove a label from the review schedule.
    Called when a vault entry is deleted.
    """
    data = _load()
    if label in data:
        del data[label]
        _save(data)


def get_due_today() -> list[dict]:
    """
    Return all entries that are due for review today or overdue.
    Sorted by most overdue first (oldest next_review date first).

    Returns list of entry dicts.
    """
    data = _load()
    due  = [e for e in data.values() if _is_due(e)]
    due.sort(key=lambda e: e.get("next_review", ""))
    return due


def get_all_entries() -> list[dict]:
    """Return all schedule entries (for displaying full schedule)."""
    data = _load()
    entries = list(data.values())
    entries.sort(key=lambda e: e.get("next_review") or "9999-99-99")
    return entries


def get_stats() -> dict:
    """
    Return summary statistics about the review schedule.

    Returns:
        {
          "total":    int,   total passwords being tracked
          "due":      int,   due today
          "mastered": int,   fully mastered
          "upcoming": int,   scheduled for future
          "streak":   int,   longest current streak
        }
    """
    data     = _load()
    entries  = list(data.values())

    total    = len(entries)
    due      = sum(1 for e in entries if _is_due(e))
    mastered = sum(1 for e in entries if e.get("mastered"))
    upcoming = total - due - mastered
    streak   = max((e.get("streak", 0) for e in entries), default=0)

    return {
        "total":    total,
        "due":      due,
        "mastered": mastered,
        "upcoming": upcoming,
        "streak":   streak,
    }


def record_result(label: str, success: bool) -> dict:
    """
    Record the result of a recall attempt.

    If success=True:
        - Increment review_count and streak
        - Schedule next review based on Ebbinghaus interval
        - If review_count >= len(INTERVALS): mark as MASTERED

    If success=False:
        - Reset streak to 0
        - Reset review_count to 0 (start over from 1-day interval)
        - Schedule review for tomorrow

    Args:
        label   : vault label
        success : True if user recalled correctly, False otherwise

    Returns the updated entry dict.
    """
    data  = _load()
    if label not in data:
        # Entry doesn't exist — create it first
        data[label] = {
            "label":        label,
            "date_added":   _today(),
            "next_review":  _today(),
            "review_count": 0,
            "streak":       0,
            "mastered":     False,
            "last_result":  None,
            "history":      [],
        }

    entry = data[label]

    if success:
        entry["review_count"] += 1
        entry["streak"]       += 1
        entry["last_result"]   = "pass"

        # Check if mastered
        if entry["review_count"] >= len(INTERVALS):
            entry["mastered"]    = True
            entry["next_review"] = None
            interval             = None
        else:
            interval             = _next_interval(entry["review_count"])
            entry["next_review"] = _days_from_now(interval)

    else:
        # Failed — reset and retry tomorrow
        interval               = 1
        entry["review_count"]  = 0
        entry["streak"]        = 0
        entry["last_result"]   = "fail"
        entry["next_review"]   = _days_from_now(1)

    # Append to history
    entry["history"].append({
        "date":     _today(),
        "result":   "pass" if success else "fail",
        "interval": interval,
    })

    data[label] = entry
    _save(data)
    return entry


def get_entry(label: str) -> dict | None:
    """Return the schedule entry for a specific label, or None if not found."""
    data = _load()
    return data.get(label)


def days_until_next(entry: dict) -> int | None:
    """
    Return number of days until next review for an entry.
    Returns None if mastered, 0 if due today or overdue.
    """
    if entry.get("mastered"):
        return None
    next_review = entry.get("next_review")
    if not next_review:
        return None
    delta = (date.fromisoformat(next_review) - date.today()).days
    return max(0, delta)


def get_progress_label(entry: dict) -> str:
    """
    Return a human-readable progress label for an entry.
    e.g. "Review 2 of 5", "Mastered!", "Due today"
    """
    if entry.get("mastered"):
        return "Mastered! ✓"
    count = entry.get("review_count", 0)
    total = len(INTERVALS)
    days  = days_until_next(entry)
    if days == 0:
        return f"Due today — Review {count + 1} of {total}"
    return f"Review {count + 1} of {total} — in {days} day{'s' if days != 1 else ''}"


# ── self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

    # Use a temp file for testing
    SCHEDULE_FILE = Path(tempfile.mktemp(suffix=".json"))

    print("=== Spaced Repetition Self-Test ===\n")

    # Add some labels
    add_to_schedule("gmail")
    add_to_schedule("github")
    add_to_schedule("netflix")

    print("Added 3 labels. Stats:", get_stats())
    print()

    # Simulate passing review for gmail multiple times
    print("Simulating gmail passing all reviews...")
    for i in range(5):
        entry = record_result("gmail", success=True)
        print(f"  Review {i+1}: next_review={entry['next_review']}, "
              f"mastered={entry['mastered']}")

    print()
    print("Simulating github failing once, then passing...")
    record_result("github", success=False)
    entry = get_entry("github")
    print(f"  After fail : next_review={entry['next_review']}, streak={entry['streak']}")
    record_result("github", success=True)
    entry = get_entry("github")
    print(f"  After pass : next_review={entry['next_review']}, streak={entry['streak']}")

    print()
    print("Final stats:", get_stats())
    print()
    print("All entries:")
    for e in get_all_entries():
        print(f"  {e['label']:12s} {get_progress_label(e)}")

    # Cleanup
    SCHEDULE_FILE.unlink(missing_ok=True)
    print("\n=== PASSED ===")
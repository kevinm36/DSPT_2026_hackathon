"""
Client-side user profile builder for ADS16 corpus.

Parses the CSV files locally and returns a profile dict ready to send
to the ranking agent.
"""

import csv
import io
from pathlib import Path


def _read_semicolon_csv(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8-sig").lstrip("\n")
    return list(csv.reader(io.StringIO(text), delimiter=";", quotechar='"'))


def _parse_inf(user_dir: Path) -> dict:
    rows = _read_semicolon_csv(user_dir / f"{user_dir.name}-INF.csv")
    rows = [r for r in rows if any(c.strip() for c in r)]
    headers = [h.strip() for h in rows[0]]
    values = [v.strip() for v in rows[1]]
    mapping = dict(zip(headers, values))
    return {
        "gender": mapping.get("Gender", ""),
        "age": mapping.get("Age", ""),
        "job": mapping.get("Type of Job", ""),
        "income": mapping.get("Income", ""),
        "timepass": mapping.get("Timepass", ""),
        "fave_sports": mapping.get("Fave Sports", ""),
    }


def _parse_pref(user_dir: Path) -> dict:
    rows = _read_semicolon_csv(user_dir / f"{user_dir.name}-PREF.csv")
    rows = [r for r in rows if any(c.strip() for c in r)]
    headers = [h.strip() for h in rows[0]]
    values = [v.strip() for v in rows[1]]
    mapping = dict(zip(headers, values))
    return {
        "websites": mapping.get("Most visited websites", ""),
        "music": mapping.get("Most listened musics", ""),
        "movies": mapping.get("Most watched movies", ""),
        "tv": mapping.get("Most watched tv programmes", ""),
        "books": mapping.get("Most read books", ""),
    }


def _parse_sentiment(user_dir: Path) -> tuple[list[str], list[str]]:
    def _extract_labels(path: Path) -> list[str]:
        rows = _read_semicolon_csv(path)
        rows = [r for r in rows if any(c.strip() for c in r)]
        if len(rows) < 3:
            return []
        return [v.strip() for v in rows[2] if v.strip()]

    pos = _extract_labels(user_dir / f"{user_dir.name}-IM-POS.csv")
    neg = _extract_labels(user_dir / f"{user_dir.name}-IM-NEG.csv")
    return pos, neg


def build_user_profile(user_id: str, corpus_root: str) -> dict:
    """Parse ADS16 CSVs and return a profile dict for the ranking agent."""
    user_dir = Path(corpus_root) / user_id
    inf = _parse_inf(user_dir)
    pref = _parse_pref(user_dir)
    pos_labels, neg_labels = _parse_sentiment(user_dir)
    return {
        "user_id": user_id,
        "inf": inf,
        "pref": pref,
        "pos_labels": pos_labels,
        "neg_labels": neg_labels,
    }

"""Shared player-name normalisation and fuzzy matching utilities.

Extracted from ``ml.data.import_quotations`` so both the listone importer
and the roster (rose) importer can reuse the same logic without duplication.

Public surface is intentionally small and pure (no DB / pandas dependency
required for the core helpers).
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, Optional, Sequence

# ── Constants (kept in sync with import_quotations) ──────────────────────────

_COMPOUND_PREFIXES: frozenset[str] = frozenset({
    "de", "del", "della", "delle", "di", "da", "dal", "dalla",
    "do", "dos", "das",
    "el", "la", "los", "las", "y",
    "van", "von", "der", "den", "ten", "bin", "al",
    "st", "st.", "san", "santa",
})

_SUFFIX_STRIP_RE = re.compile(
    r"\b(jr|sr|ii|iii|iv)\b\.?$", flags=re.IGNORECASE
)
_TEAM_SUFFIX_RE = re.compile(
    r"\b(fc|ss|asd|ac|us|as|calcio|football)\b\.?",
    flags=re.IGNORECASE,
)
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")

TEAM_ALIASES: dict[str, str] = {
    "inter": "internazionale",
}

# Thresholds used by the roster matcher (plan D3).
AUTO_MATCH_THRESHOLD: float = 0.92
REVIEW_MATCH_THRESHOLD: float = 0.75


# ── Normalisation ────────────────────────────────────────────────────────────


def strip_accents(s: str) -> str:
    """Lower-case, strip accents. Used as building block for join keys."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def normalise_player_name(name: str) -> str:
    """Normalise player name for matching.

    Examples:
        "Dybala  Paulo"     → "dybala paulo"
        "M'Bala Nzola"      → "mbala nzola"
        "Pau López"         → "pau lopez"
        "Cuenca A. *"       → "cuenca a"   (caller should strip * first)
    """
    n = strip_accents(str(name))
    n = _NONALNUM_RE.sub(" ", n)
    n = _SUFFIX_STRIP_RE.sub("", n)
    return " ".join(n.split())


def normalise_team(team: str) -> str:
    """Normalise team name. Drops common suffixes (FC, Calcio, etc.)."""
    n = strip_accents(str(team))
    n = _TEAM_SUFFIX_RE.sub(" ", n)
    n = _NONALNUM_RE.sub(" ", n)
    return " ".join(n.split())


def apply_team_alias(team_norm: str) -> str:
    """Map known Fantacalcio team variants to a canonical form."""
    return TEAM_ALIASES.get(team_norm, team_norm)


def strip_trailing_initial(name_norm: str) -> str:
    """Drop a trailing 1–2 character initial / abbreviation.

    * ``"martinez l"``  → ``"martinez"``
    * ``"martinez jo"`` → ``"martinez"``
    * ``"pessina mas"`` → ``"pessina mas"``  (3+ chars kept)
    """
    parts = name_norm.split()
    if len(parts) >= 2 and 1 <= len(parts[-1]) <= 2:
        return " ".join(parts[:-1])
    return name_norm


def last_name_token(name_norm: str) -> str:
    """Extract the surname token (or multi-word compound surname).

    Handles compound prefixes (``de ketelaere``, ``kolo muani``, ``di gregorio``)
    and trailing initials already stripped by :func:`strip_trailing_initial`.
    """
    stripped = strip_trailing_initial(name_norm)
    tokens = stripped.split()
    if not tokens:
        return ""
    if len(tokens) == 1 and len(tokens[0]) == 1:
        return ""
    if len(tokens) >= 2 and len(tokens[-1]) == 1:
        tokens = tokens[:-1]
        if not tokens:
            return ""
    if len(tokens) >= 2 and tokens[-2] in _COMPOUND_PREFIXES:
        return " ".join(tokens[-2:])
    if len(tokens) >= 2 and tokens[-1] in _COMPOUND_PREFIXES:
        return tokens[-1]
    return tokens[-1]


# ── Scoring helpers ──────────────────────────────────────────────────────────


def sequence_ratio(a: str, b: str) -> float:
    """``SequenceMatcher`` ratio in [0, 1]. Empty strings → 0."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def best_fuzzy_candidate(
    query_last_name: str,
    candidates: Sequence[tuple[str, object]],
    *,
    min_score: float = REVIEW_MATCH_THRESHOLD,
) -> Optional[tuple[object, float]]:
    """Return ``(payload, score)`` of the best candidate above ``min_score``.

    ``candidates`` is an iterable of ``(last_name_norm, payload)``.
    On ties the first highest score wins (stable).
    """
    if not query_last_name or not candidates:
        return None

    best_payload: object | None = None
    best_score = 0.0
    for cand_surname, payload in candidates:
        if not cand_surname:
            continue
        score = sequence_ratio(query_last_name, cand_surname)
        if score >= min_score and score > best_score:
            best_score = score
            best_payload = payload

    if best_payload is None:
        return None
    return best_payload, best_score


def score_name_pair(
    query_name: str,
    catalog_name: str,
    *,
    use_last_name_only: bool = True,
) -> float:
    """Score similarity between a roster name and a catalog name.

    When ``use_last_name_only`` is True (default for Fantacalcio-style
    abbreviated names), comparison is performed on the extracted surname
    tokens. Otherwise the full normalised strings are compared.
    """
    q_norm = normalise_player_name(query_name)
    c_norm = normalise_player_name(catalog_name)
    if not q_norm or not c_norm:
        return 0.0

    if use_last_name_only:
        q_ln = last_name_token(q_norm)
        c_ln = last_name_token(c_norm)
        # Also try full-string ratio as a secondary signal for short names
        # that are already surname-only.
        ln_score = sequence_ratio(q_ln, c_ln) if q_ln and c_ln else 0.0
        full_score = sequence_ratio(q_norm, c_norm)
        return max(ln_score, full_score)

    return sequence_ratio(q_norm, c_norm)

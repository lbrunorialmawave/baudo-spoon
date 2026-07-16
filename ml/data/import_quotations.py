"""Importer for Fantacalcio listoni (auction valuation spreadsheets).

Reads ``Quotazioni_Fantacalcio_Stagione_<YYYY_YY>.xlsx`` files from a
directory and persists them into two PostgreSQL tables:

* ``player_quotations``  — raw Qt.A / Qt.I / FVM values
* ``player_id_map``      — Fantacalcio-id ↔ player_fotmob_id bridge

The FotMob-side resolution happens in two passes:

1. **Deterministic exact match** on (normalised name, normalised team,
   canonical_role). Uses ``player_profiles`` joined with
   ``player_season_stats`` (only one season of stats per fantacalcio row
   is needed, so the latest season each player appears in is used as the
   reference).

2. **Fuzzy fallback** (optional) using ``difflib.SequenceMatcher`` for
   players that didn't match exactly. Anything still unmatched is logged
   and persisted with ``match_method='unmatched'`` so the operator can
   resolve it manually.

The module is CLI-runnable::

    python -m ml.data.import_quotations \\
        --quotazioni-dir ../quotazioni \\
        --source listone_fantagazzetta
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd
import sqlalchemy as sa

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: Fantacalcio role codes → canonical_role used by the rest of the pipeline.
ROLE_MAP: dict[str, str] = {
    "P": "GK",
    "D": "DEF",
    "C": "MID",
    "A": "FWD",
}

#: Expected columns of the 'Tutti' sheet after canonicalisation.
QUOTATION_COLUMNS: list[str] = [
    "fantacalcio_id", "r", "rm", "name", "team",
    "qt_a", "qt_i", "diff_val",
    "qt_a_m", "qt_i_m", "diff_val_m",
    "fvm", "fvm_m",
]

#: Sheet name to load from the workbook.
SHEET_NAME = "Tutti"

#: Header is on the second row (row index 1) of the workbook.
HEADER_ROW = 1

#: Threshold above which a surname-only fuzzy match is accepted.
#: After we isolate the *surname* on both sides the strings should be very
#: close, so 0.88 is a tight but realistic floor.
FUZZY_MATCH_THRESHOLD: float = 0.88

#: How many candidate matches to inspect per unmatched Fantacalcio row.
#: 0 = no cap (run over the entire reference). The reference has ~1k rows
#: and a season has ~500 rows, so 0.5M SequenceMatcher calls finish in
#: well under a second on a modern machine.
FUZZY_CANDIDATE_POOL: int = 0

#: Italian/Spanish/Portuguese/German/Dutch prefix particles that belong to
#: the *surname* rather than the first name ("De Ketelaere", "Kolo Muani", …).
_COMPOUND_PREFIXES: frozenset[str] = frozenset({
    "de", "del", "della", "delle", "di", "da", "dal", "dalla",
    "do", "dos", "das",
    "el", "la", "los", "las", "y",
    "van", "von", "der", "den", "ten", "bin", "al",
    "st", "st.", "san", "santa",
})

#: Canonical team aliases: Fantacalcio listone side → FotMob side, AFTER
#: ``normalise_team()`` has run. Keys must be the lowercased, accent-stripped
#: form returned by ``normalise_team``.
TEAM_ALIASES: dict[str, str] = {
    "inter": "internazionale",
}


# ── Normalisation helpers ───────────────────────────────────────────────────

_SUFFIX_STRIP_RE = re.compile(
    r"\b(jr|sr|ii|iii|iv)\b\.?$", flags=re.IGNORECASE
)
_TEAM_SUFFIX_RE = re.compile(
    r"\b(fc|ss|asd|ac|us|as|calcio|football)\b\.?",
    flags=re.IGNORECASE,
)
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def _strip_accents(s: str) -> str:
    """Lower-case, strip accents, drop punctuation. Used for fuzzy join keys."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def normalise_player_name(name: str) -> str:
    """Normalise player name for matching.

    Examples:
        "Dybala  Paulo"     → "dybala paulo"
        "M'Bala Nzola"      → "mbala nzola"
        "Pau López"         → "pau lopez"
    """
    n = _strip_accents(str(name))
    n = _NONALNUM_RE.sub(" ", n)
    n = _SUFFIX_STRIP_RE.sub("", n)
    return " ".join(n.split())


def normalise_team(team: str) -> str:
    """Normalise team name. Drops common suffixes (FC, Calcio, etc.) and
    shortens known variants (``AS Roma`` → ``roma``)."""
    n = _strip_accents(str(team))
    n = _TEAM_SUFFIX_RE.sub(" ", n)
    n = _NONALNUM_RE.sub(" ", n)
    return " ".join(n.split())


def apply_team_alias(team_norm: str) -> str:
    """Map known Fantacalcio team variants to their FotMob canonical form.

    Run AFTER :func:`normalise_team` so the lookup is on folded keys.

    >>> apply_team_alias("inter")
    'internazionale'
    >>> apply_team_alias("atalanta")
    'atalanta'
    """
    return TEAM_ALIASES.get(team_norm, team_norm)


def _strip_trailing_initial(name_norm: str) -> str:
    """Drop a trailing initial / abbreviation from a folded name.

    Fantacalcio listoni often encode ``"Surname X."`` (e.g. ``"Martinez L."``)
    or ``"Surname Jo."`` (``"Martinez Jo."``). After accent strip and case
    fold the trailing part becomes a short standalone token that is *not*
    part of the surname:

    * ``"martinez l"``  → ``"martinez"``  (single-letter initial)
    * ``"martinez jo"`` → ``"martinez"``  (2-letter abbreviation)
    * ``"pessina mas"`` → ``"pessina mas"`` (3+ chars: probably a real
      second surname, keep it)

    >>> _strip_trailing_initial("martinez l")
    'martinez'
    >>> _strip_trailing_initial("martinez jo")
    'martinez'
    >>> _strip_trailing_initial("esposito se")
    'esposito'
    """
    parts = name_norm.split()
    if len(parts) >= 2 and 1 <= len(parts[-1]) <= 2:
        return " ".join(parts[:-1])
    return name_norm


def last_name_token(name_norm: str) -> str:
    """Extract the *surname* token (or multi-word surname) for matching.

    The Fantacalcio listone almost always stores a surname-only form
    (``"Benedyczak"``, ``"Martinez L."``, ``"De Ketelaere"``), while FotMob
    stores "First Last" (``"Adrian Benedyczak"``). Comparing the *surnames*
    on both sides is the only way to bridge the two formats.

    Examples::

        >>> last_name_token("adrian benedyczak")
        'benedyczak'
        >>> last_name_token("lautaro martinez")
        'martinez'
        >>> last_name_token("charles de ketelaere")
        'de ketelaere'
        >>> last_name_token("randal kolo muani")
        'kolo muani'
        >>> last_name_token("martinez l")
        'martinez'
        >>> last_name_token("joao felix")
        'felix'
    """
    stripped = _strip_trailing_initial(name_norm)
    tokens = stripped.split()
    if not tokens:
        return ""
    if len(tokens) == 1 and len(tokens[0]) == 1:
        return ""
    if len(tokens) >= 2 and len(tokens[-1]) == 1:
        tokens = tokens[:-1]
        if not tokens:
            return ""
    # Compound surname: a prefix particle ("de", "di", …) belongs to it.
    # Handles 3-token input ("charles de ketelaere" → "de ketelaere")
    # AND 2-token input where the first token is the prefix
    # ("di gregorio" → "di gregorio", "de gea" → "de gea").
    if len(tokens) >= 2 and tokens[-2] in _COMPOUND_PREFIXES:
        return " ".join(tokens[-2:])
    if len(tokens) >= 2 and tokens[-1] in _COMPOUND_PREFIXES:
        return tokens[-1]
    return tokens[-1]


# ── Excel loading ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeasonFile:
    """A single xlsx file with its inferred season_start."""
    path: Path
    season_start: int


def discover_season_files(quotazioni_dir: Path) -> list[SeasonFile]:
    """Find all ``Quotazioni_Fantacalcio_Stagione_YYYY_YY.xlsx`` files
    and infer ``season_start`` from the filename.
    """
    pattern = re.compile(r"Quotazioni_Fantacalcio_Stagione_(\d{4})_(\d{2})\.xlsx$")
    files: list[SeasonFile] = []
    for path in sorted(quotazioni_dir.glob("Quotazioni_Fantacalcio_Stagione_*.xlsx")):
        m = pattern.search(path.name)
        if not m:
            log.warning("Skipping %s — does not match naming pattern.", path.name)
            continue
        season_start = int(m.group(1))
        files.append(SeasonFile(path=path, season_start=season_start))
    if not files:
        raise FileNotFoundError(
            f"No quotation files found in {quotazioni_dir}. "
            "Expected pattern: Quotazioni_Fantacalcio_Stagione_YYYY_YY.xlsx"
        )
    return files


def load_quotation_dataframe(file: SeasonFile) -> pd.DataFrame:
    """Load a single xlsx and return a clean DataFrame.

    The workbook's 'Tutti' sheet has the header on row 2 (index 1) and
    uses an unnamed first column (Italian: "Id") which we drop.
    """
    df = pd.read_excel(file.path, sheet_name=SHEET_NAME, header=HEADER_ROW)
    # The first column is unnamed in the source. Rename based on position.
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "fantacalcio_id"})
    df = df.iloc[:, :len(QUOTATION_COLUMNS)]
    df.columns = QUOTATION_COLUMNS
    df["season_start"] = file.season_start
    df["role"] = df["r"].map(ROLE_MAP)
    if df["role"].isna().any():
        bad = df.loc[df["role"].isna(), "r"].value_counts().to_dict()
        raise ValueError(f"Unknown role codes in {file.path.name}: {bad}")
    df["name_norm"] = df["name"].map(normalise_player_name)
    df["team_norm"] = df["team"].map(normalise_team).map(apply_team_alias)
    df["last_name_norm"] = df["name_norm"].map(last_name_token)
    # Ensure integer types for IDs and counts.
    int_cols = [
        "fantacalcio_id", "qt_a", "qt_i", "diff_val",
        "qt_a_m", "qt_i_m", "diff_val_m", "fvm", "fvm_m",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df


# ── Mapping: Fantacalcio ID → FotMob ID ─────────────────────────────────────

@dataclass
class MapResult:
    fantacalcio_id: int
    season_start: int
    player_fotmob_id: Optional[int]
    name_fantacalcio: str
    name_fotmob: Optional[str]
    team_fantacalcio: str
    team_fotmob: Optional[str]
    canonical_role: str
    match_method: str
    confidence: float


def _load_fotmob_reference(
    engine: sa.Engine, league_name: Optional[str]
) -> pd.DataFrame:
    """Load the canonical FotMob player index used as the mapping target.

    Joins ``player_profiles`` (canonical_role) with the most-recent season
    each player appeared in, taken from ``player_season_stats`` (their
    ``team_name`` is the canonical FotMob team string).
    """
    where = ""
    if league_name:
        escaped = league_name.replace("'", "''")
        where = f"WHERE l.name ILIKE '%{escaped}%'"

    sql = f"""
        WITH latest_season AS (
            SELECT pss.player_fotmob_id,
                   pss.player_name,
                   pss.team_name,
                   s.season_start,
                   ROW_NUMBER() OVER (
                       PARTITION BY pss.player_fotmob_id
                       ORDER BY s.season_start DESC
                   ) AS rn
            FROM player_season_stats pss
            JOIN seasons s ON s.id = pss.season_id
            JOIN leagues l ON l.id = s.league_id
            {where}
        )
        SELECT
            pp.player_fotmob_id,
            pp.player_name,
            pp.canonical_role,
            ls.team_name       AS team_fotmob,
            ls.season_start    AS latest_season_start
        FROM player_profiles pp
        JOIN latest_season ls ON ls.player_fotmob_id = pp.player_fotmob_id
        WHERE ls.rn = 1
    """
    df = pd.read_sql(sa.text(sql), engine)
    df["name_norm"] = df["player_name"].map(normalise_player_name)
    df["team_norm"] = df["team_fotmob"].map(normalise_team).map(apply_team_alias)
    df["last_name_norm"] = df["name_norm"].map(last_name_token)
    return df


def _exact_match(
    q: pd.DataFrame, ref: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matched_q, unmatched_q).

    Match key is (last_name_norm, team_norm, canonical_role). Switching from
    the full name to the surname is what makes the join work: the
    Fantacalcio listone encodes surnames only (``"Benedyczak"``,
    ``"Martinez L."``) while FotMob has full names (``"Adrian Benedyczak"``).
    """
    merged = q.merge(
        ref,
        on=["last_name_norm", "team_norm", "canonical_role"],
        how="left",
        suffixes=("", "_ref"),
    )
    matched = merged[merged["player_fotmob_id"].notna()].copy()
    matched["match_method"] = "exact_name_team"
    matched["confidence"] = 1.0
    unmatched = merged[merged["player_fotmob_id"].isna()][
        [
            "fantacalcio_id", "season_start", "name", "team", "name_norm",
            "last_name_norm", "team_norm", "canonical_role",
        ]
    ].copy()
    return matched, unmatched


def _exact_match_relaxed_role(
    q: pd.DataFrame, ref: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Like :func:`_exact_match` but ignores ``canonical_role``.

    This is the safety net for the common case where a player is classified
    differently across data sources: e.g. a winger (FWD) quoted as MID in
    the Fantacalcio listone, or a centre-back (DEF) that FotMob re-tagged
    as MID. When a unique FotMob player with the same surname and team
    exists in *any* role, accept it.

    Ambiguity rule: if the (surname, team) pair is non-unique in the FotMob
    reference, fall back to letting the caller keep the row unmatched so
    the operator can disambiguate manually.
    """
    if q.empty or ref.empty:
        return q.iloc[0:0].copy(), q
    merged = q.merge(
        ref,
        on=["last_name_norm", "team_norm"],
        how="left",
        suffixes=("", "_ref"),
    )
    # Count duplicates by the (fantacalcio_id, season_start) join key.
    counts = (
        merged.groupby(["fantacalcio_id", "season_start"])
        ["player_fotmob_id"].transform("count")
    )
    matched = merged[
        merged["player_fotmob_id"].notna() & (counts == 1)
    ].copy()
    matched["match_method"] = "exact_relaxed_role"
    matched["confidence"] = 0.95
    unmatched = merged[
        merged["player_fotmob_id"].isna() | (counts > 1)
    ][
        [
            "fantacalcio_id", "season_start", "name", "team", "name_norm",
            "last_name_norm", "team_norm", "canonical_role",
        ]
    ].copy()
    return matched, unmatched


def _fuzzy_match_one(
    last_name_norm: str,
    team_norm: str,
    canonical_role: str,
    candidates: pd.DataFrame,
) -> Optional[tuple[int, str, str, float]]:
    """Find the best fuzzy match for one (surname, team, role) tuple.

    The query side has already been reduced to a *surname* token, so the
    candidate side is filtered to the same role and then scored with
    :class:`difflib.SequenceMatcher` on the surname strings alone. A
    same-team tie-breaker is applied when multiple candidates share the
    top score.

    Returns ``(player_fotmob_id, name_fotmob, team_fotmob, score)`` or
    ``None`` if no candidate clears :data:`FUZZY_MATCH_THRESHOLD`.
    """
    if candidates.empty or not last_name_norm:
        return None

    pool = candidates
    if canonical_role:
        pool = pool[pool["canonical_role"] == canonical_role]
    if pool.empty:
        # Role filter is too strict (e.g. role changed across seasons, or
        # the FotMob side is missing the role). Fall back to the full pool.
        pool = candidates
    if pool.empty:
        return None

    if FUZZY_CANDIDATE_POOL > 0:
        pool = pool.head(FUZZY_CANDIDATE_POOL)

    best_id: Optional[int] = None
    best_name: str = ""
    best_team: str = ""
    best_score: float = 0.0
    for _, cand in pool.iterrows():
        cand_surname = cand.get("last_name_norm") or ""
        if not cand_surname:
            continue
        score = SequenceMatcher(None, last_name_norm, cand_surname).ratio()
        if score < FUZZY_MATCH_THRESHOLD:
            continue
        # Tie-breaker: same team wins. If still tied, prefer the closer
        # surname length to avoid "Benedyczak" → "Benedetti" collisions.
        cand_team = str(cand.get("team_norm") or "")
        same_team = 1 if cand_team and cand_team == team_norm else 0
        cand_score = score + 0.01 * same_team
        if cand_score > best_score or (
            cand_score == best_score
            and abs(len(cand_surname) - len(last_name_norm))
            < abs(len(best_name) - len(last_name_norm))
        ):
            best_score = cand_score
            best_id = int(cand["player_fotmob_id"])
            best_name = str(cand["player_name"])
            best_team = (
                str(cand["team_fotmob"])
                if pd.notna(cand.get("team_fotmob"))
                else ""
            )

    if best_id is None:
        return None
    # Strip the tie-breaker bonus before reporting the score.
    return best_id, best_name, best_team, float(best_score)


def build_player_id_map(
    quotazioni: pd.DataFrame, engine: sa.Engine, league_name: Optional[str]
) -> pd.DataFrame:
    """Build the fantacalcio_id → player_fotmob_id mapping for all rows."""
    ref = _load_fotmob_reference(engine, league_name=league_name)
    if ref.empty:
        log.warning(
            "No FotMob reference found. All Fantacalcio rows will be unmatched."
        )
    log.info(
        "  FotMob reference: %d players, %d with a canonical role.",
        len(ref), int(ref["canonical_role"].notna().sum()),
    )

    # ── Pass 1: exact match on (last_name, team, role) ──────────────────
    log.info("Pass 1: exact match on (surname, team, role) …")
    quotazioni_with_role = quotazioni.rename(columns={"role": "canonical_role"})
    matched, unmatched = _exact_match(quotazioni_with_role, ref)
    log.info("  matched: %d, unmatched: %d", len(matched), len(unmatched))

    results: list[dict] = []
    for _, row in matched.iterrows():
        results.append({
            "fantacalcio_id": int(row["fantacalcio_id"]),
            "season_start": int(row["season_start"]),
            "player_fotmob_id": int(row["player_fotmob_id"]),
            "name_fantacalcio": row["name"],
            "name_fotmob": row["player_name"],
            "team_fantacalcio": row["team"],
            "team_fotmob": row["team_fotmob"],
            "canonical_role": row["canonical_role"],
            "match_method": "exact_name_team",
            "confidence": 1.0,
        })

    # ── Pass 1b: exact match relaxed on (surname, team), role ignored ──
    # Recovers the common case where the player is classified differently
    # across data sources (e.g. a FWD in FotMob quoted as MID here).
    log.info("Pass 1b: exact match relaxed (surname, team), role ignored …")
    relaxed_matched, unmatched = _exact_match_relaxed_role(unmatched, ref)
    log.info(
        "  matched: %d, still unmatched: %d",
        len(relaxed_matched), len(unmatched),
    )
    for _, row in relaxed_matched.iterrows():
        results.append({
            "fantacalcio_id": int(row["fantacalcio_id"]),
            "season_start": int(row["season_start"]),
            "player_fotmob_id": int(row["player_fotmob_id"]),
            "name_fantacalcio": row["name"],
            "name_fotmob": row["player_name"],
            "team_fantacalcio": row["team"],
            "team_fotmob": row["team_fotmob"],
            "canonical_role": row["canonical_role"],
            "match_method": "exact_relaxed_role",
            "confidence": 0.95,
        })

    # ── Pass 2: fuzzy match on (surname, role) ───────────────────────────
    log.info("Pass 2: fuzzy match on (surname, role) …")
    fuzzy_hits = 0
    still_unmatched: list[dict] = []
    for _, row in unmatched.iterrows():
        best = _fuzzy_match_one(
            last_name_norm=row["last_name_norm"],
            team_norm=row["team_norm"],
            canonical_role=row["canonical_role"],
            candidates=ref,
        )
        if best is None:
            still_unmatched.append({
                "fantacalcio_id": int(row["fantacalcio_id"]),
                "name": row["name"],
                "team": row["team"],
                "canonical_role": row["canonical_role"],
            })
            continue
        fotmob_id, name_fotmob, team_fotmob, score = best
        results.append({
            "fantacalcio_id": int(row["fantacalcio_id"]),
            "season_start": int(row["season_start"]),
            "player_fotmob_id": fotmob_id,
            "name_fantacalcio": row["name"],
            "name_fotmob": name_fotmob,
            "team_fantacalcio": row["team"],
            "team_fotmob": team_fotmob,
            "canonical_role": row["canonical_role"],
            "match_method": "fuzzy_name",
            "confidence": round(score, 3),
        })
        fuzzy_hits += 1
    log.info("  fuzzy hits: %d, still unmatched: %d",
             fuzzy_hits, len(unmatched) - fuzzy_hits)

    # Surface a handful of the hardest cases so the operator can audit
    # the matching without grepping the log. Capped at 15 per season.
    if still_unmatched:
        sample = still_unmatched[:15]
        log.info("  sample of still-unmatched rows (up to 15):")
        for case in sample:
            log.info(
                "    [%s @ %s] %r  (surname=%r)",
                case["canonical_role"], case["team"], case["name"],
                last_name_token(normalise_player_name(case["name"])),
            )

    # ── Pass 3: record unmatched rows (operator will resolve manually) ──
    matched_keys = {
        (r["fantacalcio_id"], r["season_start"]) for r in results
    }
    for case in still_unmatched:
        key = (case["fantacalcio_id"], int(unmatched["season_start"].iloc[0]))
        if key in matched_keys:
            continue
        # Re-derive the team_fantacalcio for the missing row.
        results.append({
            "fantacalcio_id": key[0],
            "season_start": key[1],
            "player_fotmob_id": None,
            "name_fantacalcio": case["name"],
            "name_fotmob": None,
            "team_fantacalcio": case["team"],
            "team_fotmob": None,
            "canonical_role": case["canonical_role"],
            "match_method": "unmatched",
            "confidence": 0.0,
        })

    return pd.DataFrame(results)


# ── Persistence ──────────────────────────────────────────────────────────────

_UPSERT_QUOTATIONS_SQL = sa.text("""
    INSERT INTO player_quotations (
        fantacalcio_id, season_start, role, team, player_name,
        qt_a, qt_i, diff_val, qt_a_m, qt_i_m, diff_val_m,
        fvm, fvm_m, source
    )
    VALUES (
        :fantacalcio_id, :season_start, :role, :team, :player_name,
        :qt_a, :qt_i, :diff_val, :qt_a_m, :qt_i_m, :diff_val_m,
        :fvm, :fvm_m, :source
    )
    ON CONFLICT (fantacalcio_id, season_start) DO UPDATE SET
        role          = EXCLUDED.role,
        team          = EXCLUDED.team,
        player_name   = EXCLUDED.player_name,
        qt_a          = EXCLUDED.qt_a,
        qt_i          = EXCLUDED.qt_i,
        diff_val      = EXCLUDED.diff_val,
        qt_a_m        = EXCLUDED.qt_a_m,
        qt_i_m        = EXCLUDED.qt_i_m,
        diff_val_m    = EXCLUDED.diff_val_m,
        fvm           = EXCLUDED.fvm,
        fvm_m         = EXCLUDED.fvm_m,
        source        = EXCLUDED.source
""")

_UPSERT_MAP_SQL = sa.text("""
    INSERT INTO player_id_map (
        fantacalcio_id, season_start, player_fotmob_id,
        name_fantacalcio, name_fotmob,
        team_fantacalcio, team_fotmob,
        canonical_role, match_method, confidence
    )
    VALUES (
        :fantacalcio_id, :season_start, :player_fotmob_id,
        :name_fantacalcio, :name_fotmob,
        :team_fantacalcio, :team_fotmob,
        :canonical_role, :match_method, :confidence
    )
    ON CONFLICT (fantacalcio_id, season_start) DO UPDATE SET
        player_fotmob_id = EXCLUDED.player_fotmob_id,
        name_fotmob      = EXCLUDED.name_fotmob,
        team_fotmob      = EXCLUDED.team_fotmob,
        match_method     = EXCLUDED.match_method,
        confidence       = EXCLUDED.confidence,
        updated_at       = NOW()
""")


def persist_quotations(
    quotazioni: pd.DataFrame, engine: sa.Engine, source: str
) -> int:
    """Upsert all quotation rows. Returns the number of rows persisted."""
    payload = quotazioni[[
        "fantacalcio_id", "season_start", "role", "team", "name",
        "qt_a", "qt_i", "diff_val",
        "qt_a_m", "qt_i_m", "diff_val_m",
        "fvm", "fvm_m",
    ]].rename(columns={"name": "player_name"})

    # Replace pandas Int64 NA with None so psycopg binds NULL.
    payload = payload.astype(object).where(pd.notnull(payload), None)
    payload["source"] = source

    rows = payload.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(_UPSERT_QUOTATIONS_SQL, rows)
    return len(rows)


def persist_player_id_map(mapping: pd.DataFrame, engine: sa.Engine) -> int:
    """Upsert the id map. Returns the number of rows persisted."""
    rows = mapping.astype(object).where(pd.notnull(mapping), None).to_dict(
        orient="records"
    )
    with engine.begin() as conn:
        conn.execute(_UPSERT_MAP_SQL, rows)
    return len(rows)


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ml.data.import_quotations",
        description="Import Fantacalcio listoni (auction values) into PostgreSQL.",
    )
    p.add_argument(
        "--quotazioni-dir",
        type=Path,
        required=True,
        help="Directory containing Quotazioni_Fantacalcio_Stagione_*.xlsx files.",
    )
    p.add_argument(
        "--league",
        default=None,
        help="Optional league filter for the FotMob reference (e.g. 'Serie A').",
    )
    p.add_argument(
        "--source",
        default="listone_fantagazzetta",
        help="Provenance tag stored in player_quotations.source.",
    )
    p.add_argument(
        "--db-url",
        default=None,
        help="Override database URL. Defaults to ML_DATABASE_URL or API_DATABASE_URL.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    db_url = (
        args.db_url
        or __import__("os").environ.get("ML_DATABASE_URL")
        or __import__("os").environ.get("API_DATABASE_URL")
    )
    if not db_url:
        log.error(
            "Database URL not set. Pass --db-url or export ML_DATABASE_URL."
        )
        return 2

    engine = sa.create_engine(db_url, pool_pre_ping=True)
    files = discover_season_files(args.quotazioni_dir)
    log.info("Discovered %d season file(s) in %s",
             len(files), args.quotazioni_dir)

    for sf in files:
        log.info("=" * 60)
        log.info("Processing %s (season_start=%d)", sf.path.name, sf.season_start)
        df = load_quotation_dataframe(sf)
        log.info("  %d rows, role distribution: %s",
                 len(df), df["role"].value_counts().to_dict())

        # Persist quotations
        n_q = persist_quotations(df, engine=engine, source=args.source)
        log.info("  Persisted %d quotation rows.", n_q)

        # Build & persist the id map
        id_map = build_player_id_map(df, engine=engine, league_name=args.league)
        log.info("  id map distribution: %s",
                 id_map["match_method"].value_counts().to_dict())
        n_m = persist_player_id_map(id_map, engine=engine)
        log.info("  Persisted %d id map rows.", n_m)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Importer for Fantacalcio listoni (auction valuation spreadsheets).

Reads ``Quotazioni_Fantacalcio_Stagione_<YYYY_YY>.xlsx`` files from a
directory and persists them into two PostgreSQL tables:

* ``player_quotations``  — raw Qt.A / Qt.I / FVM values
* ``player_id_map``      — Fantacalcio-id ↔ player_fotmob_id bridge

The FotMob-side resolution happens in four passes:

1. **Deterministic exact match** on (normalised name, normalised team,
   canonical_role). Uses ``player_profiles`` joined with
   ``player_season_stats`` (only one season of stats per fantacalcio row
   is needed, so the latest season each player appears in is used as the
   reference).

2. **Exact relaxed role** — same as pass 1 but ignoring ``canonical_role``.
   Recovers players classified differently by Fantacalcio and FotMob.

3. **Fuzzy fallback** (optional) using ``difflib.SequenceMatcher`` for
   players that didn't match exactly.

4. **FotMob suggest API** — for players still unmatched after the fuzzy
   pass, calls FotMob's public ``/api/data/search/suggest`` endpoint.
   If exactly **one** result is returned for the full player name it is
   accepted automatically. Multiple candidates are left for manual
   resolution via the ID Mapping UI.

After automatic matching, **manual overrides** (``--overrides``) are applied
from a CSV file.  See :mod:`ml.data.match_override` for the CSV format and
the ``match_override`` CLI tool to export unresolved cases.

Module CLI::

    python -m ml.data.import_quotations \\
        --quotazioni-dir ../quotazioni \\
        --source listone_fantagazzetta \\
        [--overrides match_overrides.csv]
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

from ml.mantra.roles import calcola_ruolo_primario, normalizza_rm

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
    # Parse MANTRA roles from rm column
    df["ruoli_mantra"] = df["rm"].apply(_parse_mantra_rm)
    df["ruolo_primario"] = df["ruoli_mantra"].apply(_compute_ruolo_primario)
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

    Strategy (preference order):
    1. **Season-scoped roles** — use ``player_season_roles`` joined with
       the player's team per season from ``player_season_stats``.  This
       ensures the role used for matching matches what the player *actually
       had* that season, not their current role years later.
    2. **Fallback** — ``player_profiles`` (current role only).  Used when
       migration 003 has not been applied or ``player_season_roles`` is
       empty.

    The output always contains one row per ``(player_fotmob_id, season_start)``
    so the caller can join on either ``last_name_norm`` alone (role-agnostic)
    or on ``(last_name_norm, team_norm, canonical_role, season_start)``.
    """
    where = ""
    if league_name:
        escaped = league_name.replace("'", "''")
        where = f"AND l.name ILIKE '%{escaped}%'"

    # ── Attempt 1: season-scoped roles (preferred) ───────────────────────
    sql_season_roles = f"""
        SELECT
            psr.player_fotmob_id,
            pss.player_name,
            pss.team_name          AS team_fotmob,
            pss.season_start,
            psr.canonical_role
        FROM player_season_roles psr
        JOIN player_season_stats pss
          ON pss.player_fotmob_id = psr.player_fotmob_id
         AND pss.season_id = (
             -- Always Serie A: this function resolves the Fantacalcio Serie A
             -- listino, regardless of the caller's league_name filter (used
             -- only for the broader candidate pool below). Without this,
             -- ingesting a second league sharing this season_start (e.g.
             -- Premier League) would let Postgres arbitrarily pick either
             -- league's season_id here, breaking matching non-deterministically
             -- for anyone with rows in both.
             SELECT s2.id
             FROM seasons s2
             JOIN leagues l2 ON l2.id = s2.league_id
             WHERE s2.season_start = psr.season_start
               AND l2.name = 'Serie A'
             LIMIT 1
         )
        JOIN seasons s ON s.id = pss.season_id
        JOIN leagues l ON l.id = s.league_id
        WHERE psr.canonical_role IS NOT NULL
        {where}
        GROUP BY psr.player_fotmob_id, pss.player_name, pss.team_name,
                 pss.season_start, psr.canonical_role
    """
    try:
        df = pd.read_sql(sa.text(sql_season_roles), engine)
        if not df.empty:
            log.info(
                "FotMob reference loaded from player_season_roles: "
                "%d rows across %d seasons.",
                len(df), df["season_start"].nunique(),
            )
            df["name_norm"] = df["player_name"].map(normalise_player_name)
            df["team_norm"] = df["team_fotmob"].map(normalise_team).map(apply_team_alias)
            df["last_name_norm"] = df["name_norm"].map(last_name_token)
            return df
        log.warning(
            "player_season_roles is empty — falling back to player_profiles. "
            "Re-run the scraper with --roles to populate it."
        )
    except Exception:  # noqa: BLE001
        log.warning(
            "player_season_roles not available — falling back to player_profiles. "
            "Apply migration 003_add_player_season_roles.sql and re-run "
            "the scraper with --roles.",
        )

    # ── Fallback: current role only (player_profiles) ────────────────────
    sql_profiles = f"""
        WITH latest_season AS (
            SELECT pss.player_fotmob_id,
                   pss.player_name,
                   pss.team_name,
                   s.season_start,
                   ROW_NUMBER() OVER (
                       PARTITION BY pss.player_fotmob_id
                       -- Serie A wins ties on season_start against any other
                       -- ingested league, rather than leaving it to Postgres's
                       -- arbitrary row-arrival order (see the primary query's
                       -- season_id subquery above for the same rationale).
                       ORDER BY (l.name = 'Serie A') DESC, s.season_start DESC
                   ) AS rn
            FROM player_season_stats pss
            JOIN seasons s ON s.id = pss.season_id
            JOIN leagues l ON l.id = s.league_id
            {where.replace('AND', 'WHERE', 1) if where else ''}
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
    df = pd.read_sql(sa.text(sql_profiles), engine)
    if df.empty:
        log.warning(
            "player_profiles is also empty — FotMob reference has 0 rows. "
            "Run the scraper to populate player_season_stats and player_profiles."
        )
    else:
        log.info(
            "FotMob reference loaded from player_profiles (fallback): %d rows.",
            len(df),
        )
    df["name_norm"] = df["player_name"].map(normalise_player_name)
    df["team_norm"] = df["team_fotmob"].map(normalise_team).map(apply_team_alias)
    df["last_name_norm"] = df["name_norm"].map(last_name_token)
    return df


def _exact_match(
    q: pd.DataFrame, ref: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matched_q, unmatched_q).

    Two-phase match:
    1. **Strict** — join on ``(last_name_norm, team_norm, canonical_role,
       season_start)``.  Only possible when *ref* contains ``season_start``
       (season-scoped roles).  Recovers the per-season role the player
       actually had.
    2. **Relaxed** (fallback) — join on ``(last_name_norm, team_norm,
       canonical_role)`` without season.  When *ref* is from player_profiles
       (current role only) this is the only option.

    Switching from the full name to the surname is what makes the join work:
    the Fantacalcio listone encodes surnames only (``"Benedyczak"``,
    ``"Martinez L."``) while FotMob has full names (``"Adrian Benedyczak"``).
    """
    ref_has_season = "season_start" in ref.columns

    # Phase 1: strict per-season match (if ref has season_start)
    if ref_has_season:
        merged = q.merge(
            ref,
            on=["last_name_norm", "team_norm", "canonical_role", "season_start"],
            how="left",
            suffixes=("", "_ref"),
        )
        matched = merged[merged["player_fotmob_id"].notna()].copy()
        matched["match_method"] = "exact_name_team_role_season"
        matched["confidence"] = 1.0
        leftover = merged[merged["player_fotmob_id"].isna()][
            [
                "fantacalcio_id", "season_start", "name", "team", "name_norm",
                "last_name_norm", "team_norm", "canonical_role",
            ]
        ].copy()
        log.info(
            "exact_match[season]: %d matched, %d unmatched (strict key).",
            len(matched), len(leftover),
        )
        # Recurse on unmatched with the season-agnostic fallback
        if not leftover.empty:
            fallback_matched, leftover = _exact_match_no_season(leftover, ref)
            matched = pd.concat([matched, fallback_matched], ignore_index=True)
        return matched, leftover

    # Phase 2: no season_start in ref — standard match
    return _exact_match_no_season(q, ref)


def _exact_match_no_season(
    q: pd.DataFrame, ref: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match on (last_name_norm, team_norm, canonical_role) ignoring season.

    Used when the reference only has current roles (player_profiles fallback).
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

    When *ref* has ``season_start`` (season-scoped roles), the join also
    includes ``season_start`` so a transfer mid-season does not alias two
    different players on the same team.

    Ambiguity rule: if the (surname, team) pair is non-unique in the FotMob
    reference, fall back to letting the caller keep the row unmatched so
    the operator can disambiguate manually.
    """
    if q.empty or ref.empty:
        return q.iloc[0:0].copy(), q

    ref_has_season = "season_start" in ref.columns
    join_keys = ["last_name_norm", "team_norm"]
    if ref_has_season:
        join_keys = ["last_name_norm", "team_norm", "season_start"]

    merged = q.merge(
        ref,
        on=join_keys,
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


def _load_manual_resolutions(engine: sa.Engine) -> pd.DataFrame:
    """Load all rows from the ``manual_resolutions`` history table.

    Returns an empty DataFrame if the table does not exist or has no rows.
    """
    try:
        df = pd.read_sql(
            sa.text(
                "SELECT fantacalcio_id, player_fotmob_id, season_start, "
                "name_fantacalcio, team_fantacalcio, canonical_role, "
                "name_fotmob, team_fotmob "
                "FROM manual_resolutions "
                "ORDER BY season_start DESC, created_at DESC"
            ),
            engine,
        )
        log.info("  loaded %d historical manual resolutions", len(df))
        return df
    except Exception:  # noqa: BLE001
        log.warning(
            "manual_resolutions table not available — skipping Pass 0. "
            "Apply migration 013_add_manual_resolutions.sql."
        )
        return pd.DataFrame()


# ── FotMob suggest API helper ────────────────────────────────────────────────


def _fotmob_suggest_api(term: str, hits: int = 5) -> list[dict]:
    """Call FotMob's public suggest API and return player suggestions.

    Returns a list of dicts with keys ``id``, ``name``, ``team_id``,
    ``team_name``, ``score``. Returns an empty list on any error.
    """
    import json
    import urllib.request
    import urllib.parse

    url = (
        "https://www.fotmob.com/api/data/search/suggest"
        f"?hits={hits}&lang=it%2Cen%2Cfr&term={urllib.parse.quote(term)}"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    seen_ids: set[int] = set()
    players: list[dict] = []
    for group in data:
        for s in group.get("suggestions", []):
            if s.get("type") != "player":
                continue
            pid = int(s["id"])
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            players.append({
                "id": pid,
                "name": s["name"],
                "team_id": s.get("teamId"),
                "team_name": s.get("teamName"),
                "score": s.get("score", 0),
            })
    return players


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

    results: list[dict] = []
    matched_keys: set[tuple[int, int]] = set()

    # ── Pass 0: historical manual resolutions (permanent operator overrides) ─
    log.info("Pass 0: applying historical manual resolutions …")
    historical = _load_manual_resolutions(engine)
    if not historical.empty:
        # Build lookup: latest resolution per fantacalcio_id
        historical = historical.sort_values(
            ["fantacalcio_id", "season_start", "player_fotmob_id"],
            ascending=[True, False, False],
        )
        # Deduplicate: keep the most recent resolution per fantacalcio_id
        latest_per_id = historical.drop_duplicates(
            subset="fantacalcio_id", keep="first"
        )

        # Join with quotazioni on fantacalcio_id
        merged = quotazioni.merge(
            latest_per_id[["fantacalcio_id", "player_fotmob_id",
                           "name_fotmob", "team_fotmob",
                           "canonical_role"]],
            on="fantacalcio_id",
            how="inner",
            suffixes=("", "_hist"),
        )
        # Use the canonical_role from the resolution if the quotation has none
        merged["canonical_role"] = merged["canonical_role"].fillna(
            merged.get("role")
        )

        for _, row in merged.iterrows():
            key = (int(row["fantacalcio_id"]), int(row["season_start"]))
            matched_keys.add(key)
            results.append({
                "fantacalcio_id": key[0],
                "season_start": key[1],
                "player_fotmob_id": int(row["player_fotmob_id"]),
                "name_fantacalcio": row["name"],
                "name_fotmob": row.get("name_fotmob"),
                "team_fantacalcio": row["team"],
                "team_fotmob": row.get("team_fotmob"),
                "canonical_role": row.get("canonical_role"),
                "match_method": "manual",
                "confidence": 1.0,
                "resolved_from_history": True,
            })
        log.info(
            "  historical matches applied: %d (from %d resolutions)",
            len(results), len(historical),
        )

    # ── Pass 0.5: propagate high-confidence mappings across seasons ──────
    # If the same fantacalcio_id was already matched in a previous season
    # (e.g. via Pass 0 manual resolutions or a previous season's run),
    # re-use that mapping for all later seasons without re-running matching.
    # Exclude fuzzy_name and unmatched since they're not reliable enough.
    remaining = quotazioni[
        ~quotazioni.apply(
            lambda r: (int(r["fantacalcio_id"]), int(r["season_start"])) in matched_keys,
            axis=1,
        )
    ].copy()

    log.info(
        "Pass 0.5: propagating high-confidence mappings across seasons …"
        " (%d already matched)", len(matched_keys),
    )
    propagated = 0
    still_remaining_mask = [True] * len(remaining)
    for idx, (_, row) in enumerate(remaining.iterrows()):
        fc_id = int(row["fantacalcio_id"])
        existing = [
            r for r in results
            if r["fantacalcio_id"] == fc_id
            and r["match_method"] not in ("unmatched", "fuzzy_name")
        ]
        if not existing:
            continue
        best = max(existing, key=lambda r: r["confidence"])
        season_key = (fc_id, int(row["season_start"]))
        if season_key in matched_keys:
            continue
        matched_keys.add(season_key)
        still_remaining_mask[idx] = False
        results.append({
            "fantacalcio_id": fc_id,
            "season_start": int(row["season_start"]),
            "player_fotmob_id": best["player_fotmob_id"],
            "name_fantacalcio": row["name"],
            "name_fotmob": best["name_fotmob"],
            "team_fantacalcio": row["team"],
            "team_fotmob": best.get("team_fotmob"),
            "canonical_role": row["canonical_role"],
            "match_method": "propagated",
            "confidence": round(min(best["confidence"] * 0.95, 0.99), 3),
            "resolved_from_history": best.get("resolved_from_history", False),
        })
        propagated += 1
    log.info("  propagated: %d", propagated)
    remaining = remaining[still_remaining_mask].copy()

    # ── Pass 1: exact match on (last_name, team, role) ──────────────────
    log.info("Pass 1: exact match on (surname, team, role) … (%d remaining)", len(remaining))
    quotazioni_with_role = remaining.rename(columns={"role": "canonical_role"})
    matched, unmatched = _exact_match(quotazioni_with_role, ref)
    log.info("  matched: %d, unmatched: %d", len(matched), len(unmatched))

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
            "resolved_from_history": False,
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
            "resolved_from_history": False,
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
                "season_start": int(row["season_start"]),
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
            "resolved_from_history": False,
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

    # ── Pass 3: FotMob suggest API for still-unmatched players ──────────
    #   For players that didn't match via DB-based fuzzy, try the live
    #   FotMob suggest API. If exactly one result comes back for the full
    #   player name, accept it as a high-confidence match.
    #   When multiple candidates are returned, leave the row unmatched so
    #   the operator can decide via the ID mapping UI.
    log.info("Pass 3: FotMob suggest API for %d still-unmatched …", len(still_unmatched))
    suggest_hits = 0
    for case in still_unmatched:
        name = case["name"]
        try:
            candidates = _fotmob_suggest_api(name)
        except Exception:
            log.warning("  suggest API call failed for %r, skipping", name)
            continue

        if len(candidates) == 1:
            # Single unambiguous result — accept it automatically.
            c = candidates[0]
            key = (case["fantacalcio_id"], case["season_start"])
            if key in {(r["fantacalcio_id"], r["season_start"]) for r in results}:
                continue
            results.append({
                "fantacalcio_id": key[0],
                "season_start": key[1],
                "player_fotmob_id": c["id"],
                "name_fantacalcio": case["name"],
                "name_fotmob": c["name"],
                "team_fantacalcio": case["team"],
                "team_fotmob": c.get("team_name"),
                "canonical_role": case["canonical_role"],
                "match_method": "fotmob_suggest",
                "confidence": 0.85,
                "resolved_from_history": False,
            })
            suggest_hits += 1
        # else: multiple candidates → leave for manual resolution

    log.info("  suggest API hits: %d, still unmatched: %d",
             suggest_hits, len(still_unmatched) - suggest_hits)

    # ── Pass 4: record unmatched rows (operator will resolve manually) ──
    matched_keys = {
        (r["fantacalcio_id"], r["season_start"]) for r in results
    }
    for case in still_unmatched:
        key = (case["fantacalcio_id"], case["season_start"])
        if key in matched_keys:
            continue
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
            "resolved_from_history": False,
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
        canonical_role, match_method, confidence,
        resolved_from_history
    )
    VALUES (
        :fantacalcio_id, :season_start, :player_fotmob_id,
        :name_fantacalcio, :name_fotmob,
        :team_fantacalcio, :team_fotmob,
        :canonical_role, :match_method, :confidence,
        :resolved_from_history
    )
    ON CONFLICT (fantacalcio_id, season_start) DO UPDATE SET
        player_fotmob_id    = EXCLUDED.player_fotmob_id,
        name_fotmob         = EXCLUDED.name_fotmob,
        team_fotmob         = EXCLUDED.team_fotmob,
        match_method        = EXCLUDED.match_method,
        confidence          = EXCLUDED.confidence,
        resolved_from_history = EXCLUDED.resolved_from_history,
        updated_at          = NOW()
""")


# ── MANTRA helpers ─────────────────────────────────────────────────────────

def _parse_mantra_rm(raw: object) -> list[str]:
    """Parse a raw rm cell from the XLSX into a list of canonical MANTRA roles."""
    if not raw or pd.isna(raw):
        return []
    return normalizza_rm(str(raw))


def _compute_ruolo_primario(ruoli: list[str]) -> Optional[str]:
    """Compute the primary MANTRA role (most defensive) for a player."""
    return calcola_ruolo_primario(ruoli)


_UPSERT_MANTRA_ROLES_SQL = sa.text("""
    INSERT INTO player_mantra_roles (
        fantacalcio_id, season_start, ruolo_primario, ruoli_mantra
    )
    VALUES (
        :fantacalcio_id, :season_start, :ruolo_primario, :ruoli_mantra
    )
    ON CONFLICT (fantacalcio_id, season_start) DO UPDATE SET
        ruolo_primario = EXCLUDED.ruolo_primario,
        ruoli_mantra   = EXCLUDED.ruoli_mantra
""")


def persist_mantra_roles(quotazioni: pd.DataFrame, engine: sa.Engine) -> int:
    """Upsert MANTRA role rows. Returns the number of rows persisted."""
    # Filter rows that actually have parsed MANTRA roles
    has_roles = quotazioni["ruoli_mantra"].apply(lambda r: len(r) > 0)
    if not has_roles.any():
        log.warning("No MANTRA roles found in the quotation data (rm column empty?)")
        return 0

    payload = quotazioni.loc[has_roles, [
        "fantacalcio_id", "season_start", "ruolo_primario", "ruoli_mantra",
    ]].copy()

    # Convert list to PostgreSQL array literal
    payload["ruoli_mantra"] = payload["ruoli_mantra"].apply(
        lambda r: "{" + ",".join(r) + "}"
    )
    payload = payload.astype(object).where(pd.notnull(payload), None)
    rows = payload.to_dict(orient="records")

    with engine.begin() as conn:
        conn.execute(_UPSERT_MANTRA_ROLES_SQL, rows)
    log.info("  Persisted %d MANTRA role rows.", len(rows))
    return len(rows)


# ── Persistence ──────────────────────────────────────────────────────────────

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


def retry_unmatched(engine: sa.Engine, season_start: int) -> pd.DataFrame:
    """Second-chance resolution for rows left as match_method='unmatched'.

    Re-uses ``_fotmob_suggest_api`` with name variants (full name, surname
    only). Accepts a candidate only when the API returns exactly one hit
    (same conservative rule as Pass 3). Does **not** overwrite existing
    manual / high-confidence matches.

    Returns a DataFrame of newly resolved rows (empty if none).
    """
    query = sa.text("""
        SELECT fantacalcio_id, season_start, name_fantacalcio, team_fantacalcio,
               canonical_role, match_method
        FROM player_id_map
        WHERE season_start = :season_start
          AND match_method = 'unmatched'
          AND player_fotmob_id IS NULL
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {"season_start": season_start}).mappings().all()

    if not rows:
        log.info("retry_unmatched: no unmatched rows for season_start=%s", season_start)
        return pd.DataFrame()

    log.info(
        "retry_unmatched: retrying %d unmatched player(s) for season_start=%s",
        len(rows),
        season_start,
    )
    resolved: list[dict] = []
    for row in rows:
        name = (row["name_fantacalcio"] or "").strip()
        if not name:
            continue

        terms = [name]
        parts = name.split()
        if len(parts) >= 2:
            terms.append(parts[-1])

        chosen = None
        for term in terms:
            try:
                candidates = _fotmob_suggest_api(term)
            except Exception:
                log.warning("  retry suggest API failed for %r", term)
                continue
            if len(candidates) == 1:
                chosen = candidates[0]
                break

        if chosen is None:
            continue

        resolved.append({
            "fantacalcio_id": row["fantacalcio_id"],
            "season_start": row["season_start"],
            "player_fotmob_id": chosen["id"],
            "name_fantacalcio": name,
            "name_fotmob": chosen["name"],
            "team_fantacalcio": row["team_fantacalcio"],
            "team_fotmob": chosen.get("team_name"),
            "canonical_role": row["canonical_role"],
            "match_method": "fotmob_suggest_retry",
            "confidence": 0.85,
            "resolved_from_history": False,
        })

    if not resolved:
        log.info("retry_unmatched: no new resolutions")
        return pd.DataFrame()

    df = pd.DataFrame(resolved)
    n = persist_player_id_map(df, engine)
    log.info(
        "retry_unmatched: resolved %d player(s) via fotmob_suggest_retry",
        n,
    )
    return df


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
        "--overrides",
        type=Path,
        default=None,
        metavar="CSV",
        help="Path to a match_overrides CSV file for manual ID resolution. "
             "See ml.data.match_override for the format.",
    )
    p.add_argument(
        "--export-unresolved",
        type=Path,
        default=None,
        metavar="CSV",
        help="Export unmatched / low-confidence rows to this CSV after "
             "processing, so the operator can review and resolve them.",
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
    import os as _os

    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    db_url = (
        args.db_url
        or _os.environ.get("ML_DATABASE_URL")
        or _os.environ.get("API_DATABASE_URL")
    )
    if not db_url:
        log.error(
            "Database URL not set. Pass --db-url or export ML_DATABASE_URL."
        )
        return 2

    # Load manual overrides upfront (shared across all seasons)
    from .match_override import load_overrides_csv

    overrides = load_overrides_csv(args.overrides) if args.overrides else []
    if overrides:
        log.info("Loaded %d manual override(s) from %s.", len(overrides), args.overrides)

    engine = sa.create_engine(db_url, pool_pre_ping=True)
    files = discover_season_files(args.quotazioni_dir)
    log.info("Discovered %d season file(s) in %s",
             len(files), args.quotazioni_dir)

    all_id_maps: list[pd.DataFrame] = []

    for sf in files:
        log.info("=" * 60)
        log.info("Processing %s (season_start=%d)", sf.path.name, sf.season_start)
        df = load_quotation_dataframe(sf)
        log.info("  %d rows, role distribution: %s",
                 len(df), df["role"].value_counts().to_dict())

        # Persist quotations
        n_q = persist_quotations(df, engine=engine, source=args.source)
        log.info("  Persisted %d quotation rows.", n_q)

        # Persist MANTRA roles
        n_mr = persist_mantra_roles(df, engine=engine)
        log.info("  Persisted %d MANTRA role rows.", n_mr)

        # Build the id map
        id_map = build_player_id_map(df, engine=engine, league_name=args.league)

        # Apply manual overrides (if any) before persisting
        if overrides:
            from .match_override import apply_overrides_to_id_map
            id_map = apply_overrides_to_id_map(id_map, overrides)

        log.info("  id map distribution: %s",
                 id_map["match_method"].value_counts().to_dict())
        n_m = persist_player_id_map(id_map, engine=engine)
        log.info("  Persisted %d id map rows.", n_m)
        all_id_maps.append(id_map)

    # ── Export unresolved rows (optional) ─────────────────────────────────
    if args.export_unresolved and all_id_maps:
        from .match_override import export_unresolved
        combined_map = pd.concat(all_id_maps, ignore_index=True)
        n_exported = export_unresolved(combined_map, args.export_unresolved)
        log.info("Exported %d unresolved rows to %s.", n_exported, args.export_unresolved)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

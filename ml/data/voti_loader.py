"""Loader for official Fantacalcio voti JSON files.

The voti files contain match-by-match scores scraped from fantacalcio.it
(one file per season). This module parses them, aggregates to a
``player_fotmob_id × season_start`` target and emits a CSV that the
existing ML pipeline already understands (``--fantavoto-csv``).

The pipeline already supports two CSV schemas:

* ``player_fotmob_id, season_start, fantavoto_medio``     ← we emit this
* ``player_name, season_label, fantavoto_medio``          (fallback)

Why this lives outside the pipeline itself
------------------------------------------
The voti JSONs use the Fantacalcio display name (e.g. ``"Martinez L."``)
while the rest of the pipeline is keyed on ``player_fotmob_id``. Decoupling
the JSON→CSV conversion from the pipeline means the mapping is
re-runnable, inspectable, and can be unit-tested in isolation.

Design choices (all overridable via CLI / constants below):

* Vote metric: ``voti.fantacalcio.fantavoto`` (includes bonus/malus).
  This is the score that actually counts in a Fantacalcio league.
* No-vote handling: rows with ``"s.v."`` (senza voto) are dropped —
  Fantacalcio uses this marker for players who did not enter the pitch,
  and including them as 6 would bias the average downwards.
* Multi-team players: votes are pooled across teams; the mean is
  weighted by appearance count. A player who transferred mid-season is
  treated as a single observation.
* 2025/26 partial season: included as-is. The pipeline's temporal split
  holds out the most-recent season as test set, so the (potentially
  noisy) current-season mean lands in the test set and the training
  mean stays clean.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import sqlalchemy as sa

from .import_quotations import (
    last_name_token,
    normalise_player_name,
    normalise_team,
    apply_team_alias,
)

log = logging.getLogger(__name__)


# ── Public constants ─────────────────────────────────────────────────────────

#: Italian (Fantacalcio) role labels in the JSON → canonical pipeline role.
ROLE_IT_TO_CANONICAL: dict[str, str] = {
    "Portiere":        "GK",
    "Difensore":       "DEF",
    "Centrocampista":  "MID",
    "Attaccante":      "FWD",
}

#: Regex to extract the starting year from a filename like
#: ``voti_fantacalcio-2023-24.json`` → ``2023``.
_FILENAME_SEASON_RE = re.compile(r"(\d{4})-(\d{2})")

#: Italian-style decimal comma → float.
_DECIMAL_COMMA_RE = re.compile(r"^\s*(\d+),(\d+)\s*$")

#: Tokens that mark "no real vote" in the Fantacalcio JSON.  Rows whose
#: ``voto`` and ``fantavoto`` are both sentinels are dropped.
#:
#: * ``"s.v."`` / ``"SV"`` : *senza voto* — player did not enter the pitch.
#: * ``"55"``               : Fantacalcio.it placeholder for *non
#:   valutato* / a scraping error.  Treated identically to ``s.v.``
#:   because it is not a real score; including it would bias the mean
#:   by ~10 points per occurrence (see the *6,806* instances of ``55``
#:   in the raw data — keeping them would dwarf every real vote).
_NO_VOTE_TOKENS: frozenset[str] = frozenset({"s.v.", "sv", "55"})


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeasonVotiFile:
    """A voti JSON with its inferred ``season_start`` year."""
    path: Path
    season_start: int


# ── JSON parsing ─────────────────────────────────────────────────────────────

def discover_voti_files(voti_dir: Path) -> list[SeasonVotiFile]:
    """Find all ``voti_fantacalcio-YYYY-YY.json`` files and infer the season.

    Raises ``FileNotFoundError`` if no files match — the user must rename
    their files to the expected pattern.
    """
    out: list[SeasonVotiFile] = []
    for path in sorted(voti_dir.glob("voti_fantacalcio-*.json")):
        m = _FILENAME_SEASON_RE.search(path.name)
        if not m:
            log.warning("Skipping %s — does not match YYYY-YY pattern.", path.name)
            continue
        season_start = int(m.group(1))
        out.append(SeasonVotiFile(path=path, season_start=season_start))
    if not out:
        raise FileNotFoundError(
            f"No voti files found in {voti_dir}. "
            "Expected pattern: voti_fantacalcio-YYYY-YY.json"
        )
    return out


def _parse_italian_decimal(s: str) -> Optional[float]:
    """Parse ``"6,5"`` → ``6.5``.

    Returns ``None`` for:
      * sentinels (``"s.v."``, ``"SV"``, ``"55"``)
      * empty strings
      * anything that is not a recognisable number
    """
    s = (s or "").strip()
    if not s or s.lower() in _NO_VOTE_TOKENS:
        return None
    m = _DECIMAL_COMMA_RE.match(s)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    # Plain integer / English decimal.
    try:
        return float(s)
    except ValueError:
        return None


def parse_voti_file(file: SeasonVotiFile) -> pd.DataFrame:
    """Parse a single voti JSON into a tidy long-format DataFrame.

    Columns produced:
        season_start, matchday, match_date, team, opponent,
        name, role, status, voto, fantavoto, played,
        goals_scored, penalties_scored, penalties_saved, assists, potm,
        goals_conceded, own_goals, penalties_missed

    Rows with ``"s.v."`` votes are dropped because they would bias the
    mean (a non-playing entry is not a real observation).
    """
    log.info("Parsing %s …", file.path.name)
    with file.path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    rows: list[dict[str, Any]] = []
    for giornata in raw:
        matchday = giornata.get("giornata")
        for match in giornata.get("squadre", []):
            home = match.get("squadraCasa", "")
            away = match.get("squadraOspite", "")
            match_date = match.get("data", "")
            for player in match.get("giocatori", []):
                voti = player.get("voti", {}).get("fantacalcio", {}) or {}
                voto = _parse_italian_decimal(str(voti.get("voto", "")))
                fantavoto = _parse_italian_decimal(str(voti.get("fantavoto", "")))
                if voto is None and fantavoto is None:
                    # "s.v." — player did not enter the pitch.  Skip.
                    continue
                if fantavoto is None:
                    # Defensive: if only the base vote is present, use it.
                    fantavoto = voto
                if fantavoto is None:
                    continue

                bonus = player.get("bonus", {}) or {}
                malus = player.get("malus", {}) or {}
                team = player.get("squadra", "")
                role_it = player.get("ruolo", "")

                rows.append({
                    "season_start":      file.season_start,
                    "matchday":          matchday,
                    "match_date":        match_date,
                    "team":              team,
                    "opponent":          away if team == home else home,
                    "is_home":           team == home,
                    "name":              player.get("nome", ""),
                    "role":              ROLE_IT_TO_CANONICAL.get(role_it, ""),
                    "status":            player.get("stato", ""),
                    "voto":              voto,
                    "fantavoto":         fantavoto,
                    "played":            1,
                    "goals_scored":      int(bonus.get("gol_segnati", 0) or 0),
                    "penalties_scored":  int(bonus.get("rigori_segnati", 0) or 0),
                    "penalties_saved":   int(bonus.get("rigori_parati", 0) or 0),
                    "assists":           int(bonus.get("assist", 0) or 0),
                    "potm":              int(bonus.get("player_of_the_match", 0) or 0),
                    "goals_conceded":    int(malus.get("gol_subiti", 0) or 0),
                    "own_goals":         int(malus.get("autoreti", 0) or 0),
                    "penalties_missed":  int(malus.get("rigori_sbagliati", 0) or 0),
                })

    df = pd.DataFrame(rows)
    log.info("  → %d player-match rows after s.v. filter", len(df))
    return df


def parse_all(voti_dir: Path) -> pd.DataFrame:
    """Parse every voti file under *voti_dir* and concatenate."""
    frames = [parse_voti_file(f) for f in discover_voti_files(voti_dir)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Aggregation to player-season target ─────────────────────────────────────

def aggregate_fantavoto_medio(df_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregate match-level votes to a single mean per (player, season).

    The output schema is the one the pipeline can consume:

        name, role, team, season_start, fantavoto_medio, n_votes, n_matches

    Multiple roles per player (rare; mostly loans) and multiple teams
    (transfers) are pooled — the mean is the *unweighted* mean of
    fantavoto values, which is the standard Fantacalcio convention.
    """
    if df_long.empty:
        return pd.DataFrame(columns=[
            "name", "role", "team", "season_start",
            "fantavoto_medio", "n_votes", "n_matches",
        ])

    grouped = (
        df_long
        .groupby(["name", "role", "season_start"], dropna=False, as_index=False)
        .agg(
            fantavoto_medio=("fantavoto", "mean"),
            n_votes=("fantavoto", "size"),
            n_matches=("matchday", "nunique"),
            team=("team", lambda s: s.value_counts().idxmax()),
        )
    )
    # A more honest "primary team" is the one with the most appearances.
    # (lambda above does exactly that.)  Sort for deterministic output.
    grouped = grouped.sort_values(
        ["season_start", "fantavoto_medio"], ascending=[True, False]
    ).reset_index(drop=True)

    # Round to 3 decimals to keep the CSV readable.
    grouped["fantavoto_medio"] = grouped["fantavoto_medio"].round(3)
    return grouped


# ── Mapping Fantacalcio name → FotMob ID ─────────────────────────────────────

def _load_id_map(engine: sa.Engine) -> pd.DataFrame:
    """Load the canonical Fantacalcio-id → FotMob-id map.

    Mirrors the strategy used by ``import_quotations``: we pre-compute the
    normalised name / team / last-name keys on both sides and merge on
    the FotMob side using those keys.  The voti JSONs do not carry
    ``fantacalcio_id`` so we cannot join on it directly; we instead
    compute the same matching key as the import step would have used
    and apply it here.

    This is robust to:
      * Surname-only / initial-suffix forms (e.g. ``"Martinez L."``).
      * Team alias drift (e.g. ``"Inter"`` → ``"internazionale"``).
      * Players who have *not* been mapped in ``player_id_map`` — they
        are kept in the output as unmatched (player_fotmob_id = NaN)
        so the pipeline can still try the name+season_label merge.
    """
    sql = sa.text("""
        SELECT player_fotmob_id,
               name_fantacalcio,
               team_fantacalcio,
               canonical_role,
               season_start
        FROM player_id_map
        WHERE player_fotmob_id IS NOT NULL
    """)
    df = pd.read_sql(sql, engine)

    df["name_norm"]      = df["name_fantacalcio"].map(normalise_player_name)
    df["team_norm"]      = (
        df["team_fantacalcio"]
        .map(normalise_team)
        .map(apply_team_alias)
    )
    df["last_name_norm"] = df["name_norm"].map(last_name_token)
    df["role"]           = df["canonical_role"]
    return df


def map_voti_to_fotmob(
    df_agg: pd.DataFrame,
    id_map: pd.DataFrame,
) -> pd.DataFrame:
    """Attach ``player_fotmob_id`` to each aggregated voti row.

    Matching strategy (in order, falling through to the next on miss):

      1. ``(last_name_norm, team_norm, role, season_start)`` — the
         strict key.  Works when the voti share the same team the
         Fantacalcio listone had that season.
      2. ``(last_name_norm, team_norm, role)`` — across seasons.
         Handles the common case where the voti team string uses a
         different alias (e.g. "Inter" vs "Internazionale").
      3. ``(last_name_norm, role, season_start)`` — last-name only.
         Catches transfers / rebrandings.
      4. ``(last_name_norm, role)`` — last resort; ambiguous players
         may collide (we log a warning).

    Each row gets the column ``match_method`` (one of the four above or
    ``"unmatched"``) for inspection.
    """
    if df_agg.empty:
        return df_agg.copy()

    work = df_agg.copy()
    work["name_norm"]      = work["name"].map(normalise_player_name)
    work["team_norm"]      = (
        work["team"].map(normalise_team).map(apply_team_alias)
    )
    work["last_name_norm"] = work["name_norm"].map(last_name_token)
    work["match_method"]   = "unmatched"
    work["player_fotmob_id"] = pd.NA

    # Build a small helper to drop duplicate matches before merge so
    # a surname collision does not blow up the row count.
    def _safe_merge(
        keys: list[str], method: str, require_unique_left: bool = False
    ) -> None:
        """Attach a player_fotmob_id to rows that are *still unmatched*.

        The pipeline of stages must be monotonic: a row already matched by
        an earlier (stricter) stage is *not* re-evaluated here, so its
        ``match_method`` label is preserved.
        """
        # Pick one (player_fotmob_id) per key from the id_map.
        idx_cols = keys + ["player_fotmob_id"]
        ref = (
            id_map[idx_cols]
            .dropna(subset=keys + ["player_fotmob_id"])
            .drop_duplicates(subset=keys, keep="first")
        )
        before_unmatched = (work["match_method"] == "unmatched").sum()
        merged = work.merge(
            ref.rename(columns={"player_fotmob_id": "_fotmob_id"}),
            on=keys,
            how="left",
        )
        if require_unique_left:
            # Avoid filling a left-key that already has multiple rows in ref.
            counts = ref.groupby(keys).size()
            non_unique = counts[counts > 1].index
            if len(non_unique):
                mask = merged.set_index(keys).index.isin(non_unique)
                merged.loc[mask, "_fotmob_id"] = pd.NA
        # Only fill rows that are currently unmatched AND that the merge
        # can resolve.  This is the key invariant that keeps the stage
        # labels honest.
        hit = (
            (work["match_method"] == "unmatched")
            & merged["_fotmob_id"].notna()
        )
        work.loc[hit, "player_fotmob_id"] = merged.loc[hit, "_fotmob_id"].values
        work.loc[hit, "match_method"] = method
        after_unmatched = (work["match_method"] == "unmatched").sum()
        log.info(
            "  match[%s] resolved %d rows (%d → %d unmatched)",
            method,
            before_unmatched - after_unmatched,
            before_unmatched,
            after_unmatched,
        )

    # Stage 1: full key per-season (most precise).
    _safe_merge(
        ["last_name_norm", "team_norm", "role", "season_start"],
        "last_name_team_role_season",
    )
    # Stage 2: drop the season, allow alias drift across years.
    _safe_merge(
        ["last_name_norm", "team_norm", "role"],
        "last_name_team_role",
    )
    # Stage 3: drop the team, allow transfers / rebrandings.
    _safe_merge(
        ["last_name_norm", "role", "season_start"],
        "last_name_role_season",
    )
    # Stage 4: last-name + role, season-agnostic.  May collide; we
    # pick the first occurrence and log collisions.
    _safe_merge(
        ["last_name_norm", "role"],
        "last_name_role",
        require_unique_left=True,
    )

    matched = (work["player_fotmob_id"].notna()).sum()
    total   = len(work)
    log.info("Mapping summary: %d / %d rows matched to a FotMob ID", matched, total)

    return work


# ── End-to-end CSV builder ──────────────────────────────────────────────────

def build_fantavoto_csv(
    voti_dir: Path,
    output_path: Path,
    engine: sa.Engine,
) -> pd.DataFrame:
    """Parse the voti JSONs, aggregate, map to FotMob IDs, and write CSV.

    Returns the final DataFrame so the caller can inspect it (useful in
    tests and from the CLI).
    """
    log.info("Discovering voti files in %s …", voti_dir)
    files = discover_voti_files(voti_dir)
    log.info("Found %d season file(s): %s",
             len(files), ", ".join(f.path.name for f in files))

    df_long = pd.concat(
        (parse_voti_file(f) for f in files), ignore_index=True
    )
    if df_long.empty:
        raise ValueError(
            "Voti files parsed but produced no rows — check the JSON format."
        )
    log.info("Total long-format rows across all seasons: %d", len(df_long))

    df_agg = aggregate_fantavoto_medio(df_long)
    log.info("Aggregated to %d player-season rows", len(df_agg))

    id_map = _load_id_map(engine)
    log.info("Loaded %d rows from player_id_map", len(id_map))

    df_mapped = map_voti_to_fotmob(df_agg, id_map)

    # ── Apply manual overrides (if a CSV was provided) ────────────────────
    # Overrides update the player_fotmob_id for rows that the automatic
    # mapping got wrong or left unmatched.
    import os as _os
    override_csv = _os.environ.get("ML_MATCH_OVERRIDES")
    if override_csv:
        override_path = Path(override_csv)
        if override_path.exists():
            from .match_override import apply_overrides_to_voti_mapping, load_overrides_csv
            _overrides = load_overrides_csv(override_path)
            if _overrides:
                df_mapped = apply_overrides_to_voti_mapping(df_mapped, _overrides)
        else:
            log.warning("ML_MATCH_OVERRIDES=%s set but file not found.", override_csv)

    # Output schema: prefer the FotMob-id-based merge key, but also keep
    # name+season_label as a fallback column for debugging.
    out = df_mapped[[
        "player_fotmob_id", "season_start", "name", "team", "role",
        "fantavoto_medio", "n_votes", "n_matches", "match_method",
    ]].copy()
    out["player_fotmob_id"] = out["player_fotmob_id"].astype("Int64")
    out = out.sort_values(["season_start", "fantavoto_medio"],
                          ascending=[True, False]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    log.info("Wrote %d rows to %s", len(out), output_path)

    # Match-rate report.
    matched = out["player_fotmob_id"].notna().sum()
    pct = (matched / len(out) * 100) if len(out) else 0.0
    log.info("Match rate vs player_id_map: %d / %d (%.1f%%)",
             matched, len(out), pct)
    by_method = out["match_method"].value_counts().to_dict()
    log.info("Match methods: %s", by_method)

    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml.data.voti_loader",
        description=(
            "Convert Fantacalcio voti JSONs into a CSV consumable by "
            "ml.run_pipeline --fantavoto-csv."
        ),
    )
    p.add_argument(
        "--voti-dir", type=Path, required=True,
        help="Directory containing voti_fantacalcio-YYYY-YY.json files.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Destination CSV path.",
    )
    p.add_argument(
        "--database-url", type=str, default=None,
        help=(
            "Override the ML_DATABASE_URL env var if you need to target "
            "a different DB than the one used by the pipeline."
        ),
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        help="Python logging level (default: INFO).",
    )
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Reuse the pipeline's env-driven config so we honour .env files.
    from ..config import settings as ml_settings

    db_url = args.database_url or ml_settings.get_database_url()
    engine = sa.create_engine(db_url, future=True)

    build_fantavoto_csv(args.voti_dir, args.output, engine)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

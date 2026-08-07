"""Manual match-override system for Fantacalcio ↔ FotMob ID resolution.

Motivation
----------
The automatic matching pipeline (exact name+team+role → exact relaxed role →
fuzzy surname) resolves ~95 % of Fantacalcio rows, but some players remain
unmatched or are matched with low confidence.  This module lets the operator
resolve those cases manually by editing a simple CSV file.

Workflow
--------
1. **Export** unresolved / low-confidence cases to a CSV.
2. **Edit** the CSV: fill in the ``player_fotmob_id`` and optionally adjust
   ``canonical_role`` / ``team_fotmob``.
3. **Apply** the CSV — the overrides are merged into ``player_id_map`` with
   ``match_method='manual'`` and ``confidence=1.0``.

The same CSV format is shared between ``import_quotations`` (Fantacalcio
listoni → player_id_map) and ``voti_loader`` (Fantacalcio voti → FotMob IDs).

CSV format
----------
.. code-block:: csv

    fantacalcio_id,season_start,name,team,role,player_fotmob_id,canonical_role,team_fotmob,note
    12345,2024,"Doe J.","FC Example",A,98765,DEF,"Example FC","Trasferito a gennaio"
    12346,2024,"Smith","Other FC",C,,,,"Ancora da risolvere"

Columns:
  * ``fantacalcio_id``  — Fantacalcio ID (required, integer).
  * ``season_start``    — Season start year (required, integer).
  * ``name``            — Fantacalcio display name (informational).
  * ``team``            — Fantacalcio team name (informational).
  * ``role``            — Fantacalcio role (P/D/C/A) (informational).
  * ``player_fotmob_id``— FotMob ID to assign (leave blank to keep unmatched).
  * ``canonical_role``  — Override canonical role (GK/DEF/MID/FWD, optional).
  * ``team_fotmob``     — Override FotMob team name (optional).
  * ``note``            — Free-text note (optional).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa

log = logging.getLogger(__name__)

# ── Data structures ──────────────────────────────────────────────────────────

#: Default filename for the override CSV (relative to the quotazioni dir or
#: the project root).
DEFAULT_OVERRIDE_FILENAME = "match_overrides.csv"

#: Columns in the override CSV.
OVERRIDE_COLUMNS = [
    "fantacalcio_id",
    "season_start",
    "name",
    "team",
    "role",
    "player_fotmob_id",
    "canonical_role",
    "team_fotmob",
    "note",
]

#: Minimum confidence threshold below which a match is considered "dubious"
#: and should be surfaced for manual review.
DUBIOUS_CONFIDENCE_THRESHOLD: float = 0.95


@dataclass
class MatchOverride:
    """A single manual override record."""

    fantacalcio_id: int
    season_start: int
    player_fotmob_id: int | None
    name_fantacalcio: str
    name_fotmob: str | None = None
    team_fantacalcio: str | None = None
    team_fotmob: str | None = None
    canonical_role: str | None = None
    note: str | None = None


# ── CSV I/O ──────────────────────────────────────────────────────────────────


def export_unresolved(
    id_map: pd.DataFrame,
    output_path: Path,
    include_dubious: bool = True,
) -> int:
    """Export unmatched / low-confidence rows to a CSV for manual editing.

    Args:
        id_map: DataFrame from ``build_player_id_map()`` or the voti
            equivalent, with columns ``fantacalcio_id``, ``season_start``,
            ``name_fantacalcio``, ``team_fantacalcio``, ``canonical_role``,
            ``match_method``, ``confidence``.
        output_path: Destination CSV path.
        include_dubious: When True, also includes rows with fuzzy / relaxed
            matches whose confidence < ``DUBIOUS_CONFIDENCE_THRESHOLD``.

    Returns:
        Number of rows exported.
    """
    # Filter: unmatched + low-confidence fuzzy/relaxed matches
    is_unmatched = id_map.get("match_method") == "unmatched"
    is_dubious = (
        include_dubious
        & id_map.get("match_method", "").isin({"fuzzy_name", "exact_relaxed_role"})
        & (
            pd.to_numeric(id_map.get("confidence", 1.0), errors="coerce")
            < DUBIOUS_CONFIDENCE_THRESHOLD
        )
    )
    unresolved = id_map[is_unmatched | is_dubious].copy()

    if unresolved.empty:
        log.info("No unresolved rows to export.")
        # Write an empty file with headers anyway
        pd.DataFrame(columns=OVERRIDE_COLUMNS).to_csv(
            output_path, index=False, encoding="utf-8"
        )
        return 0

    rows = []
    for _, row in unresolved.iterrows():
        rows.append(
            {
                "fantacalcio_id": int(row.get("fantacalcio_id", 0)),
                "season_start": int(row.get("season_start", 0)),
                "name": row.get("name_fantacalcio", row.get("name", "")),
                "team": row.get("team_fantacalcio", row.get("team", "")),
                "role": row.get("canonical_role", ""),
                "player_fotmob_id": "",  # blank — operator fills this
                "canonical_role": "",  # blank — operator can override
                "team_fotmob": "",  # blank — operator can override
                "note": "",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(output_path, index=False, encoding="utf-8")
    log.info(
        "Exported %d unresolved rows to %s (include_dubious=%s). "
        "Edit the CSV, fill 'player_fotmob_id', then re-run with --overrides.",
        len(out),
        output_path,
        include_dubious,
    )
    return len(out)


def load_overrides_csv(override_path: Path) -> list[MatchOverride]:
    """Load manual overrides from a CSV file.

    Returns an empty list if the file does not exist or is empty.

    Args:
        override_path: Path to the CSV file.

    Returns:
        List of :class:`MatchOverride` objects.
    """
    if not override_path.exists():
        log.info("Override file not found: %s — skipping.", override_path)
        return []

    df = pd.read_csv(override_path, encoding="utf-8")
    if df.empty:
        return []

    # Validate required columns
    required = {"fantacalcio_id", "season_start"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Override CSV {override_path} is missing required columns: {missing}. "
            f"Expected at least: {required}. Found: {list(df.columns)}"
        )

    overrides: list[MatchOverride] = []
    for _, row in df.iterrows():
        fotmob_raw = row.get("player_fotmob_id")
        fotmob_id: int | None = None
        if pd.notna(fotmob_raw) and str(fotmob_raw).strip():
            try:
                fotmob_id = int(float(str(fotmob_raw).strip()))
            except (ValueError, TypeError):
                log.warning(
                    "Invalid player_fotmob_id=%r in override CSV (row: %s). Skipping.",
                    fotmob_raw,
                    row.to_dict(),
                )
                continue

        overrides.append(
            MatchOverride(
                fantacalcio_id=int(row["fantacalcio_id"]),
                season_start=int(row["season_start"]),
                player_fotmob_id=fotmob_id,
                name_fantacalcio=str(row.get("name", "")),
                team_fantacalcio=str(row.get("team", "")),
                team_fotmob=str(row.get("team_fotmob", "")) or None,
                canonical_role=str(row.get("canonical_role", "")) or None,
                note=str(row.get("note", "")) or None,
            )
        )

    log.info("Loaded %d manual overrides from %s.", len(overrides), override_path)
    return overrides


def apply_overrides_to_id_map(
    id_map: pd.DataFrame,
    overrides: list[MatchOverride],
) -> pd.DataFrame:
    """Apply manual overrides to a ``player_id_map`` DataFrame.

    For each override that references an existing row in *id_map*, the
    ``player_fotmob_id``, ``match_method``, and ``confidence`` are updated
    in-place.  Overrides for rows not present in *id_map* are appended.

    Args:
        id_map: DataFrame from ``build_player_id_map()`` with at minimum
            columns ``fantacalcio_id``, ``season_start``, ``player_fotmob_id``,
            ``match_method``, ``confidence``.
        overrides: List of :class:`MatchOverride` to apply.

    Returns:
        A new DataFrame with overrides applied.
    """
    if not overrides:
        return id_map.copy()

    df = id_map.copy()

    for o in overrides:
        df = _apply_single_override(df, o)

    n_applied = len(overrides)
    log.info("Applied %d manual override(s) to id_map.", n_applied)
    return df


def _apply_single_override(df: pd.DataFrame, o: MatchOverride) -> pd.DataFrame:
    """Apply one override to the id_map DataFrame — returns a new DataFrame."""
    mask = (df["fantacalcio_id"] == o.fantacalcio_id) & (
        df["season_start"] == o.season_start
    )
    if mask.any():
        # Update existing row
        df.loc[mask, "player_fotmob_id"] = o.player_fotmob_id
        df.loc[mask, "match_method"] = "manual"
        df.loc[mask, "confidence"] = 1.0
        if o.team_fotmob:
            df.loc[mask, "team_fotmob"] = o.team_fotmob
        if o.canonical_role:
            df.loc[mask, "canonical_role"] = o.canonical_role
        log.info(
            "Override applied: fantacalcio_id=%d season=%d → player_fotmob_id=%s",
            o.fantacalcio_id,
            o.season_start,
            o.player_fotmob_id,
        )
        return df

    # Append new row (edge case: override for a row not in the original map)
    new_row: dict[str, Any] = {
        "fantacalcio_id": o.fantacalcio_id,
        "season_start": o.season_start,
        "player_fotmob_id": o.player_fotmob_id,
        "name_fantacalcio": o.name_fantacalcio,
        "name_fotmob": None,
        "team_fantacalcio": o.team_fantacalcio,
        "team_fotmob": o.team_fotmob,
        "canonical_role": o.canonical_role,
        "match_method": "manual",
        "confidence": 1.0,
    }
    for col in df.columns:
        if col not in new_row:
            new_row[col] = None
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    log.info(
        "Override appended (new row): fantacalcio_id=%d season=%d → player_fotmob_id=%s",
        o.fantacalcio_id,
        o.season_start,
        o.player_fotmob_id,
    )
    return df


def apply_overrides_to_voti_mapping(
    df_voti: pd.DataFrame,
    overrides: list[MatchOverride],
) -> pd.DataFrame:
    """Apply manual overrides to a voti-mapped DataFrame.

    The voti loader produces a DataFrame with columns ``fantacalcio_id?``
    … actually the voti loader does not have ``fantacalcio_id`` — it maps
    via the same ``player_id_map`` table.  This function updates the
    ``player_fotmob_id`` in the voti DataFrame for rows that were matched
    to a ``fantacalcio_id`` via the id_map.

    Args:
        df_voti: Voti DataFrame output by ``map_voti_to_fotmob()``.
        overrides: List of :class:`MatchOverride` to apply.

    Returns:
        A new DataFrame with overrides applied.
    """
    if not overrides:
        return df_voti.copy()

    df = df_voti.copy()
    applied = 0
    for o in overrides:
        if o.player_fotmob_id is None:
            continue
        # Voti rows don't carry fantacalcio_id directly — we rely on the
        # name+team+season triple to locate the row.
        mask = (df.get("season_start", 0) == o.season_start) & (
            df.get("name", "").astype(str).str.strip()
            == str(o.name_fantacalcio).strip()
        )
        if o.team_fantacalcio:
            mask = mask & (
                df.get("team", "").astype(str).str.strip()
                == str(o.team_fantacalcio).strip()
            )

        if mask.any():
            df.loc[mask, "player_fotmob_id"] = o.player_fotmob_id
            df.loc[mask, "match_method"] = "manual"
            applied += 1
            log.info(
                "Voti override: name=%s season=%d → player_fotmob_id=%s",
                o.name_fantacalcio,
                o.season_start,
                o.player_fotmob_id,
            )
        else:
            log.warning(
                "Voti override not applied — no matching row found for name=%s season=%d",
                o.name_fantacalcio,
                o.season_start,
            )

    log.info("Applied %d/%d voti override(s).", applied, len(overrides))
    return df


# ── CLI (interactive) ────────────────────────────────────────────────────────


def _list_unmatched_from_db(
    engine: sa.Engine,
    season_start: int | None = None,
) -> pd.DataFrame:
    """Query the DB for unmatched or low-confidence mapping rows.

    Returns a DataFrame suitable for CSV export.
    """
    season_filter = ""
    if season_start is not None:
        season_filter = f"AND pim.season_start = {int(season_start)}"

    sql = f"""
        SELECT
            pim.fantacalcio_id,
            pim.season_start,
            pim.name_fantacalcio AS name,
            pim.team_fantacalcio  AS team,
            pim.canonical_role    AS role,
            pim.player_fotmob_id,
            pim.match_method,
            pim.confidence
        FROM player_id_map pim
        WHERE (
            pim.match_method = 'unmatched'
            OR (
                pim.match_method IN ('fuzzy_name', 'exact_relaxed_role')
                AND pim.confidence < {DUBIOUS_CONFIDENCE_THRESHOLD}
            )
        )
        {season_filter}
        ORDER BY pim.season_start DESC, pim.fantacalcio_id
    """
    return pd.read_sql(sa.text(sql), engine)


# ── Standalone CLI ────────────────────────────────────────────────────────────


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml.data.match_override",
        description="Manual match-override tools for Fantacalcio ↔ FotMob ID resolution.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── export subcommand ────────────────────────────────────────────────
    export_p = sub.add_parser(
        "export",
        help="Export unmatched/low-confidence rows to a CSV for manual editing.",
    )
    export_p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(DEFAULT_OVERRIDE_FILENAME),
        help=f"Output CSV path (default: {DEFAULT_OVERRIDE_FILENAME}).",
    )
    export_p.add_argument(
        "--season",
        type=int,
        default=None,
        help="Filter to a specific season start year.",
    )
    export_p.add_argument(
        "--no-dubious",
        action="store_true",
        help="Exclude fuzzy/relaxed matches (export only truly unmatched).",
    )
    export_p.add_argument(
        "--db-url",
        default=None,
        help="Database URL. Defaults to ML_DATABASE_URL or API_DATABASE_URL.",
    )

    # ── apply subcommand ─────────────────────────────────────────────────
    apply_p = sub.add_parser(
        "apply",
        help="Apply a manually-edited override CSV to the DB.",
    )
    apply_p.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="CSV file with overrides (see CSV format above).",
    )
    apply_p.add_argument(
        "--db-url",
        default=None,
        help="Database URL. Defaults to ML_DATABASE_URL or API_DATABASE_URL.",
    )
    apply_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without touching the DB.",
    )

    return p


def _cli_export(args: argparse.Namespace) -> int:
    """Run the ``export`` subcommand."""
    import os

    db_url = (
        args.db_url
        or os.environ.get("ML_DATABASE_URL")
        or os.environ.get("API_DATABASE_URL")
    )
    if not db_url:
        log.error("Database URL not set. Pass --db-url or export ML_DATABASE_URL.")
        return 2

    engine = sa.create_engine(db_url, pool_pre_ping=True)
    df = _list_unmatched_from_db(engine, season_start=args.season)
    if df.empty:
        log.info("No unresolved rows found. Nothing to export.")
        pd.DataFrame(columns=OVERRIDE_COLUMNS).to_csv(
            args.output, index=False, encoding="utf-8"
        )
        return 0

    # Build override-format columns
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "fantacalcio_id": int(row["fantacalcio_id"]),
                "season_start": int(row["season_start"]),
                "name": row.get("name", ""),
                "team": row.get("team", ""),
                "role": row.get("role", ""),
                "player_fotmob_id": int(row["player_fotmob_id"])
                if pd.notna(row.get("player_fotmob_id"))
                else "",
                "canonical_role": "",
                "team_fotmob": "",
                "note": "",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False, encoding="utf-8")
    n = len(out)
    log.info("Exported %d rows to %s.", n, args.output)
    print(f"\n✅ Exported {n} unresolved rows to {args.output}")
    print("   Edit the CSV, fill 'player_fotmob_id' column, then run:")
    print(f"   python -m ml.data.match_override apply -i {args.output}\n")
    return 0


def _cli_apply(args: argparse.Namespace) -> int:
    """Run the ``apply`` subcommand."""
    import os

    db_url = (
        args.db_url
        or os.environ.get("ML_DATABASE_URL")
        or os.environ.get("API_DATABASE_URL")
    )
    if not db_url:
        log.error("Database URL not set. Pass --db-url or export ML_DATABASE_URL.")
        return 2

    overrides = load_overrides_csv(args.input)
    if not overrides:
        log.info("No overrides found in %s — nothing to do.", args.input)
        return 0

    engine = sa.create_engine(db_url, pool_pre_ping=True)

    # 1. Load current id_map from DB
    current = pd.read_sql(
        sa.text("SELECT * FROM player_id_map"),
        engine,
    )
    log.info("Loaded %d existing id_map rows from DB.", len(current))

    # 2. Apply overrides
    updated = apply_overrides_to_id_map(current, overrides)

    # 3. Detect changed rows
    merged = current.merge(
        updated,
        on=["fantacalcio_id", "season_start"],
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
        validate="one_to_one",
    )
    changed = merged[merged["_merge"] == "both"].copy()
    # Compare specific columns
    for col in ("player_fotmob_id", "match_method", "confidence"):
        changed = changed[
            changed[f"{col}_old"].fillna(-1).astype(str)
            != changed[f"{col}_new"].fillna(-1).astype(str)
        ]

    if args.dry_run:
        print(f"\n📋 Dry-run: {len(changed)} rows WOULD be updated:")
        for _, row in changed.iterrows():
            print(
                f"  - id={int(row['fantacalcio_id'])} "
                f"season={int(row['season_start'])} "
                f"→ fotmob_id={row.get('player_fotmob_id_new', '?')} "
                f"(was: {row.get('player_fotmob_id_old', '?')})"
            )
        return 0

    # 4. Persist
    _UPSERT_MANUAL_SQL = sa.text("""
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
            canonical_role   = EXCLUDED.canonical_role,
            match_method     = EXCLUDED.match_method,
            confidence       = EXCLUDED.confidence,
            updated_at       = NOW()
    """)

    payload = (
        updated.astype(object)
        .where(pd.notnull(updated), None)
        .to_dict(orient="records")
    )
    with engine.begin() as conn:
        conn.execute(_UPSERT_MANUAL_SQL, payload)

    n_manual = int((updated["match_method"] == "manual").sum())
    log.info(
        "Persisted %d rows to player_id_map (%d with match_method='manual').",
        len(payload),
        n_manual,
    )
    print(f"\n✅ Applied {len(overrides)} override(s) to player_id_map.")
    return 0


def main() -> int:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.command == "export":
        return _cli_export(args)
    elif args.command == "apply":
        return _cli_apply(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

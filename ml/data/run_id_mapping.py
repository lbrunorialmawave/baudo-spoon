"""One-shot ID mapping: match Fantacalcio players to FotMob data.

Reads Fantacalcio quotations and FotMob player data from DB,
performs exact + fuzzy matching, persists to player_id_map.

Usage:
    python -m ml.data.run_id_mapping
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import sqlalchemy as sa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_id_mapping")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize names and teams for matching."""
    from ml.data.import_quotations import (
        apply_team_alias,
        last_name_token,
        normalise_player_name,
        normalise_team,
    )
    df = df.copy()
    if "player_name" in df.columns:
        df["name_norm"] = df["player_name"].map(normalise_player_name)
        df["last_name_norm"] = df["name_norm"].map(last_name_token)
    if "team_fotmob" in df.columns:
        df["team_norm"] = df["team_fotmob"].map(normalise_team).map(apply_team_alias)
    elif "team" in df.columns:
        df["team_norm"] = df["team"].map(normalise_team).map(apply_team_alias)
    return df


def _load_manual_resolutions(engine: sa.Engine) -> pd.DataFrame:
    """Load all rows from the ``manual_resolutions`` history table."""
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


def main() -> int:
    db_url = (
        os.environ.get("ML_DATABASE_URL")
        or os.environ.get("API_DATABASE_URL")
        or "postgresql://fbref:DevPassword123@db:5432/fbref"
    )
    engine = sa.create_engine(db_url, pool_pre_ping=True)

    # ── 1. Load Fantacalcio quotations ──────────────────────────────────────
    log.info("Loading Fantacalcio quotations …")
    quotes = pd.read_sql(
        "SELECT fantacalcio_id, season_start, player_name, team, role "
        "FROM player_quotations ORDER BY fantacalcio_id",
        engine,
    )
    log.info("  %d rows across %d seasons",
             len(quotes), quotes["season_start"].nunique())
    quotes = _normalize(quotes)
    quotes["canonical_role"] = quotes["role"]
    quotes["name"] = quotes["player_name"]  # matching functions expect 'name' column

    # ── 2. Build FotMob reference from player_season_stats ──────────────────
    log.info("Building FotMob reference from player_season_stats …")
    fotmob = pd.read_sql(
        """
        SELECT DISTINCT
            pss.player_fotmob_id,
            pss.player_name,
            pss.team_name AS team_fotmob,
            s.season_start
        FROM player_season_stats pss
        JOIN seasons s ON s.id = pss.season_id
        WHERE pss.player_fotmob_id IS NOT NULL
        """,
        engine,
    )
    log.info("  %d distinct player-season rows", len(fotmob))
    if fotmob.empty:
        log.error("No FotMob data found!")
        return 1
    fotmob = _normalize(fotmob)
    fotmob["canonical_role"] = None  # player_season_roles is empty

    # ── 3. Matching ────────────────────────────────────────────────────────
    from ml.data.import_quotations import (
        _exact_match_relaxed_role,
        _fuzzy_match_one,
        persist_player_id_map,
    )

    results: list[dict] = []
    matched_keys: set[tuple[int, int]] = set()

    # ── Pass 0: historical manual resolutions ──────────────────────────────
    log.info("Pass 0: applying historical manual resolutions …")
    historical = _load_manual_resolutions(engine)
    if not historical.empty:
        historical = historical.sort_values(
            ["fantacalcio_id", "season_start", "player_fotmob_id"],
            ascending=[True, False, False],
        )
        latest_per_id = historical.drop_duplicates(
            subset="fantacalcio_id", keep="first"
        )
        merged = quotes.merge(
            latest_per_id[["fantacalcio_id", "player_fotmob_id",
                           "name_fotmob", "team_fotmob",
                           "canonical_role"]],
            on="fantacalcio_id",
            how="inner",
            suffixes=("", "_hist"),
        )
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

    # ── Filter out Pass-0 matches from remaining rows ──────────────────────
    remaining = quotes[
        ~quotes.apply(
            lambda r: (int(r["fantacalcio_id"]), int(r["season_start"])) in matched_keys,
            axis=1,
        )
    ].copy()

    # Pass 1: exact surname + team (relaxed role)
    log.info("Pass 1: exact match on (surname, team) … (%d remaining)", len(remaining))
    matched, unmatched = _exact_match_relaxed_role(remaining, fotmob)
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
            "canonical_role": row.get("canonical_role"),
            "match_method": "exact_name_team",
            "confidence": 1.0,
            "resolved_from_history": False,
        })

    # Pass 2: fuzzy surname match
    log.info("Pass 2: fuzzy match on surname …")
    fuzzy_hits = 0
    still_unmatched: list[dict] = []
    for _, row in unmatched.iterrows():
        best = _fuzzy_match_one(
            last_name_norm=row["last_name_norm"],
            team_norm=row["team_norm"],
            canonical_role=row.get("canonical_role"),
            candidates=fotmob,
        )
        if best is None:
            still_unmatched.append({
                "fantacalcio_id": int(row["fantacalcio_id"]),
                "season_start": int(row["season_start"]),
                "name": row["name"],
                "team": row["team"],
                "canonical_role": row.get("canonical_role"),
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
            "canonical_role": row.get("canonical_role"),
            "match_method": "fuzzy_name",
            "confidence": min(round(score, 3), 1.0),
            "resolved_from_history": False,
        })
        fuzzy_hits += 1
    log.info("  fuzzy hits: %d, still unmatched: %d",
             fuzzy_hits, len(still_unmatched))

    # Pass 3: unmatched rows
    all_matched_keys = {(r["fantacalcio_id"], r["season_start"]) for r in results}
    for case in still_unmatched:
        key = (case["fantacalcio_id"], case.get("season_start", 2025))
        if key in all_matched_keys:
            continue
        results.append({
            "fantacalcio_id": key[0],
            "season_start": key[1],
            "player_fotmob_id": None,
            "name_fantacalcio": case["name"],
            "name_fotmob": None,
            "team_fantacalcio": case["team"],
            "team_fotmob": None,
            "canonical_role": case.get("canonical_role"),
            "match_method": "unmatched",
            "confidence": 0.0,
            "resolved_from_history": False,
        })

    id_map = pd.DataFrame(results)
    dist = id_map["match_method"].value_counts().to_dict()
    log.info("ID map distribution: %s", dist)
    total = len(id_map)
    matched_count = total - dist.get("unmatched", 0)
    log.info("Match rate: %.1f%% (%d/%d)", matched_count / total * 100, matched_count, total)

    # ── 4. Persist automatic results ─────────────────────────────────────
    log.info("Persisting automatic mapping results …")
    persist_player_id_map(id_map, engine=engine)

    # ── 5. Re-apply historical resolutions on top (preserve Pass 0) ──────
    # This ensures that even if Pass 1-3 matched a player differently, the
    # historical manual override wins.
    if not historical.empty:
        log.info("Re-applying %d historical manual resolutions …", len(historical))
        # Map historical rows to player_id_map format
        hist_rows = []
        for _, row in historical.iterrows():
            hist_rows.append({
                "fantacalcio_id": int(row["fantacalcio_id"]),
                "season_start": int(row["season_start"]),
                "player_fotmob_id": int(row["player_fotmob_id"]),
                "name_fantacalcio": str(row.get("name_fantacalcio", "")),
                "name_fotmob": row.get("name_fotmob"),
                "team_fantacalcio": row.get("team_fantacalcio"),
                "team_fotmob": row.get("team_fotmob"),
                "canonical_role": row.get("canonical_role"),
                "match_method": "manual",
                "confidence": 1.0,
                "resolved_from_history": True,
            })
        hist_df = pd.DataFrame(hist_rows)
        persist_player_id_map(hist_df, engine=engine)
        log.info("Re-applied %d historical resolutions.", len(hist_rows))

    log.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

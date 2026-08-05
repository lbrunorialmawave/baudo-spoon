"""Orchestrator: DB → compute → JSON output.

Reads data from the PostgreSQL database, runs the full MANTRA scoring
pipeline, and produces a structured JSON result with all pillars,
classifications, and metadata.

MANTRA pre-season fallback
--------------------------
P1 (Solidita), P2 (Potenziale), and P3 (Peso Squadra) are built from
``player_season_aggregates`` / ``team_strength_aggregates`` for the target
season — real match data that, before the season kicks off (e.g. while
prepping for the auction), simply does not exist yet. ``_PLAYER_DATA_SQL``
therefore falls back to ``season_start - 1`` per player/team whenever the
target season has no rows, so a pre-season run scores players on their most
recent actual performance instead of collapsing every pillar to a
league-wide median (or zero). Market data (Pz1/Pz2/Pz3, used by P4) always
comes from the target season's own quotations, since that's the one input
that *is* available pre-season. Each player record carries a
``stats_from_prior_season`` flag so callers can tell which rows used the
fallback.

Cross-league fallback for players new to Serie A
-------------------------------------------------
A player with zero Serie A history (e.g. a transfer just arrived from
another league) has no rows in ``player_season_aggregates`` for either the
target season or the one before it. A third COALESCE tier,
``player_latest_stats_any_league`` (a view with no league filter — see
migration 018), supplies his single most recent season anywhere, so he's
scored on real performance instead of ``pilastro1.py``'s role-median guess
for neo-arrivi. This tier deliberately does **not** feed ``Stagioni_IT``
(``seasons_in_italy`` stays Serie-A-only): "has real stats somewhere" and
"has adapted to Italian football" are different facts, and ``is_neo_arrivo``
downstream must keep meaning the latter. Rows using this tier carry
``stats_from_foreign_league = True``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import sqlalchemy as sa

from ml.domain.predictions import resolve_season_value_fields
from ml.mantra.config import MantraConfig
from ml.mantra.pilastro1 import compute_p1
from ml.mantra.pilastro2 import compute_p2
from ml.mantra.pilastro3 import (
    compute_ps_corretto,
    compute_p3,
)
from ml.mantra.pilastro4 import compute_cp, compute_p4
from ml.mantra.scoring import compute_fp, compute_fp_corr
from ml.mantra.fase7 import classify_fase7
from ml.mantra.fase8 import (
    top_per_ruolo,
    multi_eleggibilita,
    low_cost,
    scommesse_multi_ruolo,
    watchlist_giovani,
    rischio_contestuale,
    consigliati_giornata,
    indisponibili,
)

log = logging.getLogger(__name__)


# ── SQL queries ──────────────────────────────────────────────────────────────

_PLAYER_DATA_SQL = sa.text("""
    SELECT
        pq.fantacalcio_id,
        pq.season_start,
        pq.player_name,
        pq.team,
        pq.qt_a AS "Pz1",
        pq.qt_i AS "Pz2",
        pq.fvm  AS "Pz3",
        pmr.ruolo_primario,
        pmr.ruoli_mantra,
        pim.player_fotmob_id,
        -- Stats from player_season_stats (aggregated), falling back to the
        -- prior season when the target season has no played matches yet
        -- (pre-season valuation, e.g. before the auction), then to the
        -- player's most recent season in ANY league when he has no Serie A
        -- history at all — see the module docstring's two fallback notes.
        COALESCE(pss.minutes_avg, pss_prev.minutes_avg, pss_foreign.minutes_avg)             AS "Min_annuo",
        COALESCE(pss.vote_avg, pss_prev.vote_avg, pss_foreign.vote_avg)                       AS "V",
        COALESCE(pss.vote_std, pss_prev.vote_std, pss_foreign.vote_std)                       AS "DV",
        COALESCE(pss.presence_rate, pss_prev.presence_rate, pss_foreign.presence_rate)        AS "Pr",
        COALESCE(pss.xg_per90, pss_prev.xg_per90, pss_foreign.xg_per90)                       AS "xG90",
        COALESCE(pss.xa_per90, pss_prev.xa_per90, pss_foreign.xa_per90)                       AS "xA90",
        COALESCE(pss.goals_per90, pss_prev.goals_per90, pss_foreign.goals_per90)              AS "G90",
        COALESCE(pss.assists_per90, pss_prev.assists_per90, pss_foreign.assists_per90)        AS "A90",
        COALESCE(pss.saves_per90, pss_prev.saves_per90, pss_foreign.saves_per90)              AS saves_per90,
        COALESCE(pss.clean_sheet_per90, pss_prev.clean_sheet_per90, pss_foreign.clean_sheet_per90) AS clean_sheet_per90,
        -- Serie-A-only, deliberately no foreign tier — see docstring.
        COALESCE(pss.seasons_in_italy, pss_prev.seasons_in_italy)   AS "Stagioni_IT",
        (pss.fantacalcio_id IS NULL AND pss_prev.fantacalcio_id IS NOT NULL) AS stats_from_prior_season,
        (pss.fantacalcio_id IS NULL AND pss_prev.fantacalcio_id IS NULL
         AND pss_foreign.fantacalcio_id IS NOT NULL)                AS stats_from_foreign_league,
        -- Context (nullable when player_profiles match is missing)
        CAST(NULL AS FLOAT)   AS "Eta",
        FALSE                 AS "Cambio_Squadra",
        -- Team strength (from team_season_stats), same prior-season fallback.
        -- Pinned to Serie A: pq.team is always an Italian club, and this
        -- table now holds other leagues' clubs too (migration 018).
        COALESCE(ts.team_rank_norm, ts_prev.team_rank_norm)         AS team_rank_norm,
        COALESCE(ts.prev_season_points, ts_prev.prev_season_points) AS prev_season_points,
        COALESCE(ts.goal_difference, ts_prev.goal_difference)       AS goal_difference,
        COALESCE(ts.avg_team_rating, ts_prev.avg_team_rating)       AS avg_team_rating
    FROM player_quotations pq
    JOIN player_mantra_roles pmr
        ON pmr.fantacalcio_id = pq.fantacalcio_id
        AND pmr.season_start = pq.season_start
    -- Bridge fantacalcio_id → player_fotmob_id via the id map
    LEFT JOIN player_id_map pim
        ON pim.fantacalcio_id = pq.fantacalcio_id
        AND pim.season_start = pq.season_start
    LEFT JOIN player_season_aggregates pss
        ON pss.fantacalcio_id = pim.player_fotmob_id::bigint
        AND pss.season_start = pq.season_start
    LEFT JOIN player_season_aggregates pss_prev
        ON pss_prev.fantacalcio_id = pim.player_fotmob_id::bigint
        AND pss_prev.season_start = pq.season_start - 1
    LEFT JOIN player_latest_stats_any_league pss_foreign
        ON pss_foreign.fantacalcio_id = pim.player_fotmob_id::bigint
    LEFT JOIN player_profiles p
        ON p.player_fotmob_id = pim.player_fotmob_id::bigint
    LEFT JOIN team_strength_aggregates ts
        ON ts.team_name = pq.team
        AND ts.season_start = pq.season_start
        AND ts.league_name = 'Serie A'
    LEFT JOIN team_strength_aggregates ts_prev
        ON ts_prev.team_name = pq.team
        AND ts_prev.season_start = pq.season_start - 1
        AND ts_prev.league_name = 'Serie A'
    WHERE pq.season_start = :season_start
    ORDER BY pq.player_name
""")


# ── Predictions artefact plumbing (season_value / start_probability) ───────

# Convention: the ML trainer writes its latest run to
# ``<artifacts_dir>/results_latest.json`` (the same convention used by
# ``DataRepository._load_predictions`` in the API layer). The mantra
# runner reads the same file, keyed by ``player_fotmob_id``.
_PREDICTIONS_FILENAME: str = "results_latest.json"


def _load_predictions_by_id(
    artifacts_dir: Optional[Path],
) -> dict[int, dict[str, Any]]:
    """Return a ``player_fotmob_id → prediction_record`` lookup.

    The predictions artefact is written by ``ml.pipeline.trainer`` and
    is purely informational from the MANTRA side: the two fields we
    pull from it (``fantapunti_totali`` / ``probabilita_titolarita``)
    sit alongside ``FP_Mantra`` / ``VR`` in the output without
    blending, reconciling, or overriding them — see the P1-4 ADR.

    Missing artefacts, missing ``player_fotmob_id`` keys, and NaN ids
    are silently treated as "no prediction" — the caller can then
    default the two extra fields to ``None`` without special casing.
    """
    if artifacts_dir is None:
        return {}
    path = Path(artifacts_dir) / _PREDICTIONS_FILENAME
    if not path.is_file():
        log.debug("Predictions artefact not found at %s; season_value will be None.", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        # Informational plumbing only — never block the MANTRA pipeline
        # because a missing/old artefact is the common case (e.g. a
        # fresh season where the trainer has not been run yet).
        log.warning("Could not read predictions artefact at %s: %s", path, exc)
        return {}

    lookup: dict[int, dict[str, Any]] = {}
    for record in payload.get("predictions", []) or []:
        raw_id = record.get("player_fotmob_id")
        if not isinstance(raw_id, (int, float)):
            continue
        if isinstance(raw_id, float) and raw_id != raw_id:  # NaN
            continue
        lookup[int(raw_id)] = record
    return lookup


def load_data(
    engine: sa.Engine,
    season_start: int,
) -> pd.DataFrame:
    """Load player data from the DB for MANTRA computation."""
    df = pd.read_sql(_PLAYER_DATA_SQL, engine, params={"season_start": season_start})

    # ── Ensure numeric columns are proper floats ──────────────────────────────
    # LEFT JOINs on aggregated views can produce NULLs → psycopg2 returns
    # Python None → pandas infers object dtype.  Convert them explicitly
    # so numpy ufuncs (tanh, etc.) don't choke on object-dtype arrays.
    _NUMERIC_COLS = [
        "Pz1", "Pz2", "Pz3",
        "Min_annuo", "V", "DV", "Pr",
        "xG90", "xA90", "G90", "A90",
        "saves_per90", "clean_sheet_per90",
        "Stagioni_IT",
        "Eta",
        "team_rank_norm", "prev_season_points",
        "goal_difference", "avg_team_rating",
    ]
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["stats_from_prior_season"] = df["stats_from_prior_season"].fillna(False).astype(bool)
    df["stats_from_foreign_league"] = df["stats_from_foreign_league"].fillna(False).astype(bool)

    if df.empty:
        raise ValueError(
            f"No data found for season_start={season_start}. "
            "Run import_quotations first to populate player_quotations "
            "and player_mantra_roles."
        )

    # Compute Num_Ruoli from ruoli_mantra array
    df["num_ruoli"] = df["ruoli_mantra"].apply(
        lambda r: len(r) if isinstance(r, (list, np.ndarray)) else 1
    )
    df["is_neo_arrivo"] = df["Stagioni_IT"].fillna(0) == 0
    df["is_starter"] = df["Pz1"].fillna(0) >= 15

    log.info(
        "Loaded %d players for season %d (%d neo-arrivi, %d with MANTRA roles)",
        len(df), season_start,
        df["is_neo_arrivo"].sum(),
        df["ruolo_primario"].notna().sum(),
    )
    return df


def run_mantra(
    engine: sa.Engine,
    season_start: int,
    cfg: Optional[MantraConfig] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute the full MANTRA scoring pipeline.

    Parameters
    ----------
    engine:
        SQLAlchemy engine connected to the PostgreSQL database.
    season_start:
        Target season (e.g. 2025 for 2025/26).
    cfg:
        Optional configuration override. Uses defaults if None.
    output_dir:
        Optional directory to write JSON output files.

    Returns
    -------
    dict with keys:
        - meta: run metadata (season, timestamp, config)
        - players: list of player records with all scores
        - classifications: phase 8 results
    """
    if cfg is None:
        cfg = MantraConfig()

    log.info("=" * 60)
    log.info("MANTRA scoring engine — season %d", season_start)
    log.info("Config: P1=%.2f P2=%.2f P3=%.2f P4=%.2f",
             cfg.PESO_P1, cfg.PESO_P2, cfg.PESO_P3, cfg.PESO_P4)

    # 1. Load data
    df = load_data(engine, season_start)

    # 2. Compute pillars
    log.info("Computing P1 (Solidità) …")
    p1 = compute_p1(df, cfg)

    log.info("Computing P2 (Potenziale) …")
    p2 = compute_p2(df, cfg)

    log.info("Computing P3 (Peso Squadra) …")
    # Aggregate team-level PS_corretto first
    team_cols = [
        "team_rank_norm", "prev_season_points", "goal_difference",
        "avg_team_rating", "season_start",
    ]
    team_df = df[["team"] + [c for c in team_cols if c in df.columns]].drop_duplicates(
        subset="team"
    ).copy()
    # Add squad_value_market from player quotations
    squad_value = df.groupby("team")["Pz1"].sum().reset_index()
    team_df = team_df.merge(squad_value, on="team", how="left")
    team_df = team_df.rename(columns={"Pz1": "squad_value_market"})

    ps_corretto = compute_ps_corretto(team_df, cfg)
    # Merge back to player-level
    ps_map = dict(zip(team_df["team"], ps_corretto))
    player_ps = df["team"].map(ps_map).fillna(50.0)

    p3 = compute_p3(df, player_ps, cfg)

    log.info("Computing P4 (Mercato Storico) …")
    p4 = compute_p4(df, p1, p2, p3, cfg)
    cp = compute_cp(p1, p2, p3)

    # 3. Scoring
    log.info("Computing FP, VR, Prezzo …")
    fp = compute_fp(p1, p2, p3, p4, cfg)
    scores = compute_fp_corr(fp, cp, df["ruolo_primario"], df["num_ruoli"], df["Pz1"], cfg)

    # 4. Fase 7
    log.info("Classifying Fase 7 …")
    fase7_label, fase7_motivo = classify_fase7(df, fp, scores["fp_mantra"], scores["vr"], p1, cfg)

    # 5. Fase 8
    log.info("Classifying Fase 8 …")
    df_all = df.copy()
    df_all["p1"] = p1
    df_all["p2"] = p2
    df_all["p3"] = p3
    df_all["p4"] = p4
    df_all["cp"] = cp
    df_all["fp"] = fp
    for k, v in scores.items():
        df_all[k] = v
    df_all["fase7_label"] = fase7_label

    classification_8a = top_per_ruolo(df_all)
    classification_8a2 = multi_eleggibilita(df_all)
    classification_8c = low_cost(df_all, cfg.LOW_COST_SOGLIA)
    classification_8d = low_cost(df_all, cfg.LOW_COST_SOGLIA, require_titolare=True)
    classification_8e = scommesse_multi_ruolo(df_all, fase7_label)
    classification_8f = watchlist_giovani(df_all, cfg.GIOVANE_ETA_MAX)
    classification_8g = rischio_contestuale(df_all)

    # 6. Build output
    # Load the ML predictions artefact (if present) so the two
    # informational fields ``season_value`` / ``start_probability`` can
    # be projected onto each player record. The lookup is keyed by
    # ``player_fotmob_id`` (matching the convention used by
    # ``DataRepository.get_player_pool`` and the trainer).
    predictions_by_id = _load_predictions_by_id(output_dir)
    players_out: list[dict] = []
    for idx in df.index:
        # player_fotmob_id may be NaN when the LEFT JOIN on player_id_map
        # does not find a match; map to None so the JSON round-trips cleanly.
        raw_fotmob = df.at[idx, "player_fotmob_id"]
        player_fotmob_id: int | None = (
            int(raw_fotmob) if pd.notna(raw_fotmob) else None
        )
        # Look up the prediction (if any) and project the two informational
        # fields. ``resolve_season_value_fields`` is the same helper the
        # API/optimizer pool uses, so the three surfaces stay in lock-step.
        pred_record = (
            predictions_by_id.get(player_fotmob_id)
            if player_fotmob_id is not None
            else None
        )
        season_value, start_probability = resolve_season_value_fields(pred_record)
        players_out.append({
            "fantacalcio_id": int(df.at[idx, "fantacalcio_id"]),
            "player_fotmob_id": player_fotmob_id,
            "season_start": int(df.at[idx, "season_start"]),
            "player_name": str(df.at[idx, "player_name"]),
            "team": str(df.at[idx, "team"]),
            "ruolo_primario": str(df.at[idx, "ruolo_primario"]),
            "ruoli_mantra": (
                list(df.at[idx, "ruoli_mantra"])
                if isinstance(df.at[idx, "ruoli_mantra"], (list, np.ndarray))
                else []
            ),
            # Newspaper prices
            "Pz1": int(df.at[idx, "Pz1"]),
            "Pz2": int(df.at[idx, "Pz2"]),
            "Pz3": int(df.at[idx, "Pz3"]),
            # True when P1/P2/P3 fell back to the prior season's performance
            # data because the target season has no played matches yet.
            "stats_from_prior_season": bool(df.at[idx, "stats_from_prior_season"]),
            # True when the player has no Serie A history at all and P1/P2/P3
            # instead used his most recent season in another league.
            "stats_from_foreign_league": bool(df.at[idx, "stats_from_foreign_league"]),
            # Computed pillars
            "P1": round(float(p1.iloc[idx]), 2),
            "P2": round(float(p2.iloc[idx]), 2),
            "P3": round(float(p3.iloc[idx]), 2),
            "P4": round(float(p4.iloc[idx]), 2),
            "CP": round(float(cp.iloc[idx]), 2),
            "FP": round(float(fp.iloc[idx]), 2),
            "FP_Corr": round(float(scores["fp_corr"].iloc[idx]), 2),
            "CP_Corr": round(float(scores["cp_corr"].iloc[idx]), 2),
            "FP_Mantra": round(float(scores["fp_mantra"].iloc[idx]), 2),
            "VR": round(float(scores["vr"].iloc[idx]), 2),
            "Prezzo_Massimo": round(float(scores["prezzo_massimo"].iloc[idx]), 2),
            "Percentile_Ruolo": round(float(scores["percentile_ruolo"].iloc[idx]), 4),
            "Fase7": str(fase7_label.iloc[idx]) if pd.notna(fase7_label.iloc[idx]) else None,
            "Fase7_Motivo": str(fase7_motivo.iloc[idx]) if pd.notna(fase7_motivo.iloc[idx]) else None,
            "rischio": str(classification_8g.iloc[idx]) if pd.notna(classification_8g.iloc[idx]) else None,
            # ML predictions — informational only, not blended with the
            # 4-pillar system. See P1-4 for the still-open reconciliation.
            "season_value": season_value,
            "start_probability": start_probability,
        })

    result: dict[str, Any] = {
        "meta": {
            "season_start": season_start,
            "generated_at": datetime.utcnow().isoformat(),
            "config": {
                "PESO_P1": cfg.PESO_P1,
                "PESO_P2": cfg.PESO_P2,
                "PESO_P3": cfg.PESO_P3,
                "PESO_P4": cfg.PESO_P4,
                "SOGLIA_MINUTI_MIN": cfg.SOGLIA_MINUTI_MIN,
                "SOGLIA_MINUTI_MAX": cfg.SOGLIA_MINUTI_MAX,
            },
            "n_players": len(players_out),
            "n_players_prior_season_fallback": int(df["stats_from_prior_season"].sum()),
            "n_players_foreign_fallback": int(df["stats_from_foreign_league"].sum()),
        },
        "players": players_out,
        "classifications": {
            "top_per_ruolo": {
                ruolo: subset["player_name"].head(15).tolist()
                for ruolo, subset in classification_8a.items()
            },
            "multi_eleggibilita": {
                ruolo: subset["player_name"].head(15).tolist()
                for ruolo, subset in classification_8a2.items()
            },
            "low_cost": classification_8c["player_name"].head(30).tolist(),
            "low_cost_titolari": classification_8d["player_name"].head(30).tolist(),
            "scommesse_multi_ruolo": classification_8e["player_name"].head(20).tolist(),
            "watchlist_giovani": classification_8f["player_name"].head(20).tolist(),
        },
    }

    # Optionally write to disk (and upload to R2, best-effort — see
    # design doc "R2 come source of truth", Fase 2)
    if output_dir:
        output_dir = Path(output_dir)
        from ml.storage.artifact_store import ArtifactStore, R2Config

        store = ArtifactStore(local_dir=output_dir, r2_config=R2Config.from_env())
        out_path = store.save_json(result, f"mantra_results_{season_start}.json")
        log.info("Results written to %s", out_path)

    log.info("MANTRA pipeline complete. %d players scored.", len(players_out))
    return result

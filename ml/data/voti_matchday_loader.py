"""Ingest per-matchday Fantacalcio grades into player_matchday_votes.

Reads the JSON produced by ``voti/scraper.js`` (single giornata or range)
and upserts rows keyed by (fantacalcio_id, season_start, giornata, fonte).

Name → fantacalcio_id resolution reuses the same heuristics as
``ml.data.voti_loader`` (normalise + last-name token + team alias + fuzzy
fallback against the quotations / id-map tables) so we never diverge on
matching rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .import_quotations import (
    apply_team_alias,
    last_name_token,
    normalise_player_name,
    normalise_team,
)

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

ROLE_IT_TO_CANONICAL: dict[str, str] = {
    "Portiere": "GK",
    "Difensore": "DEF",
    "Centrocampista": "MID",
    "Attaccante": "FWD",
}

#: Season label ``2025-26`` → season_start year 2025
_SEASON_RE = re.compile(r"(\d{4})-\d{2}")

#: Bonus/malus title → column counters.
#: Order matters: more specific fragments MUST come before generic ones
#: (e.g. ``gol_subiti`` before ``gol``, otherwise "gol_subiti" matches "gol").
#:
#: Penalties: Fantacalcio stores a converted pen as ``rigori_segnati`` with
#: ``gol_segnati = 0``.  Under classic rules that is still a goal (+3), so we
#: map it onto the ``gol`` counter.
_EVENT_KEY_RULES: tuple[tuple[str, str], ...] = (
    ("gol_subiti", "gol_subiti"),
    ("gol_subito", "gol_subiti"),
    ("gol_segnati", "gol"),
    ("gol_segnato", "gol"),
    ("rigori_segnati", "gol"),       # penalty scored → counts as goal
    ("rigore_segnato", "gol"),
    ("rigore_parato", "rigori_parati"),
    ("rigori_parati", "rigori_parati"),
    ("rigore_sbagliato", "rigori_sbagliati"),
    ("rigori_sbagliati", "rigori_sbagliati"),
    ("ammonizione", "ammonizioni"),
    ("espulsione", "espulsioni"),
    ("assist", "assist"),
    # Generic "gol" / "goal" only after the specific variants above.
    ("gol", "gol"),
    ("goal", "gol"),
)

#: Role-aware allow-lists (Italian role labels from the voti JSON).
_GK_EVENT_COLS = frozenset({"gol_subiti", "rigori_parati", "assist", "ammonizioni", "espulsioni"})
_OUTFIELD_EVENT_COLS = frozenset({
    "gol", "assist", "ammonizioni", "espulsioni", "rigori_sbagliati",
})


def _parse_season_start(label: str) -> int:
    m = _SEASON_RE.search(label)
    if not m:
        raise ValueError(f"Cannot parse season_start from label {label!r}")
    return int(m.group(1))


def _parse_vote(raw: Any, *, kind: str = "voto") -> Optional[float]:
    """Convert scraper grade string / number to float; ``s.v.`` → None.

    Some Fantacalcio payloads encode decimals without a separator
    (``"55"`` / ``55`` meaning ``5.5``).  Heuristic:

    * ``voto`` (0–10 scale): values strictly greater than 10 → ÷10
    * ``fantavoto`` (can exceed 10 with bonuses): values strictly greater
      than 20 → ÷10  (a single-match fantavoto of 21+ is unrealistic)
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
    else:
        s = str(raw).strip().lower().replace(",", ".")
        if s in ("", "s.v.", "sv", "n/a", "-"):
            return None
        try:
            val = float(s)
        except ValueError:
            return None

    if kind == "fantavoto":
        if val > 20:
            val = val / 10.0
    else:
        # raw grade / any other 0–10 style field
        if val > 10:
            val = val / 10.0
    return val


# Classic Fantacalcio point contributions used to recover cards that the
# voti JSON does not expose as icons (only the fantavoto delta remains).
_EVENT_POINTS: dict[str, float] = {
    "gol": 3.0,
    "assist": 1.0,
    "gol_subiti": -1.0,
    "rigori_parati": 3.0,
    "rigori_sbagliati": -3.0,
}


def _infer_cards_from_delta(
    voto: Optional[float],
    fantavoto: Optional[float],
    events: dict[str, int],
) -> dict[str, int]:
    """Fill ammonizioni/espulsioni from fantavoto − voto residual.

    Fantacalcio does not put yellow/red icons in the scraped bonus/malus
    object; the only signal is the score gap, e.g. voto 5.5 / fantavoto 5.0
    with empty bonus/malus → one ammonizione (−0.5).

    Espulsione = −1, ammonizione = −0.5.  We only infer when the JSON did
    not already carry card counts and the residual is negative after
    subtracting known event points (gol, assist, …).
    """
    if voto is None or fantavoto is None:
        return events
    if events.get("ammonizioni", 0) or events.get("espulsioni", 0):
        return events

    known = 0.0
    for key, pts in _EVENT_POINTS.items():
        known += events.get(key, 0) * pts

    residual = fantavoto - voto - known
    # Snap to nearest half-point to absorb float noise
    residual = round(residual * 2.0) / 2.0
    if residual > -0.5 + 1e-9:
        return events

    n_red = 0
    n_yellow = 0
    r = residual
    while r <= -1.0 + 1e-9:
        n_red += 1
        r += 1.0
    while r <= -0.5 + 1e-9:
        n_yellow += 1
        r += 0.5

    out = dict(events)
    out["espulsioni"] = n_red
    out["ammonizioni"] = n_yellow
    return out


def _count_events(
    bonus: dict,
    malus: dict,
    *,
    ruolo: str | None = None,
) -> dict[str, int]:
    """Aggregate bonus/malus icons into counter columns.

    Critical:
    * ``gol_subiti`` must not be counted as ``gol``.
    * ``rigori_segnati`` (penalty scored) counts as ``gol``.
    * Role filter (Italian labels from voti JSON):
        - Portiere → gol_subiti, rigori_parati, assist, ammonizioni, espulsioni
        - others   → gol, assist, cards, rigori_sbagliati
          (never gol_subiti / rigori_parati)
    """
    counts = {
        "gol": 0,
        "assist": 0,
        "ammonizioni": 0,
        "espulsioni": 0,
        "gol_subiti": 0,
        "rigori_parati": 0,
        "rigori_sbagliati": 0,
    }
    is_gk = (ruolo or "").strip().lower() in ("portiere", "por", "gk")
    allowed = _GK_EVENT_COLS if is_gk else _OUTFIELD_EVENT_COLS

    for bucket in (bonus or {}, malus or {}):
        for key, value in bucket.items():
            k = key.lower().replace(" ", "_")
            col: str | None = None
            for frag, target in _EVENT_KEY_RULES:
                if frag not in k:
                    continue
                # Guard: bare "gol"/"goal" must not fire on conceded-goal keys
                if frag in ("gol", "goal") and "subit" in k:
                    continue
                # Guard: bare "gol" inside unrelated "rigore_*" keys other than
                # the explicit ``rigori_segnati`` rule above.
                if frag in ("gol", "goal") and "rigor" in k:
                    continue
                col = target
                break
            if col is None or col not in allowed:
                continue
            try:
                counts[col] += int(value)
            except (TypeError, ValueError):
                counts[col] += 1
    return counts


# ── ID resolution ────────────────────────────────────────────────────────────


def _load_name_index(
    engine: sa.Engine, season_start: int
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    """Build the (name, team) → fantacalcio_id indices used by resolution.

    Returns a pair of dicts:

    1. ``name_index`` — keyed by ``(normalised_full_name, normalised_team)``
       (with a no-team fallback).  Used when the voti side carries the
       same display form the DB stores (rare for matchday voti, which are
       always surname-only, but kept for parity with the listone side).
    2. ``surname_index`` — keyed by ``(last_name_token, normalised_team)``
       (with a no-team fallback).  This is the index that actually
       matches the voti JSON: Fantacalcio's per-matchday reports only
       ship the surname (``"Zemura"``, ``"Buksa"``, ``"Kristensen T."``)
       while ``player_quotations`` and ``player_id_map`` typically store
       the full name (``"Jordan Zemura"``).
    """
    name_index: dict[tuple[str, str], int] = {}
    surname_index: dict[tuple[str, str], int] = {}
    with engine.connect() as conn:
        # quotations are the richest source of display names + team
        try:
            # Schema: player_quotations.player_name (NOT "name") — see models.PlayerQuotation
            rows = conn.execute(
                sa.text(
                    """
                    SELECT fantacalcio_id, player_name, team
                    FROM player_quotations
                    WHERE season_start = :ss
                      AND fantacalcio_id IS NOT NULL
                    """
                ),
                {"ss": season_start},
            ).fetchall()
            n_q = 0
            for fid, name, team in rows:
                nname = normalise_player_name(name or "")
                # normalise FIRST, then alias (apply_team_alias expects folded keys)
                nteam = apply_team_alias(normalise_team(team or ""))
                if not nname:
                    continue
                fid_i = int(fid)
                name_index.setdefault((nname, nteam), fid_i)
                name_index.setdefault((nname, ""), fid_i)
                # listone is "Surname First" ("Carnesecchi Marco")
                token = last_name_token(nname, assume_surname_first=True)
                if token:
                    surname_index.setdefault((token, nteam), fid_i)
                    surname_index.setdefault((token, ""), fid_i)
                n_q += 1
            log.info(
                "player_quotations: indexed %d rows for season_start=%s",
                n_q, season_start,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("player_quotations lookup failed: %s", exc)

        try:
            # Schema: player_id_map.name_fantacalcio / team_fantacalcio
            # (NOT fantacalcio_name) — see models.PlayerIdMap / voti_loader._load_id_map
            rows = conn.execute(
                sa.text(
                    """
                    SELECT fantacalcio_id, name_fantacalcio, team_fantacalcio
                    FROM player_id_map
                    WHERE fantacalcio_id IS NOT NULL
                      AND name_fantacalcio IS NOT NULL
                    """
                )
            ).fetchall()
            n_m = 0
            for fid, name, team in rows:
                nname = normalise_player_name(name or "")
                nteam = apply_team_alias(normalise_team(team or "")) if team else ""
                if not nname:
                    continue
                fid_i = int(fid)
                name_index.setdefault((nname, nteam), fid_i)
                name_index.setdefault((nname, ""), fid_i)
                token = last_name_token(nname, assume_surname_first=True)
                if token:
                    if nteam:
                        surname_index.setdefault((token, nteam), fid_i)
                    surname_index.setdefault((token, ""), fid_i)
                n_m += 1
            log.info("player_id_map: indexed %d rows", n_m)
        except Exception as exc:  # noqa: BLE001
            log.warning("player_id_map lookup failed: %s", exc)

    log.info(
        "Name index: %d full-name keys, %d surname keys",
        len(name_index), len(surname_index),
    )
    if not name_index and not surname_index:
        log.error(
            "Name index is EMPTY — check that player_quotations (season_start=%s) "
            "and/or player_id_map are populated. Backfill will match 0 players.",
            season_start,
        )
    return name_index, surname_index


def resolve_fantacalcio_id(
    nome: str,
    squadra: str,
    name_index: dict[tuple[str, str], int],
    surname_index: dict[tuple[str, str], int] | None = None,
) -> Optional[int]:
    """Resolve a voti ``(nome, squadra)`` to a ``fantacalcio_id``.

    Strategy (first hit wins):

    1. ``(normalised_full_name, normalised_team)``   — strict.
    2. ``(normalised_full_name, "")``                 — same name, any team.
    3. ``(last_name_token, normalised_team)``         — voti has only the
       surname (the standard Fantacalcio per-matchday format).
    4. ``(last_name_token, "")``                       — last resort.
    """
    nname = normalise_player_name(nome or "")
    nteam = apply_team_alias(normalise_team(squadra or ""))
    if not nname:
        return None
    # 1) exact name + team
    fid = name_index.get((nname, nteam))
    if fid is not None:
        return fid
    # 2) exact name only
    fid = name_index.get((nname, ""))
    if fid is not None:
        return fid
    # 3/4) surname-based fallback (the path that actually matches voti)
    if surname_index:
        # voti JSON is always surname-only ("Zemura") or "Surname X."
        # ("Kristensen T.") — for both shapes the leading surviving
        # token after ``_strip_trailing_initial`` IS the surname, so
        # ``assume_surname_first`` is correct here too.
        token = last_name_token(nname, assume_surname_first=True)
        if token:
            fid = surname_index.get((token, nteam))
            if fid is not None:
                return fid
            fid = surname_index.get((token, ""))
            if fid is not None:
                return fid
    return None


# ── Row extraction ───────────────────────────────────────────────────────────


def iter_player_rows(
    payload: list[dict],
    *,
    season_start: int,
    name_index: dict[tuple[str, str], int],
    surname_index: dict[tuple[str, str], int],
) -> Iterable[dict]:
    """Yield dicts ready for upsert into player_matchday_votes."""
    scraped_at = datetime.now(timezone.utc)
    unmatched = 0
    total = 0

    for giornata_block in payload:
        giornata = int(giornata_block["giornata"])
        for squadra_block in giornata_block.get("squadre") or []:
            for g in squadra_block.get("giocatori") or []:
                total += 1
                nome = g.get("nome") or ""
                team = g.get("squadra") or ""
                fid = resolve_fantacalcio_id(
                    nome, team, name_index, surname_index,
                )
                if fid is None:
                    unmatched += 1
                    log.debug("Unmatched player: %s (%s) g%d", nome, team, giornata)
                    continue

                voti = g.get("voti") or {}
                fc = voti.get("fantacalcio") or {}
                st = voti.get("statistico") or {}
                it = voti.get("italia") or {}
                events = _count_events(
                    g.get("bonus") or {}, g.get("malus") or {}, ruolo=g.get("ruolo")
                )
                voto_fc = _parse_vote(fc.get("voto"), kind="voto")
                fantavoto_fc = _parse_vote(fc.get("fantavoto"), kind="fantavoto")
                events = _infer_cards_from_delta(voto_fc, fantavoto_fc, events)

                yield {
                    "fantacalcio_id": fid,
                    "season_start": season_start,
                    "giornata": giornata,
                    "team": team,
                    "ruolo": g.get("ruolo"),
                    "voto_fantacalcio": voto_fc,
                    "fantavoto": fantavoto_fc,
                    "voto_statistico": _parse_vote(st.get("voto"), kind="voto"),
                    "fantavoto_statistico": _parse_vote(st.get("fantavoto"), kind="fantavoto"),
                    "voto_italia": _parse_vote(it.get("voto"), kind="voto"),
                    "fantavoto_italia": _parse_vote(it.get("fantavoto"), kind="fantavoto"),
                    **events,
                    "fonte": "fantacalcio",
                    "scraped_at": scraped_at,
                }

    if total:
        log.info(
            "Resolved %d/%d players (%.1f%% unmatched)",
            total - unmatched,
            total,
            100.0 * unmatched / total,
        )


# ── Upsert ───────────────────────────────────────────────────────────────────

_UPSERT_SQL = """
INSERT INTO player_matchday_votes (
    fantacalcio_id, season_start, giornata, team, ruolo,
    voto_fantacalcio, fantavoto,
    voto_statistico, fantavoto_statistico,
    voto_italia, fantavoto_italia,
    gol, assist, ammonizioni, espulsioni, gol_subiti,
    rigori_parati, rigori_sbagliati,
    fonte, scraped_at
) VALUES (
    :fantacalcio_id, :season_start, :giornata, :team, :ruolo,
    :voto_fantacalcio, :fantavoto,
    :voto_statistico, :fantavoto_statistico,
    :voto_italia, :fantavoto_italia,
    :gol, :assist, :ammonizioni, :espulsioni, :gol_subiti,
    :rigori_parati, :rigori_sbagliati,
    :fonte, :scraped_at
)
ON CONFLICT (fantacalcio_id, season_start, giornata, fonte) DO UPDATE SET
    team = EXCLUDED.team,
    ruolo = EXCLUDED.ruolo,
    voto_fantacalcio = EXCLUDED.voto_fantacalcio,
    fantavoto = EXCLUDED.fantavoto,
    voto_statistico = EXCLUDED.voto_statistico,
    fantavoto_statistico = EXCLUDED.fantavoto_statistico,
    voto_italia = EXCLUDED.voto_italia,
    fantavoto_italia = EXCLUDED.fantavoto_italia,
    gol = EXCLUDED.gol,
    assist = EXCLUDED.assist,
    ammonizioni = EXCLUDED.ammonizioni,
    espulsioni = EXCLUDED.espulsioni,
    gol_subiti = EXCLUDED.gol_subiti,
    rigori_parati = EXCLUDED.rigori_parati,
    rigori_sbagliati = EXCLUDED.rigori_sbagliati,
    scraped_at = EXCLUDED.scraped_at
"""


def upsert_rows(
    engine: sa.Engine,
    rows: Iterable[dict],
    *,
    batch_size: int = 500,
) -> int:
    """Upsert rows in batches; logs progress so long runs do not look hung."""
    # Materialise once so we can report totals up front.
    row_list = list(rows)
    total = len(row_list)
    if total == 0:
        log.info("Nothing to upsert")
        return 0

    log.info("Upserting %d rows (batch_size=%d) …", total, batch_size)
    n = 0
    with engine.begin() as conn:
        for start in range(0, total, batch_size):
            batch = row_list[start : start + batch_size]
            conn.execute(sa.text(_UPSERT_SQL), batch)
            n += len(batch)
            # Progress every batch so remote DB latency is visible
            log.info("  upsert progress: %d / %d (%.0f%%)", n, total, 100.0 * n / total)
    return n


# ── CLI ──────────────────────────────────────────────────────────────────────


def load_json(path: Optional[Path], stdin: bool) -> list[dict]:
    if stdin or path is None:
        raw = sys.stdin.read()
    else:
        raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of giornata objects")
    return data


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to scraper JSON (omit or use - for stdin)",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON from stdin (same as --json -)",
    )
    parser.add_argument(
        "--season",
        required=True,
        help="Season label e.g. 2025-26 (used to derive season_start)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL (default: $DATABASE_URL)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    import os

    db_url = args.database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        log.error("DATABASE_URL not set and --database-url not provided")
        return 2

    season_start = _parse_season_start(args.season)
    payload = load_json(args.json, args.stdin or (args.json is None))
    engine = sa.create_engine(db_url)
    name_index, surname_index = _load_name_index(engine, season_start)
    log.info("Matching players …")
    rows = list(
        iter_player_rows(
            payload,
            season_start=season_start,
            name_index=name_index,
            surname_index=surname_index,
        )
    )
    log.info("Matched rows ready for DB: %d — starting upsert", len(rows))
    n = upsert_rows(engine, rows)
    log.info("Upserted %d rows into player_matchday_votes (season_start=%d)", n, season_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
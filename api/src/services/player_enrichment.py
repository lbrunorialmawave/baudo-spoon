"""Cross-router player-list enrichment — shared by /mantra/players and
/overview/players so both attach the real scraped titolarità (and, for
overview, Gruppo Esperti ratings) the same way, without duplicating the
join logic per router.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)


async def enrich_with_matchday_status(
    db: AsyncSession, players: list[dict]
) -> list[dict]:
    """Add the real scraped titolarità to each MANTRA player.

    Reads ``player_matchday_status`` (populated by the probabili-formazioni
    scraper) for the season of the loaded MANTRA results and the most recent
    available matchday, keyed by ``fantacalcio_id``. Each player gains two
    read-only fields — ``status_scraped`` (starter/bench/doubtful) and
    ``probability_scraped`` (0-100) — that reflect the real line-up picture,
    kept distinct from the ML-derived ``start_probability``. Players without
    a scrape row are left with ``None``; an empty/absent table is a no-op.
    """
    if not players:
        return players

    # Season: prefer the MANTRA results' own season, else the latest quotation.
    season = None
    for p in players:
        if p.get("season_start") is not None:
            season = int(p["season_start"])
            break
    if season is None:
        season = await db.scalar(
            sa.text("SELECT MAX(season_start) FROM player_quotations")
        )
    if season is None:
        return players

    # Most recent matchday present for that season in the scrape table.
    matchday = await db.scalar(
        sa.text(
            "SELECT MAX(matchday) FROM player_matchday_status WHERE season_start = :s"
        ),
        {"s": season},
    )
    if matchday is None:
        return players

    rows = (
        await db.execute(
            sa.text(
                "SELECT fantacalcio_id, status, probability "
                "FROM player_matchday_status "
                "WHERE season_start = :s AND matchday = :m"
            ),
            {"s": season, "m": matchday},
        )
    ).all()
    by_id = {r.fantacalcio_id: r for r in rows}

    out: list[dict] = []
    for p in players:
        record = dict(p)
        row = by_id.get(p.get("fantacalcio_id"))
        if row is not None:
            record["status_scraped"] = row.status
            record["probability_scraped"] = row.probability
        else:
            record["status_scraped"] = None
            record["probability_scraped"] = None
        out.append(record)
    return out


async def enrich_with_expert_ratings(
    db: AsyncSession,
    players: list[dict],
    season_start: int,
    source: str = "gruppo_esperti",
) -> list[dict]:
    """Add Gruppo Esperti ratings to each player, keyed by ``fantacalcio_id``.

    ``expert_ratings.player_id`` uses the scraper's own id space
    (``fc-{fantacalcio_id}``) — same convention as
    ``GET /experts/ratings/for-season/{season}``. A player can have several
    rows (one per expert/matchday); we keep exactly one row per
    fantacalcio_id — the most recent matchday for the season (``NULL``
    last), and the highest ``id`` as a tie-breaker — since it reflects the
    most up-to-date opinion available for that season.

    Adds ``expert_*``-prefixed fields (all ``None`` when no rating exists):
    ``expert_rating``, ``expert_name``, ``expert_comment``,
    ``expert_titolarita``, ``expert_media_voto``, ``expert_salute``,
    ``expert_bonus_label``, ``expert_bonus_value``, ``expert_totale``,
    ``expert_url``, ``expert_matchday``.
    """
    if not players:
        return players

    rows = (
        await db.execute(
            sa.text(
                """
                SELECT DISTINCT ON (fantacalcio_id)
                    fantacalcio_id, rating, expert_name, comment, titolarita,
                    media_voto, salute, bonus_label, bonus_value, totale,
                    url, matchday
                FROM (
                    SELECT *, SUBSTRING(player_id FROM 'fc-(\\d+)')::int AS fantacalcio_id
                    FROM expert_ratings
                    WHERE season_start = :season AND source = :source
                      AND player_id LIKE 'fc-%'
                ) sub
                WHERE fantacalcio_id IS NOT NULL
                ORDER BY fantacalcio_id, matchday DESC NULLS LAST, id DESC
                """
            ),
            {"season": season_start, "source": source},
        )
    ).all()
    by_id = {r.fantacalcio_id: r for r in rows}

    out: list[dict] = []
    for p in players:
        record = dict(p)
        row = by_id.get(p.get("fantacalcio_id"))
        record["expert_rating"] = row.rating if row is not None else None
        record["expert_name"] = row.expert_name if row is not None else None
        record["expert_comment"] = row.comment if row is not None else None
        record["expert_titolarita"] = row.titolarita if row is not None else None
        record["expert_media_voto"] = row.media_voto if row is not None else None
        record["expert_salute"] = row.salute if row is not None else None
        record["expert_bonus_label"] = row.bonus_label if row is not None else None
        record["expert_bonus_value"] = row.bonus_value if row is not None else None
        record["expert_totale"] = row.totale if row is not None else None
        record["expert_url"] = row.url if row is not None else None
        record["expert_matchday"] = row.matchday if row is not None else None
        out.append(record)
    return out

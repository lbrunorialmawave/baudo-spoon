from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from .models import LeagueMeta

log = logging.getLogger(__name__)

_CORE_COLS: frozenset[str] = frozenset(
    {"Date", "Round", "Match", "Score", "Status", "Url", "Team", "Side", "Opponent", "Goal scored", "Goal conceded", "points"}
)


# ──────────────────────────────────────────────────────────────────────────────
# ORM models
# ──────────────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    comp_id: Mapped[str] = mapped_column(String(10), nullable=False)
    slug: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    seasons: Mapped[list[Season]] = relationship("Season", back_populates="league")


class Season(Base):
    __tablename__ = "seasons"
    __table_args__ = (UniqueConstraint("league_id", "season_start", name="uq_season"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False
    )
    season_start: Mapped[int] = mapped_column(Integer, nullable=False)
    season_label: Mapped[str] = mapped_column(String(20), nullable=False)
    scraped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    league: Mapped[League] = relationship("League", back_populates="seasons")
    match_stats: Mapped[list[MatchStat]] = relationship(
        "MatchStat", back_populates="season"
    )
    player_season_stats: Mapped[list[PlayerSeasonStat]] = relationship(
        "PlayerSeasonStat", back_populates="season"
    )
    team_season_stats: Mapped[list[TeamSeasonStat]] = relationship(
        "TeamSeasonStat", back_populates="season"
    )


class MatchStat(Base):
    __tablename__ = "match_stats"
    __table_args__ = (
        UniqueConstraint("season_id", "match_name", "team", name="uq_match_stat"),
        Index("idx_match_stats_match_name", "match_name"),
        Index("idx_match_stats_team", "team"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    season_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False
    )
    match_date: Mapped[str | None] = mapped_column(String(50), nullable=True)
    round_num: Mapped[int | None] = mapped_column(Integer, nullable=True)
    match_name: Mapped[str] = mapped_column(String(200), nullable=False)
    score: Mapped[str | None] = mapped_column(String(20), nullable=True)
    status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    team: Mapped[str] = mapped_column(String(100), nullable=False)
    side: Mapped[str | None] = mapped_column(String(10), nullable=True)
    opponent: Mapped[str | None] = mapped_column(String(100), nullable=True)
    goals_scored: Mapped[int | None] = mapped_column(Integer, nullable=True)
    goals_conceded: Mapped[int | None] = mapped_column(Integer, nullable=True)
    points: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stats: Mapped[dict] = mapped_column(JSONB, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    season: Mapped[Season] = relationship("Season", back_populates="match_stats")


class PlayerSeasonStat(Base):
    __tablename__ = "player_season_stats"
    __table_args__ = (
        UniqueConstraint(
            "season_id", "stat_category", "player_fotmob_id",
            name="uq_player_season_stat",
        ),
        Index("idx_pss_category", "stat_category"),
        Index("idx_pss_player", "player_fotmob_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    season_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False
    )
    fotmob_season_id: Mapped[int] = mapped_column(Integer, nullable=False)
    stat_category: Mapped[str] = mapped_column(String(100), nullable=False)
    rank: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    player_fotmob_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    player_name: Mapped[str] = mapped_column(String(200), nullable=False)
    team_fotmob_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    team_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    value: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    season: Mapped[Season] = relationship("Season", back_populates="player_season_stats")


class TeamSeasonStat(Base):
    __tablename__ = "team_season_stats"
    __table_args__ = (
        UniqueConstraint(
            "season_id", "stat_category", "team_fotmob_id",
            name="uq_team_season_stat",
        ),
        Index("idx_tss_category", "stat_category"),
        Index("idx_tss_team", "team_fotmob_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    season_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("seasons.id", ondelete="CASCADE"), nullable=False
    )
    fotmob_season_id: Mapped[int] = mapped_column(Integer, nullable=False)
    stat_category: Mapped[str] = mapped_column(String(100), nullable=False)
    rank: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    team_fotmob_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    team_name: Mapped[str] = mapped_column(String(200), nullable=False)
    value: Mapped[Decimal | None] = mapped_column(Numeric(12, 3), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    season: Mapped[Season] = relationship("Season", back_populates="team_season_stats")


class PlayerProfile(Base):
    """Player-level role/position data, sourced from FotMob playerData API."""

    __tablename__ = "player_profiles"

    player_fotmob_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    player_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role_key: Mapped[str | None] = mapped_column(String(50), nullable=True)
    canonical_role: Mapped[str | None] = mapped_column(String(5), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ingestion logic
# ──────────────────────────────────────────────────────────────────────────────


def ingest_dataframe(
    session: Session,
    df: pd.DataFrame,
    league_name: str,
    meta: LeagueMeta,
    season_start: int | None,
) -> int:
    """Upsert match stats dataframe into the database. Returns number of rows inserted/updated."""
    if df.empty:
        return 0

    # 1. Upsert League
    stmt_league = pg_insert(League).values(
        name=league_name,
        comp_id=meta.comp_id,
        slug=meta.slug,
    )
    stmt_league = stmt_league.on_conflict_do_update(
        index_elements=["name"],
        set_={"comp_id": meta.comp_id, "slug": meta.slug},
    ).returning(League.id)
    league_id = session.execute(stmt_league).scalar_one()

    # 2. Upsert Season
    actual_season_start = season_start if season_start is not None else 9999
    season_label = "latest" if season_start is None else str(season_start)

    stmt_season = pg_insert(Season).values(
        league_id=league_id,
        season_start=actual_season_start,
        season_label=season_label,
        scraped_at=datetime.now(timezone.utc),
    )
    stmt_season = stmt_season.on_conflict_do_update(
        constraint="uq_season",
        set_={"scraped_at": datetime.now(timezone.utc), "season_label": season_label},
    ).returning(Season.id)
    season_id = session.execute(stmt_season).scalar_one()

    # 3. Prepare match stats payload
    records: list[dict[str, Any]] = []
    
    # Pre-calculate which columns are core vs extra (stats)
    df_cols = set(df.columns)
    extra_cols = df_cols - _CORE_COLS

    for _, row in df.iterrows():
        # Build JSON dict of all extra stats
        stats_dict = {col: row[col] for col in extra_cols if pd.notna(row.get(col))}

        # Some fields might be missing if scraping was partial
        records.append(
            {
                "season_id": season_id,
                "match_date": str(row.get("Date", "")) if pd.notna(row.get("Date")) else None,
                "round_num": int(row.get("Round", 0)) if pd.notna(row.get("Round")) else None,
                "match_name": str(row.get("Match", "")),
                "score": str(row.get("Score", "")) if pd.notna(row.get("Score")) else None,
                "status": str(row.get("Status", "")) if pd.notna(row.get("Status")) else None,
                "url": str(row.get("Url", "")) if pd.notna(row.get("Url")) else None,
                "team": str(row.get("Team", "")),
                "side": str(row.get("Side", "")) if pd.notna(row.get("Side")) else None,
                "opponent": str(row.get("Opponent", "")) if pd.notna(row.get("Opponent")) else None,
                "goals_scored": int(row.get("Goal scored", 0)) if pd.notna(row.get("Goal scored")) else None,
                "goals_conceded": int(row.get("Goal conceded", 0)) if pd.notna(row.get("Goal conceded")) else None,
                "points": int(row.get("points", 0)) if pd.notna(row.get("points")) else None,
                "stats": stats_dict,
                "ingested_at": datetime.now(timezone.utc),
            }
        )

    if not records:
        return 0

    # 4. Upsert Match Stats
    stmt_stats = pg_insert(MatchStat).values(records)
    stmt_stats = stmt_stats.on_conflict_do_update(
        constraint="uq_match_stat",
        set_={
            "match_date": stmt_stats.excluded.match_date,
            "round_num": stmt_stats.excluded.round_num,
            "score": stmt_stats.excluded.score,
            "status": stmt_stats.excluded.status,
            "url": stmt_stats.excluded.url,
            "side": stmt_stats.excluded.side,
            "opponent": stmt_stats.excluded.opponent,
            "goals_scored": stmt_stats.excluded.goals_scored,
            "goals_conceded": stmt_stats.excluded.goals_conceded,
            "points": stmt_stats.excluded.points,
            "stats": stmt_stats.excluded.stats,
            "ingested_at": stmt_stats.excluded.ingested_at,
        },
    )

    result = session.execute(stmt_stats)
    session.commit()
    return result.rowcount


def _upsert_season(
    session: Session,
    league_name: str,
    meta: LeagueMeta,
    season_label: str,
) -> int:
    """Upsert league + season rows and return the season PK."""
    stmt = (
        pg_insert(League)
        .values(name=league_name, comp_id=meta.comp_id, slug=meta.slug)
        .on_conflict_do_update(
            index_elements=["name"],
            set_={"comp_id": meta.comp_id, "slug": meta.slug},
        )
        .returning(League.id)
    )
    league_id: int = session.execute(stmt).scalar_one()

    try:
        season_start = int(season_label.split("-")[0])
    except (ValueError, IndexError):
        season_start = 9999

    stmt2 = (
        pg_insert(Season)
        .values(
            league_id=league_id,
            season_start=season_start,
            season_label=season_label,
            scraped_at=datetime.now(timezone.utc),
        )
        .on_conflict_do_update(
            constraint="uq_season",
            set_={
                "scraped_at": datetime.now(timezone.utc),
                "season_label": season_label,
            },
        )
        .returning(Season.id)
    )
    return session.execute(stmt2).scalar_one()


def upsert_player_profiles(
    session: Session,
    profiles: list[dict[str, Any]],
) -> int:
    """Upsert player role data into player_profiles.

    Each dict in *profiles* must have:
        player_fotmob_id (int), player_name (str),
        role_key (str | None), canonical_role (str | None)

    Returns:
        Number of rows inserted / updated.
    """
    if not profiles:
        return 0

    now = datetime.now(timezone.utc)
    records = [
        {
            "player_fotmob_id": p["player_fotmob_id"],
            "player_name": p["player_name"],
            "role_key": p.get("role_key"),
            "canonical_role": p.get("canonical_role"),
            "updated_at": now,
        }
        for p in profiles
    ]
    stmt = pg_insert(PlayerProfile).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_fotmob_id"],
        set_={
            "player_name": stmt.excluded.player_name,
            "role_key": stmt.excluded.role_key,
            "canonical_role": stmt.excluded.canonical_role,
            "updated_at": stmt.excluded.updated_at,
        },
    )
    result = session.execute(stmt)
    session.commit()
    return result.rowcount


def ingest_league_stats(
    session: Session,
    rows: list[dict[str, Any]],
    league_name: str,
    meta: LeagueMeta,
    season_label: str,
    fotmob_season_id: int,
    stat_type: str,
    stat_category: str,
) -> int:
    """
    Upsert player or team season stats into the database.

    Args:
        stat_type: ``"players"`` or ``"teams"``.
        rows: output of ``FotMobLeagueStatsScraper.run()`` for one category.

    Returns:
        Number of rows inserted / updated.
    """
    if not rows:
        return 0

    season_id = _upsert_season(session, league_name, meta, season_label)
    now = datetime.now(timezone.utc)

    if stat_type == "players":
        records: list[dict[str, Any]] = [
            {
                "season_id": season_id,
                "fotmob_season_id": fotmob_season_id,
                "stat_category": stat_category,
                "rank": r.get("rank"),
                "player_fotmob_id": r["entity_id"],
                "player_name": r["entity_name"],
                "team_fotmob_id": r.get("team_id"),
                "team_name": r.get("team_name") or "",
                "value": r.get("value"),
                "ingested_at": now,
            }
            for r in rows
        ]
        stmt = pg_insert(PlayerSeasonStat).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_player_season_stat",
            set_={
                "rank": stmt.excluded.rank,
                "team_fotmob_id": stmt.excluded.team_fotmob_id,
                "team_name": stmt.excluded.team_name,
                "value": stmt.excluded.value,
                "ingested_at": stmt.excluded.ingested_at,
            },
        )
    else:
        records = [
            {
                "season_id": season_id,
                "fotmob_season_id": fotmob_season_id,
                "stat_category": stat_category,
                "rank": r.get("rank"),
                "team_fotmob_id": r["entity_id"],
                "team_name": r["entity_name"],
                "value": r.get("value"),
                "ingested_at": now,
            }
            for r in rows
        ]
        stmt = pg_insert(TeamSeasonStat).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_team_season_stat",
            set_={
                "rank": stmt.excluded.rank,
                "team_name": stmt.excluded.team_name,
                "value": stmt.excluded.value,
                "ingested_at": stmt.excluded.ingested_at,
            },
        )

    result = session.execute(stmt)
    session.commit()
    return result.rowcount

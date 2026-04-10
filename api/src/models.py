from __future__ import annotations

from datetime import datetime
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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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

from __future__ import annotations

import enum
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
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


# ── Fantacalcio quotations & ID mapping ──────────────────────────────────────

class MatchMethodEnum(str, enum.Enum):
    """Algorithm used to map a Fantacalcio ID to a FotMob player."""
    EXACT_NAME_TEAM = "exact_name_team"
    EXACT_NAME_ROLE = "exact_name_role"
    EXACT_NAME_TEAM_ROLE_SEASON = "exact_name_team_role_season"
    EXACT_RELAXED_ROLE = "exact_relaxed_role"
    FUZZY_NAME = "fuzzy_name"
    MANUAL = "manual"
    UNMATCHED = "unmatched"


class PlayerQuotation(Base):
    """Raw auction valuation for a single Fantacalcio player/season.

    Populated by ``ml.data.import_quotations`` from the
    ``Quotazioni_Fantacalcio_Stagione_YYYY_YY.xlsx`` listoni.
    """
    __tablename__ = "player_quotations"
    __table_args__ = (
        UniqueConstraint("fantacalcio_id", "season_start", name="uq_player_quotation"),
        Index("idx_pq_season", "season_start"),
        Index("idx_pq_role", "role"),
        Index("idx_pq_team", "team"),
        Index("idx_pq_season_role", "season_start", "role"),
        CheckConstraint(
            "role IN ('GK', 'DEF', 'MID', 'FWD')",
            name="player_quotations_role_check",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fantacalcio_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season_start: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(5), nullable=False)
    team: Mapped[str] = mapped_column(String(100), nullable=False)
    player_name: Mapped[str] = mapped_column(String(200), nullable=False)

    qt_a: Mapped[int] = mapped_column(Integer, nullable=False)
    qt_i: Mapped[int] = mapped_column(Integer, nullable=False)
    diff_val: Mapped[int] = mapped_column(Integer, nullable=False)

    qt_a_m: Mapped[int | None] = mapped_column(Integer, nullable=True)
    qt_i_m: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diff_val_m: Mapped[int | None] = mapped_column(Integer, nullable=True)

    fvm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fvm_m: Mapped[int | None] = mapped_column(Integer, nullable=True)

    source: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="listone_fantagazzetta"
    )
    imported_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fantacalcio_id": self.fantacalcio_id,
            "season_start": self.season_start,
            "role": self.role,
            "team": self.team,
            "player_name": self.player_name,
            "qt_a": self.qt_a,
            "qt_i": self.qt_i,
            "diff_val": self.diff_val,
            "qt_a_m": self.qt_a_m,
            "qt_i_m": self.qt_i_m,
            "diff_val_m": self.diff_val_m,
            "fvm": self.fvm,
            "fvm_m": self.fvm_m,
            "source": self.source,
            "imported_at": self.imported_at.isoformat() if self.imported_at else None,
        }


class PlayerIdMap(Base):
    """Bridge table: Fantacalcio ID ↔ FotMob player ID.

    One row per (fantacalcio_id, season_start) — the same Fantacalcio ID
    can be re-assigned across years if the operator corrects a mismatch.
    """
    __tablename__ = "player_id_map"
    __table_args__ = (
        UniqueConstraint("fantacalcio_id", "season_start", name="uq_id_map"),
        Index("idx_pim_fotmob", "player_fotmob_id"),
        Index("idx_pim_season", "season_start"),
        Index("idx_pim_method", "match_method"),
        Index("idx_pim_role", "canonical_role"),
        CheckConstraint(
            "match_method IN ('exact_name_team', 'exact_name_role', "
            "'exact_relaxed_role', 'fuzzy_name', 'manual', 'unmatched')",
            name="player_id_map_match_method_check",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fantacalcio_id: Mapped[int] = mapped_column(Integer, nullable=False)
    season_start: Mapped[int] = mapped_column(Integer, nullable=False)
    player_fotmob_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    name_fantacalcio: Mapped[str] = mapped_column(String(200), nullable=False)
    name_fotmob: Mapped[str | None] = mapped_column(String(200), nullable=True)
    team_fantacalcio: Mapped[str | None] = mapped_column(String(100), nullable=True)
    team_fotmob: Mapped[str | None] = mapped_column(String(100), nullable=True)
    canonical_role: Mapped[str | None] = mapped_column(String(5), nullable=True)
    match_method: Mapped[MatchMethodEnum] = mapped_column(
        Enum(
            MatchMethodEnum,
            name="match_method_enum",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fantacalcio_id": self.fantacalcio_id,
            "season_start": self.season_start,
            "player_fotmob_id": self.player_fotmob_id,
            "name_fantacalcio": self.name_fantacalcio,
            "name_fotmob": self.name_fotmob,
            "team_fantacalcio": self.team_fantacalcio,
            "team_fotmob": self.team_fotmob,
            "canonical_role": self.canonical_role,
            "match_method": self.match_method.value if isinstance(self.match_method, MatchMethodEnum) else self.match_method,
            "confidence": float(self.confidence) if self.confidence is not None else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

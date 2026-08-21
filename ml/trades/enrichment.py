"""Build EnrichedTradePlayer from TradePlayer + pre-fetched DB signal maps.

Keeps SQL out of the pure fairness module; the API layer fetches the maps
and calls ``enrich_players``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ml.trades.advisor import TradePlayer
from ml.trades.fairness import EnrichedTradePlayer
from ml.trades.signals import (
    FormaResult,
    MatchdayVote,
    TitolaritaInputs,
    forma_recente_score,
    indice_titolarita,
    pool_stats,
)

# Classic role from Mantra codes (same mapping as fairness)
_MANTRA_TO_CLASSIC: dict[str, str] = {
    "Por": "GK",
    "Dd": "DEF", "Ds": "DEF", "Dc": "DEF", "B": "DEF",
    "E": "MID", "M": "MID", "C": "MID", "W": "MID", "T": "MID",
    "A": "FWD", "Pc": "FWD",
}

CLASSIC_FROM_LETTER = {"P": "GK", "D": "DEF", "C": "MID", "A": "FWD",
                       "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def classic_role_of(player: TradePlayer, explicit: str | None = None) -> str:
    if explicit and explicit in ("GK", "DEF", "MID", "FWD"):
        return explicit
    for r in player.eligible_roles:
        mapped = _MANTRA_TO_CLASSIC.get(r)
        if mapped:
            return mapped
    return "MID"


def enrich_players(
    players: Sequence[TradePlayer],
    *,
    votes_by_fid: Mapping[int, Sequence[MatchdayVote]],
    status_by_fid: Mapping[int, dict],
    experts_by_fid: Mapping[int, dict],
    classic_role_by_fid: Mapping[int, str] | None = None,
    pool_ewma_by_role: Mapping[str, Sequence[float]] | None = None,
) -> list[EnrichedTradePlayer]:
    """Attach forma / titolarità / status to each TradePlayer."""

    # Pre-compute role-pool stats for z-score normalisation of form
    role_stats: dict[str, tuple[float, float]] = {}
    if pool_ewma_by_role:
        for role, vals in pool_ewma_by_role.items():
            role_stats[role] = pool_stats(list(vals))

    enriched: list[EnrichedTradePlayer] = []
    for p in players:
        try:
            fid = int(p.player_id)
        except (TypeError, ValueError):
            fid = -1

        role = classic_role_of(
            p,
            (classic_role_by_fid or {}).get(fid),
        )

        votes = list(votes_by_fid.get(fid, ()))
        mean, std = role_stats.get(role, (6.0, 0.8))
        forma_res: FormaResult = forma_recente_score(
            votes, pool_mean=mean, pool_std=std
        )

        st = status_by_fid.get(fid) or {}
        ex = experts_by_fid.get(fid) or {}
        tit_res = indice_titolarita(
            TitolaritaInputs(
                probability_matchday=_as_float(st.get("probability")),
                titolarita_esperti=_as_float(ex.get("titolarita")),
                status=str(st.get("status") or "unknown"),
            )
        )

        enriched.append(
            EnrichedTradePlayer(
                player=p,
                forma_recente=forma_res.forma,
                games_available_for_form=forma_res.games_available,
                indice_titolarita=tit_res.indice,
                status=tit_res.status,
                classic_role=role,
            )
        )
    return enriched


def _as_float(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def season_notice_if_cold_start(
    votes_by_fid: Mapping[int, Sequence[MatchdayVote]],
    player_ids: Sequence[str],
) -> Optional[str]:
    """Emit a league-level notice when nobody has matchday grades yet."""
    any_votes = False
    for pid in player_ids:
        try:
            fid = int(pid)
        except (TypeError, ValueError):
            continue
        if votes_by_fid.get(fid):
            any_votes = True
            break
    if any_votes:
        return None
    return (
        "Stagione appena iniziata: nessuna pagella disponibile ancora. "
        "Valutazione basata su valore storico e titolarità; il peso della "
        "forma recente aumenterà progressivamente dopo le prime giornate."
    )

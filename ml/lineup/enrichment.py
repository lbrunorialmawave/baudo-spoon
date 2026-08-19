"""Build matchday EV for lineup candidates from hybrid + matchday status.

Pure functions — no DB.  The API layer loads hybrid predictions and matchday
rows and passes them in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ml.lineup.optimizer import (
    LineupCandidate,
    compute_ev,
    opponent_adjustment,
)
from ml.roster_import.matcher import MatchStatus, MatchedPlayer

# Status values observed in player_matchday_status
_OUT_STATUSES = frozenset({"injured", "suspended", "unavailable", "out"})
_DEFAULT_FP = 6.0
_DEFAULT_SP = 0.55


@dataclass(frozen=True, slots=True)
class MatchdayInfo:
    fantacalcio_id: int
    probability: float
    """Starter probability in [0, 1]."""

    status: str
    """Raw status string (starter, bench, injured, …)."""

    team: str = ""
    opponent_team: str = ""
    """Real Serie A opponent for this matchday when known."""


@dataclass(frozen=True, slots=True)
class HybridInfo:
    fantacalcio_id: int
    fp_ibrido_voto: float
    """Vote-scale score ~4–10 (or normalised equivalent)."""

    fp_corr: float | None = None
    confidence: float | None = None
    source: str = "hybrid"


@dataclass(frozen=True, slots=True)
class EnrichmentStats:
    total: int
    with_hybrid: int
    with_matchday: int
    excluded_out: int
    baseline_fallback: int


def _status_probability(info: MatchdayInfo | None) -> tuple[float, bool, str]:
    """Return (starter_prob, is_out, note).

    Injured / suspended are hard-excluded (prob=0, is_out=True).
    """
    if info is None:
        return _DEFAULT_SP, False, "no matchday data — default SP"

    status = (info.status or "").strip().lower()
    if status in _OUT_STATUSES:
        return 0.0, True, f"status={status} — excluded"

    prob = float(info.probability)
    # Some sources store 0–100
    if prob > 1.0:
        prob = prob / 100.0
    prob = max(0.0, min(1.0, prob))
    return prob, False, f"status={status or 'n/a'} SP={prob:.2f}"


def _fp_from_hybrid(info: HybridInfo | None) -> tuple[float, str]:
    if info is None:
        return _DEFAULT_FP, "baseline FP (no hybrid)"
    fp = float(info.fp_ibrido_voto)
    # Guard: if someone stored 0–100 scale, compress to ~4–10
    if fp > 12:
        fp = 4.0 + (fp / 100.0) * 6.0
    fp = max(4.0, min(10.0, fp))
    return fp, f"FP_Ibrido={fp:.2f} ({info.source})"


def enrich_matched_players(
    players: Sequence[MatchedPlayer],
    *,
    hybrid_by_fid: Mapping[int, HybridInfo] | None = None,
    matchday_by_fid: Mapping[int, MatchdayInfo] | None = None,
    opponent_strength_by_team: Mapping[str, float] | None = None,
    k_att: float = 0.30,
    k_def: float = 0.20,
) -> tuple[list[LineupCandidate], EnrichmentStats]:
    """Convert matched roster players into EV-enriched lineup candidates.

    Players that are unmatched or hard-out (injured/suspended) are skipped.
    """
    hybrid_by_fid = hybrid_by_fid or {}
    matchday_by_fid = matchday_by_fid or {}
    opponent_strength_by_team = opponent_strength_by_team or {}

    candidates: list[LineupCandidate] = []
    with_hybrid = with_matchday = excluded_out = baseline = 0

    for mp in players:
        if mp.status == MatchStatus.UNMATCHED or mp.catalog is None:
            continue

        fid = int(mp.catalog.fantacalcio_id)
        roles = mp.catalog.roles_mantra
        if not roles:
            classic = (mp.catalog.role_classic or "").upper()
            mapping = {"P": ("Por",), "D": ("Dc",), "C": ("C",), "A": ("A",)}
            roles = mapping.get(classic, ())
        if not roles:
            continue
        roles_fs = frozenset(roles)

        h = hybrid_by_fid.get(fid)
        m = matchday_by_fid.get(fid)

        fp, fp_note = _fp_from_hybrid(h)
        if h is not None:
            with_hybrid += 1
        else:
            baseline += 1

        sp, is_out, sp_note = _status_probability(m)
        if m is not None:
            with_matchday += 1
        if is_out:
            excluded_out += 1
            continue

        # Opponent adjustment from real Serie A opponent when known
        primary_role = next(iter(roles_fs))
        opp_team = (m.opponent_team if m else "") or ""
        opp_strength = opponent_strength_by_team.get(opp_team.lower(), 0.5)
        adj = opponent_adjustment(
            primary_role, opp_strength, k_att=k_att, k_def=k_def
        )

        ev = compute_ev(
            fp_ibrido_voto=fp,
            starter_probability=sp,
            opponent_adjustment=adj,
        )
        note = (
            f"{fp_note} × {sp_note} × adj={adj:.3f}"
            + (f" (vs {opp_team})" if opp_team else "")
        )

        candidates.append(
            LineupCandidate(
                player_id=str(fid),
                name=mp.catalog.name or mp.parsed.name_clean,
                eligible_roles=roles_fs,
                expected_value=ev,
                starter_probability=sp,
                cost=mp.parsed.cost,
                team_serie_a=mp.catalog.team or "",
                breakdown_note=note,
            )
        )

    stats = EnrichmentStats(
        total=len(candidates),
        with_hybrid=with_hybrid,
        with_matchday=with_matchday,
        excluded_out=excluded_out,
        baseline_fallback=baseline,
    )
    return candidates, stats


def parse_hybrid_rows(rows: Sequence[dict]) -> dict[int, HybridInfo]:
    """Index hybrid prediction dicts by fantacalcio_id.

    Accepts common key variants produced by ``get_hybrid_predictions`` /
    mantra_ibrido artefacts.
    """
    out: dict[int, HybridInfo] = {}
    for r in rows:
        fid = r.get("fantacalcio_id") or r.get("fantacalcioId") or r.get("id")
        if fid is None:
            continue
        try:
            fid_i = int(fid)
        except (TypeError, ValueError):
            continue

        # Prefer vote-scale fields; fall back to 0-100 FP scaled later
        fp = (
            r.get("fp_ibrido_voto")
            or r.get("FP_Ibrido_voto")
            or r.get("fp_ibrido")
            or r.get("FP_Ibrido")
            or r.get("predicted_fantavoto")
            or r.get("voto")
        )
        if fp is None:
            continue
        try:
            fp_f = float(fp)
        except (TypeError, ValueError):
            continue

        fp_corr = r.get("fp_corr") or r.get("FP_Corr")
        conf = r.get("confidence_score") or r.get("Confidence_Score") or r.get("confidence")

        out[fid_i] = HybridInfo(
            fantacalcio_id=fid_i,
            fp_ibrido_voto=fp_f,
            fp_corr=float(fp_corr) if fp_corr is not None else None,
            confidence=float(conf) if conf is not None else None,
            source="hybrid",
        )
    return out


def parse_matchday_rows(rows: Sequence[dict]) -> dict[int, MatchdayInfo]:
    """Index matchday status rows by fantacalcio_id."""
    out: dict[int, MatchdayInfo] = {}
    for r in rows:
        fid = r.get("fantacalcio_id") or r.get("fantacalcioId")
        if fid is None:
            continue
        try:
            fid_i = int(fid)
        except (TypeError, ValueError):
            continue
        prob = r.get("probability")
        if prob is None:
            prob = 0.5
        status = str(r.get("status") or "")
        team = str(r.get("team") or "")
        opponent = str(
            r.get("opponent")
            or r.get("opponent_team")
            or r.get("avversario")
            or ""
        )
        try:
            prob_f = float(prob)
        except (TypeError, ValueError):
            prob_f = 0.5
        out[fid_i] = MatchdayInfo(
            fantacalcio_id=fid_i,
            probability=prob_f,
            status=status,
            team=team,
            opponent_team=opponent,
        )
    return out

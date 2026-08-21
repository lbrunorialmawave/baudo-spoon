"""Build matchday EV for lineup candidates from hybrid + matchday status + form.

Pure functions — no DB.  The API layer loads hybrid predictions, matchday
status and ``player_matchday_votes`` rows and passes them in.

Form blend (pre-match):
    FP_eff = (1-λ) * FP_Ibrido + λ * EWMA(fantavoto)
    where votes are restricted to giornate **strictly before** the target
    matchday so a leaked/partial post-match vote never influences the XI.
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
from ml.trades.signals import (
    DEFAULT_EWMA_LAMBDA,
    DEFAULT_FORM_WINDOW,
    MatchdayVote,
    ewma_fantavoto,
)

# Status values observed in player_matchday_status
_OUT_STATUSES = frozenset({"injured", "suspended", "unavailable", "out"})
_DEFAULT_FP = 6.0
_DEFAULT_SP = 0.55

# Conservative blend weights — hybrid remains the primary signal.
# λ grows only when enough pre-match samples exist.
_FORM_WEIGHT_BY_GAMES: tuple[tuple[int, float], ...] = (
    (1, 0.15),  # 1–2 games
    (3, 0.25),  # 3–4 games
    (5, 0.35),  # 5+ games
)


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
    with_form: int
    """Players where form EWMA contributed (λ > 0)."""

    excluded_out: int
    baseline_fallback: int


def form_blend_weight(games_available: int) -> float:
    """Return λ ∈ [0, 1] given the number of usable pre-match votes."""
    if games_available <= 0:
        return 0.0
    weight = 0.15
    for min_games, w in _FORM_WEIGHT_BY_GAMES:
        if games_available >= min_games:
            weight = w
    return weight


def filter_votes_pre_match(
    votes: Sequence[MatchdayVote],
    *,
    target_matchday: int | None,
) -> list[MatchdayVote]:
    """Keep only votes strictly before the target giornata (pre-match).

    When ``target_matchday`` is None the filter is a no-op: the caller has
    no resolved giornata, so we cannot safely drop a "current" round.
    """
    if target_matchday is None:
        return list(votes)
    target = int(target_matchday)
    return [v for v in votes if v.giornata < target]


def blend_fp_with_form(
    fp_hybrid: float,
    *,
    form_ewma: float | None,
    games_available: int,
) -> tuple[float, float, str]:
    """Blend hybrid FP with form EWMA on the same ~4–10 scale.

    Returns ``(fp_effective, lambda_used, note)``.
    """
    lam = form_blend_weight(games_available)
    if form_ewma is None or lam <= 0.0:
        return float(fp_hybrid), 0.0, "no form blend"

    form_clamped = max(4.0, min(10.0, float(form_ewma)))
    fp_eff = (1.0 - lam) * float(fp_hybrid) + lam * form_clamped
    note = f"form blend λ={lam:.2f} EWMA={form_clamped:.2f} (n={games_available})"
    return fp_eff, lam, note


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
    votes_by_fid: Mapping[int, Sequence[MatchdayVote]] | None = None,
    target_matchday: int | None = None,
    opponent_strength_by_team: Mapping[str, float] | None = None,
    k_att: float = 0.30,
    k_def: float = 0.20,
    form_window: int = DEFAULT_FORM_WINDOW,
    form_ewma_lambda: float = DEFAULT_EWMA_LAMBDA,
) -> tuple[list[LineupCandidate], EnrichmentStats]:
    """Convert matched roster players into EV-enriched lineup candidates.

    Players that are unmatched or hard-out (injured/suspended) are skipped.

    Form uses only pre-match votes (``giornata < target_matchday``) when the
    target giornata is known, then blends conservatively into FP_Ibrido.
    """
    hybrid_by_fid = hybrid_by_fid or {}
    matchday_by_fid = matchday_by_fid or {}
    votes_by_fid = votes_by_fid or {}
    opponent_strength_by_team = opponent_strength_by_team or {}

    candidates: list[LineupCandidate] = []
    with_hybrid = with_matchday = with_form = excluded_out = baseline = 0

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

        # Pre-match form from historical fantavoto
        raw_votes = votes_by_fid.get(fid) or ()
        pre_votes = filter_votes_pre_match(
            raw_votes, target_matchday=target_matchday
        )
        form_ewma, games = ewma_fantavoto(
            pre_votes, lam=form_ewma_lambda, window=form_window
        )
        fp_eff, form_lam, form_note = blend_fp_with_form(
            fp, form_ewma=form_ewma, games_available=games
        )
        if form_lam > 0.0:
            with_form += 1

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
            fp_ibrido_voto=fp_eff,
            starter_probability=sp,
            opponent_adjustment=adj,
        )
        note_parts = [fp_note]
        if form_lam > 0.0:
            note_parts.append(form_note)
        note_parts.append(sp_note)
        note_parts.append(f"adj={adj:.3f}")
        if opp_team:
            note_parts.append(f"vs {opp_team}")
        note = " × ".join(note_parts)

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
        with_form=with_form,
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

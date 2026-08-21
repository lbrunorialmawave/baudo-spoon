"""Backtest the Trade Fairness Engine on historical matchday grades.

Uses only the voti JSON (no Postgres / hybrid model). Approximates the
structural base from the pre-cutoff season mean fantavoto so the test is
reproducible offline.

Protocol
--------
1. Load ``voti/voti_fantacalcio-{season}.json``.
2. At each cut-point giornata C (default: 10, 15, 19, 25):
   a. Build per-player pre-C form (EWMA) and pre-C mean (base proxy).
   b. Sample random same-role 1-for-1 pairs among players with ≥3 pre-C games.
   c. Run ``player_trade_value`` + verdict logic (tolerance band).
   d. Score against the *actual* post-C mean fantavoto delta:
        realized_delta% = (mean_recv_post - mean_give_post) / mean_give_post * 100
      A verdict is a *hit* when:
        - vantaggioso  and realized_delta > +tolerance
        - sfavorevole  and realized_delta < -tolerance
        - equilibrato  and |realized_delta| ≤ 2 * tolerance
3. Report hit-rate, confusion-style counts, and suggested tolerance.

Usage
-----
    PYTHONPATH=. python -m ml.trades.backtest_fairness \\
        --voti voti/voti_fantacalcio-2025-26.json \\
        --cuts 10,15,19,25 \\
        --pairs 400 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

# ── Lightweight copies of signal/PTV logic (avoid heavy package imports) ─────

ROLE_MAP = {
    "Portiere": "GK",
    "Difensore": "DEF",
    "Centrocampista": "MID",
    "Attaccante": "FWD",
}


def _parse_vote(raw) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower().replace(",", ".")
    if s in ("", "s.v.", "sv", "n/a", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def ewma(values: list[float], lam: float = 0.65) -> Optional[float]:
    if not values:
        return None
    # values newest-first
    num = den = 0.0
    w = 1.0
    for v in values:
        num += w * v
        den += w
        w *= lam
    return num / den


def fantavoto_to_100(fv: float, mean: float = 6.0, std: float = 0.85) -> float:
    if std > 1e-6:
        z = (fv - mean) / std
        return max(0.0, min(100.0, 50.0 + z * 15.0))
    return max(0.0, min(100.0, (fv - 4.0) / 6.0 * 100.0))


@dataclass
class PlayerSeries:
    name: str
    role: str  # GK/DEF/MID/FWD
    # giornata -> fantavoto (only played games)
    by_giornata: dict[int, float]

    def votes_up_to(self, cut: int) -> list[tuple[int, float]]:
        return sorted(
            ((g, v) for g, v in self.by_giornata.items() if g <= cut),
            key=lambda x: x[0],
        )

    def votes_after(self, cut: int) -> list[tuple[int, float]]:
        return sorted(
            ((g, v) for g, v in self.by_giornata.items() if g > cut),
            key=lambda x: x[0],
        )


def load_player_series(path: Path) -> list[PlayerSeries]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # name+role -> giornata -> list of fantavoto (multi-team / duplicates)
    acc: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for block in raw:
        giornata = int(block["giornata"])
        for squadra in block.get("squadre") or []:
            for g in squadra.get("giocatori") or []:
                nome = (g.get("nome") or "").strip()
                ruolo_it = g.get("ruolo") or ""
                role = ROLE_MAP.get(ruolo_it, "MID")
                fc = (g.get("voti") or {}).get("fantacalcio") or {}
                fv = _parse_vote(fc.get("fantavoto"))
                if fv is None or not nome:
                    continue
                acc[(nome, role)][giornata].append(fv)

    players: list[PlayerSeries] = []
    for (nome, role), by_g in acc.items():
        series = {g: statistics.fmean(vs) for g, vs in by_g.items()}
        players.append(PlayerSeries(name=nome, role=role, by_giornata=series))
    return players


@dataclass
class Snapshot:
    name: str
    role: str
    base_100: float
    forma_100: Optional[float]
    games: int
    post_mean: Optional[float]
    post_games: int


def build_snapshots(
    players: list[PlayerSeries],
    cut: int,
    *,
    min_pre: int = 3,
    min_post: int = 3,
) -> list[Snapshot]:
    # role pool stats from pre-cut means
    pre_means_by_role: dict[str, list[float]] = defaultdict(list)
    pre_data: list[tuple[PlayerSeries, list[float], list[float]]] = []

    for p in players:
        pre = p.votes_up_to(cut)
        post = p.votes_after(cut)
        if len(pre) < min_pre or len(post) < min_post:
            continue
        pre_vals = [v for _, v in pre]
        post_vals = [v for _, v in post]
        pre_means_by_role[p.role].append(statistics.fmean(pre_vals))
        pre_data.append((p, pre_vals, post_vals))

    role_stats = {
        role: (
            statistics.fmean(vals),
            statistics.pstdev(vals) if len(vals) > 1 else 0.85,
        )
        for role, vals in pre_means_by_role.items()
    }

    out: list[Snapshot] = []
    for p, pre_vals, post_vals in pre_data:
        mean_r, std_r = role_stats.get(p.role, (6.0, 0.85))
        # newest-first for EWMA
        pre_sorted = sorted(
            ((g, v) for g, v in p.by_giornata.items() if g <= cut),
            key=lambda x: x[0],
            reverse=True,
        )
        recent = [v for _, v in pre_sorted[:5]]
        ewma_raw = ewma(recent)
        forma = (
            fantavoto_to_100(ewma_raw, mean_r, std_r) if ewma_raw is not None else None
        )
        base = fantavoto_to_100(statistics.fmean(pre_vals), mean_r, std_r)
        out.append(
            Snapshot(
                name=p.name,
                role=p.role,
                base_100=base,
                forma_100=forma,
                games=len(pre_vals),
                post_mean=statistics.fmean(post_vals),
                post_games=len(post_vals),
            )
        )
    return out


def ptv(
    base: float,
    forma: Optional[float],
    games: int,
    tit: float = 55.0,
    *,
    w_base: float = 0.55,
    w_forma: float = 0.25,
    w_tit: float = 0.20,
) -> float:
    ramp = min(games / 5.0, 1.0) if forma is not None else 0.0
    pb = w_base + w_forma * (1.0 - ramp)
    pf = w_forma * ramp
    if forma is None or games == 0:
        denom = pb + w_tit
        return (base * pb + tit * w_tit) / denom if denom else base
    return base * pb + forma * pf + tit * w_tit


def verdict(delta_pct: float, tol: float) -> str:
    if abs(delta_pct) <= tol:
        return "equilibrato"
    return "vantaggioso" if delta_pct > 0 else "sfavorevole"


def realized_delta_pct(give: Snapshot, recv: Snapshot) -> Optional[float]:
    if give.post_mean is None or recv.post_mean is None:
        return None
    if give.post_mean <= 0:
        return None
    return (recv.post_mean - give.post_mean) / give.post_mean * 100.0


def is_hit(pred: str, realized: float, tol: float) -> bool:
    if pred == "vantaggioso":
        return realized > tol
    if pred == "sfavorevole":
        return realized < -tol
    # equilibrato: allow a wider band (2x) — noise in residual season
    return abs(realized) <= 2 * tol


def run_cut(
    snaps: list[Snapshot],
    *,
    n_pairs: int,
    tol: float,
    seed: int,
    weights: tuple[float, float, float],
) -> dict:
    by_role: dict[str, list[Snapshot]] = defaultdict(list)
    for s in snaps:
        by_role[s.role].append(s)

    rng = random.Random(seed)
    pairs: list[tuple[Snapshot, Snapshot]] = []
    for role, pool in by_role.items():
        if len(pool) < 2:
            continue
        # sample without replacement pairs
        idxs = list(range(len(pool)))
        rng.shuffle(idxs)
        # generate up to n_pairs // 4 per role
        target = max(1, n_pairs // 4)
        count = 0
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                a, b = pool[idxs[i]], pool[idxs[j]]
                # skip near-identical bases (noise)
                if abs(a.base_100 - b.base_100) < 1.0 and abs(
                    (a.forma_100 or 50) - (b.forma_100 or 50)
                ) < 1.0:
                    continue
                pairs.append((a, b))
                count += 1
                if count >= target:
                    break
            if count >= target:
                break

    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]

    w_base, w_forma, w_tit = weights
    counts = {"vantaggioso": 0, "equilibrato": 0, "sfavorevole": 0}
    hits = 0
    total = 0
    abs_err = []
    sign_agree = 0

    for give, recv in pairs:
        score_g = ptv(
            give.base_100, give.forma_100, give.games,
            w_base=w_base, w_forma=w_forma, w_tit=w_tit,
        )
        score_r = ptv(
            recv.base_100, recv.forma_100, recv.games,
            w_base=w_base, w_forma=w_forma, w_tit=w_tit,
        )
        if score_g <= 0:
            continue
        delta = (score_r - score_g) / score_g * 100.0
        pred = verdict(delta, tol)
        real = realized_delta_pct(give, recv)
        if real is None:
            continue
        counts[pred] += 1
        total += 1
        if is_hit(pred, real, tol):
            hits += 1
        abs_err.append(abs(delta - real))
        if (delta > tol and real > 0) or (delta < -tol and real < 0) or (
            abs(delta) <= tol and abs(real) <= 2 * tol
        ):
            sign_agree += 1

    return {
        "n_pairs": total,
        "hit_rate": hits / total if total else 0.0,
        "sign_agree": sign_agree / total if total else 0.0,
        "mae_delta": statistics.fmean(abs_err) if abs_err else None,
        "pred_counts": counts,
        "n_eligible_players": len(snaps),
    }


def sweep_tolerance(
    snaps: list[Snapshot],
    *,
    n_pairs: int,
    seed: int,
    weights: tuple[float, float, float],
    grid: Iterable[float] = (4, 6, 8, 10, 12, 15),
) -> list[dict]:
    rows = []
    for tol in grid:
        r = run_cut(snaps, n_pairs=n_pairs, tol=tol, seed=seed, weights=weights)
        rows.append({"tolerance": tol, **r})
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voti",
        type=Path,
        default=Path("voti/voti_fantacalcio-2025-26.json"),
        help="Path to season voti JSON",
    )
    parser.add_argument(
        "--cuts",
        default="10,15,19,25",
        help="Comma-separated cut giornate",
    )
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=8.0)
    parser.add_argument(
        "--weights",
        default="0.55,0.25,0.20",
        help="base,forma,titolarita weights",
    )
    parser.add_argument("--sweep-tol", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.voti.exists():
        log.error("File not found: %s", args.voti)
        return 2

    weights = tuple(float(x) for x in args.weights.split(","))
    if len(weights) != 3:
        log.error("Need 3 weights")
        return 2
    cuts = [int(x) for x in args.cuts.split(",") if x.strip()]

    log.info("Loading %s …", args.voti)
    players = load_player_series(args.voti)
    log.info("Players with ≥1 vote: %d", len(players))

    print(f"\n{'='*64}")
    print(f" Backtest Fairness — {args.voti.name}")
    print(f" weights base/forma/tit={weights}  pairs/cut={args.pairs}  seed={args.seed}")
    print(f"{'='*64}\n")

    summary = []
    for cut in cuts:
        snaps = build_snapshots(players, cut)
        if args.sweep_tol:
            rows = sweep_tolerance(
                snaps, n_pairs=args.pairs, seed=args.seed + cut, weights=weights
            )
            print(f"── Cut giornata {cut}  (eligible players={len(snaps)}) ──")
            print(f"{'tol':>5}  {'pairs':>6}  {'hit%':>7}  {'sign%':>7}  {'MAEΔ':>7}  preds")
            best = None
            for r in rows:
                mae = f"{r['mae_delta']:.1f}" if r["mae_delta"] is not None else "n/a"
                print(
                    f"{r['tolerance']:5.0f}  {r['n_pairs']:6d}  "
                    f"{100*r['hit_rate']:6.1f}%  {100*r['sign_agree']:6.1f}%  "
                    f"{mae:>7}  {r['pred_counts']}"
                )
                if best is None or r["hit_rate"] > best["hit_rate"]:
                    best = r
            if best:
                summary.append({"cut": cut, **best})
            print()
        else:
            r = run_cut(
                snaps,
                n_pairs=args.pairs,
                tol=args.tol,
                seed=args.seed + cut,
                weights=weights,
            )
            mae = f"{r['mae_delta']:.1f}" if r["mae_delta"] is not None else "n/a"
            print(
                f"Cut g{cut:02d}: eligible={r['n_eligible_players']:4d}  "
                f"pairs={r['n_pairs']:4d}  hit={100*r['hit_rate']:5.1f}%  "
                f"sign={100*r['sign_agree']:5.1f}%  MAEΔ={mae}  "
                f"preds={r['pred_counts']}"
            )
            summary.append({"cut": cut, "tolerance": args.tol, **r})

    if summary:
        avg_hit = statistics.fmean(s["hit_rate"] for s in summary)
        avg_sign = statistics.fmean(s["sign_agree"] for s in summary)
        print(f"\nMedia hit-rate: {100*avg_hit:.1f}%   media sign-agree: {100*avg_sign:.1f}%")
        # suggest tolerance from best average if sweep was on
        if args.sweep_tol:
            by_tol: dict[float, list[float]] = defaultdict(list)
            # re-sweep aggregate — already stored best per cut only; print guidance
            print(
                "\nSuggerimento: scegli la tolleranza con hit-rate più alta e "
                "sign-agree ≥ 55%. Default di progetto resta 8% se il gap è <2pp."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

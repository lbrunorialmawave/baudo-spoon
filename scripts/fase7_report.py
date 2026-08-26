#!/usr/bin/env python3
"""One-shot sanity-check report for the Fase7 v2 classification (two axes).

Loads real player data for a season, runs the MANTRA scoring pillars +
``classify_fase7``, and prints label counts (globally and per role pool)
for both axes, plus a sample of players in each label — for manual review
before trusting the new gap-based rules on a live season. Does not write
anywhere and is not wired into ``run_mantra`` / ``season_refresh.py``.

Usage:
  python scripts/fase7_report.py --season 2026 [--db-url ...] [--sample 5]
"""
from __future__ import annotations

import argparse
import os
import sys

import sqlalchemy as sa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.mantra.config import MantraConfig
from ml.mantra.fase7 import classify_fase7
from ml.mantra.pilastro1 import compute_p1
from ml.mantra.pilastro2 import compute_p2
from ml.mantra.pilastro3 import compute_p3, compute_ps_corretto
from ml.mantra.pilastro4 import compute_cp, compute_p4
from ml.mantra.runner import _attach_fase7_external_signals, load_data
from ml.mantra.scoring import compute_fp, compute_fp_corr


def _compute_scores(df, cfg: MantraConfig):
    p1 = compute_p1(df, cfg)
    p2 = compute_p2(df, cfg)

    team_cols = ["team_rank_norm", "prev_season_points", "goal_difference", "avg_team_rating", "season_start"]
    team_df = df[["team"] + [c for c in team_cols if c in df.columns]].drop_duplicates(subset="team").copy()
    squad_value = df.groupby("team")["Pz1"].sum().reset_index()
    team_df = team_df.merge(squad_value, on="team", how="left").rename(columns={"Pz1": "squad_value_market"})
    ps_corretto = compute_ps_corretto(team_df, cfg)
    ps_map = dict(zip(team_df["team"], ps_corretto))
    player_ps = df["team"].map(ps_map).fillna(50.0)
    p3 = compute_p3(df, player_ps, cfg)

    p4 = compute_p4(df, p1, p2, p3, cfg)
    cp = compute_cp(p1, p2, p3)
    fp = compute_fp(p1, p2, p3, p4, cfg)
    scores = compute_fp_corr(fp, cp, df["ruolo_primario"], df["num_ruoli"], df["Pz1"], cfg)
    return fp, p1, scores


def _print_axis_report(title: str, label: "pd.Series", roles: "pd.Series") -> None:
    print(f"\n=== {title} ===")
    counts = label.fillna("(nessuna etichetta)").value_counts()
    print(counts.to_string())
    print("\nPer ruolo:")
    cross = label.fillna("(nessuna etichetta)").groupby(roles).value_counts().unstack(fill_value=0)
    print(cross.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True, help="season_start, e.g. 2026")
    parser.add_argument("--db-url", default=os.environ.get("ML_DATABASE_URL"))
    parser.add_argument("--sample", type=int, default=5, help="players to print per label")
    args = parser.parse_args()

    if not args.db_url:
        print("Pass --db-url or set ML_DATABASE_URL", file=sys.stderr)
        sys.exit(1)

    engine = sa.create_engine(args.db_url, pool_pre_ping=True)
    cfg = MantraConfig()

    df = load_data(engine, args.season)
    df = _attach_fase7_external_signals(df, engine, args.season, None, cfg)
    fp, p1, scores = _compute_scores(df, cfg)

    (
        label_rend, _motivo_rend, gap_rend,
        label_prezzo, _motivo_prezzo, gap_prezzo,
    ) = classify_fase7(df, fp, scores["fp_mantra"], scores["vr"], p1, cfg)

    _print_axis_report("Asse Rendimento/Affidabilità", label_rend, df["ruolo_primario"])
    _print_axis_report("Asse Prezzo/Valore", label_prezzo, df["ruolo_primario"])

    for axis_name, label, gap in (
        ("Rendimento", label_rend, gap_rend),
        ("Prezzo", label_prezzo, gap_prezzo),
    ):
        print(f"\n=== Campioni per etichetta ({axis_name}) ===")
        for value in [v for v in label.dropna().unique()]:
            mask = label == value
            sample = df.loc[mask, ["player_name", "team", "ruolo_primario"]].copy()
            sample["gap"] = gap[mask].round(1)
            print(f"\n-- {value} ({int(mask.sum())} giocatori) --")
            print(sample.head(args.sample).to_string(index=False))


if __name__ == "__main__":
    main()

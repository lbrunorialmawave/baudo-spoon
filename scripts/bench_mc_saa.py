#!/usr/bin/env python3
"""Spike: SAA wall-time vs N on a synthetic CLASSIC pool.

Usage:
  PYTHONPATH=. python scripts/bench_mc_saa.py
  PYTHONPATH=. python scripts/bench_mc_saa.py --n 10,25,50,100
"""
from __future__ import annotations

import argparse
import json
import time

from ml.optimizer.models import Formation, OptimizationConfig, Player
from ml.optimizer.monte_carlo_opt import MonteCarloOptConfig, run_saa_frequency
from ml.optimizer.strategies import default_strategies


def make_pool(n_extra: int = 0) -> list[Player]:
    teams = [f"T{i}" for i in range(20)]
    pool: list[Player] = []
    for i in range(4):
        pool.append(Player(player_id=f"P{i}", name=f"P{i}", role="P", real_team=teams[i], cost=5+i, projected_score=6.0+0.1*i, prediction_std=0.25))
    for i in range(10 + n_extra // 3):
        pool.append(Player(player_id=f"D{i}", name=f"D{i}", role="D", real_team=teams[i % 20], cost=8+i%5, projected_score=6.2+0.04*i, prediction_std=0.25))
    for i in range(10 + n_extra // 3):
        pool.append(Player(player_id=f"C{i}", name=f"C{i}", role="C", real_team=teams[i % 20], cost=10+i%5, projected_score=6.5+0.04*i, prediction_std=0.25))
    for i in range(8 + n_extra // 3):
        pool.append(Player(player_id=f"A{i}", name=f"A{i}", role="A", real_team=teams[i % 20], cost=15+i%5, projected_score=7.0+0.05*i, prediction_std=0.3))
    return pool


def cfg() -> OptimizationConfig:
    return OptimizationConfig(
        budget=500,
        formations=[
            Formation(label="4-3-3", defenders=4, midfielders=3, forwards=3),
            Formation(label="3-5-2", defenders=3, midfielders=5, forwards=2),
        ],
        num_participants=8, min_distinct_teams=8, max_players_per_team=5,
        big_teams=frozenset(), big_teams_cap=25,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="5,10,25,50", help="comma-separated N list")
    ap.add_argument("--pool-extra", type=int, default=0)
    args = ap.parse_args()
    ns = [int(x) for x in args.n.split(",") if x.strip()]
    pool = make_pool(args.pool_extra)
    strategy = default_strategies()[0]
    config = cfg()
    results = []
    print(f"pool_size={len(pool)} strategy={strategy.name}")
    for n in ns:
        mc = MonteCarloOptConfig(enabled=True, n_simulations=n, mode="saa_frequency", random_seed=42)
        t0 = time.perf_counter()
        saa = run_saa_frequency(pool, config, strategy, mc)
        elapsed = time.perf_counter() - t0
        row = {
            "n": n,
            "wall_s": round(elapsed, 3),
            "scenarios_completed": saa.scenarios_completed,
            "stability_index": round(saa.stability_index, 4),
            "per_scenario_s": round(elapsed / max(saa.scenarios_completed, 1), 4),
        }
        results.append(row)
        print(json.dumps(row))
    print("--- summary ---")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

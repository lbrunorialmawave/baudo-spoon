"""Quick test: run hybrid merger and print stats."""
from ml.mantra_ibrido.runner import run_hybrid_computation
from pathlib import Path

a = Path("artifacts")
r = run_hybrid_computation(
    mantra_path=a / "mantra_results_2025.json",
    ml_path=a / "results_latest.json",
    output_dir=a,
)

total = len(r["players"])
with_ml = sum(1 for p in r["players"] if p.get("has_ml_data"))
without = sum(1 for p in r["players"] if not p.get("has_ml_data"))

print(f"Total: {total}")
print(f"With ML: {with_ml}")
print(f"Without ML: {without}")

matched = [p for p in r["players"] if p.get("has_ml_data")]
for p in matched[:10]:
    print(f'  {p["player_name"]:30s} ({p["team"]:15s}) pred={p.get("predicted_fantavoto")}, fpI={p.get("fpIbrido")}')

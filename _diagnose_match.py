"""Diagnose why the backfill is 100% unmatched."""
import os
import sys
sys.path.insert(0, r'c:\Users\L.Brunori\Documents\Progetti\personal\AAAAA\baudo-spoon')

db_url = os.environ.get('DATABASE_URL') or os.environ.get('ML_DATABASE_URL')
if not db_url:
    print("ERROR: no DATABASE_URL/ML_DATABASE_URL set")
    sys.exit(1)

import sqlalchemy as sa
from ml.data.import_quotations import (
    normalise_player_name,
    normalise_team,
    apply_team_alias,
)

engine = sa.create_engine(db_url)
with engine.connect() as conn:
    print("=" * 60)
    print("player_quotations for season_start=2025:")
    n = conn.execute(sa.text(
        "SELECT COUNT(*) FROM player_quotations WHERE season_start=2025"
    )).scalar()
    print(f"  count = {n}")
    if n:
        rows = conn.execute(sa.text(
            "SELECT fantacalcio_id, name, team FROM player_quotations "
            "WHERE season_start=2025 ORDER BY name LIMIT 5"
        )).fetchall()
        for r in rows:
            print(f"    {r}")
    print()
    print("player_id_map count:")
    n2 = conn.execute(sa.text(
        "SELECT COUNT(*) FROM player_id_map WHERE season_start=2025"
    )).scalar()
    print(f"  count = {n2}")
    if n2:
        rows = conn.execute(sa.text(
            "SELECT fantacalcio_id, fantacalcio_name, name_fotmob "
            "FROM player_id_map WHERE season_start=2025 "
            "ORDER BY fantacalcio_name LIMIT 5"
        )).fetchall()
        for r in rows:
            print(f"    {r}")
    print()
    # Search for the specific players that failed
    print("Searching for matching players in player_quotations (season_start=2025):")
    for nm in ['Zemura', 'Solet', 'Buksa', 'Montipò', 'Kristensen']:
        rows = conn.execute(sa.text(
            "SELECT fantacalcio_id, name, team FROM player_quotations "
            "WHERE season_start=2025 AND "
            "(LOWER(name) LIKE :q OR LOWER(name) LIKE :q2) "
            "ORDER BY name"
        ), {"q": f"%{nm.lower()}%", "q2": f"%{nm.lower().split()[0]}%"}).fetchall()
        print(f"  {nm}: {len(rows)} matches")
        for r in rows[:3]:
            print(f"    {r}")

import json
d = json.load(open(r'c:\Users\L.Brunori\Documents\Progetti\personal\AAAAA\baudo-spoon\voti\voti_fantacalcio-2025-26.json', encoding='utf-8'))
# Find Zemura and Kristensen T. in g38
g38 = d[-1]
for sq in g38.get('squadre', []):
    for p in sq.get('giocatori', []):
        if 'zemura' in (p.get('nome') or '').lower() or 'kristensen' in (p.get('nome') or '').lower():
            print(p)
            break
print()
# Now test the normalize function
import sys
sys.path.insert(0, r'c:\Users\L.Brunori\Documents\Progetti\personal\AAAAA\baudo-spoon')
from ml.data.import_quotations import (
    normalise_player_name,
    normalise_team,
    apply_team_alias,
    last_name_token,
)
for nm in ['Zemura', 'Kristensen T.', 'Buksa', 'Solet', 'Montipò']:
    n = normalise_player_name(nm)
    print(f"  {nm!r} -> {n!r} (last_name_token={last_name_token(n)!r})")
print()
for t in ['Udinese', 'Verona', 'Atalanta', 'Bologna']:
    n = normalise_team(t)
    a = apply_team_alias(n)
    print(f"  {t!r} -> {n!r} -> {a!r}")

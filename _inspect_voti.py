import json
d = json.load(open(r'c:\Users\L.Brunori\Documents\Progetti\personal\AAAAA\baudo-spoon\voti\voti_fantacalcio-2025-26.json', encoding='utf-8'))
print('giornate:', len(d))
print('g0 keys:', list(d[0].keys()))
print('squadre[0] keys:', list(d[0]['squadre'][0].keys()))
print('first 3 players:')
for p in d[0]['squadre'][0]['giocatori'][:3]:
    print(' ', p.get('nome'), '|', p.get('squadra'), '|', p.get('ruolo'))
print()
# Sample of g38 (last giornata)
print('g38 sample:')
g38 = d[-1]
print('giornata:', g38.get('giornata'))
for sq in g38.get('squadre', [])[:2]:
    print(' team:', sq.get('squadra'))
    for p in sq.get('giocatori', [])[:3]:
        print('  ', p.get('nome'), '|', p.get('squadra'), '|', p.get('ruolo'))

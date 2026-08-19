# La Mia Squadra — modulo runtime-only

Gestione rosa post-asta, ottimizzazione formazione giornata e cruscotto scambi.

## Vincolo architetturale

```
Upload Excel → Parse → Match → RosterContext (memoria) → Lineup / Trades → risposta
```

- **Nessuna** tabella rosa utente/avversari in PostgreSQL
- **Nessun** Redis dedicato alle rose
- Se il client perde il `contextId` (TTL store process-local) → ri-upload Excel

## Backend

| Area | Path |
|------|------|
| Parser rose | `ml/roster_import/parser.py` |
| Matching nomi | `ml/roster_import/matcher.py` + `ml/data/name_matching.py` |
| Context + store TTL | `ml/roster_import/context.py`, `store.py` |
| Lineup (Hungarian) | `ml/lineup/optimizer.py` |
| EV hybrid/matchday | `ml/lineup/enrichment.py` |
| Scambi + penalità | `ml/trades/advisor.py`, `credit_penalty.py` |

### API

| Metodo | Endpoint |
|--------|----------|
| `POST` | `/api/v1/roster/import` |
| `POST` | `/api/v1/roster/claim` |
| `GET` | `/api/v1/roster/context/{id}/team` |
| `POST` | `/api/v1/lineup/optimize` |
| `POST` | `/api/v1/trades/dashboard` |
| `POST` | `/api/v1/trades/execute` |
| `POST` | `/api/v1/trades/credit-penalty/preview` |

Avversario e controparte scambio: **solo stessa divisione** del file importato.

## Frontend

Route: `/my-team` (nav **La Mia Squadra**).

1. Modalità Mantra  
2. Upload Excel rose  
3. Card selezione squadra  
4. Workspace  
   - **Formazione**: optimize, campo SVG, swap manuale titolare↔panchina  
   - **Scambi**: coverage, out/in, execute con penalità opzionale  

File principali: `frontend/src/app/features/my-team/`,  
`core/services/my-team.service.ts`, `core/models/my-team.models.ts`.

## Test

```bash
PYTHONPATH=. pytest ml/roster_import ml/lineup ml/trades -q
# include contract formazioni FE↔BE:
PYTHONPATH=. pytest ml/lineup/tests/test_formation_catalog_contract.py -q
```

Vitest (quando disponibile in CI frontend):

```bash
npx vitest run src/app/features/my-team/lineup-swap.spec.ts
```

## Fixture

`ml/roster_import/tests/fixture_lido_di_ostia.xlsx` — 3 divisioni (A vuota, B/C popolate).

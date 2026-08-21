# Trade Fairness Engine — Rollout checklist

Stato implementazione: codice pronto in working tree.  
Ordine consigliato di merge/deploy.

## 0. Prerequisiti

- [ ] Branch basato su `main` aggiornato
- [ ] Secrets CI: `ML_DATABASE_URL` (già usato da season_refresh) oppure `DATABASE_URL`
- [ ] Repo variables (per schedule `voti-refresh`): `CURRENT_SEASON`, `CURRENT_GIORNATA`
- [ ] Accesso admin al DB di staging/prod

## 1. Database

```bash
# Applica la migration 030
psql "$ML_DATABASE_URL" -f db/migrations/030_add_player_matchday_votes.sql
```

Verifica:

```sql
\d player_matchday_votes
SELECT COUNT(*) FROM player_matchday_votes;  -- atteso 0 prima del backfill
```

## 2. Backfill storico (una tantum)

Con i JSON già in `voti/`:

```bash
export DATABASE_URL="$ML_DATABASE_URL"
export PYTHONPATH=.
python scripts/backfill_matchday_votes.py \
  --voti-dir voti \
  --seasons 2023-24,2024-25,2025-26 \
  -v
```

O solo la stagione corrente:

```bash
python scripts/backfill_matchday_votes.py --seasons 2025-26 -v
```

Verifica:

```sql
SELECT season_start, COUNT(*), MIN(giornata), MAX(giornata)
FROM player_matchday_votes
GROUP BY 1 ORDER BY 1;
```

## 3. Deploy API

File toccati lato backend:

- `db/migrations/030_add_player_matchday_votes.sql`
- `ml/data/voti_matchday_loader.py`
- `ml/trades/fairness.py`, `signals.py`, `enrichment.py`
- `ml/trades/tests/test_fairness.py`, `test_signals.py`
- `api/src/routers/trades.py` → `POST /trades/evaluate`
- `voti/scraper.js` (CLI incrementale)
- `.github/workflows/voti-refresh.yml`

Dopo il deploy, smoke:

```bash
export API_URL=https://<host>
export API_KEY=<key>
# contextId da /roster/import + claim
python scripts/smoke_trades_evaluate.py \
  --context-id <uuid> \
  --sheet "<divisione>" \
  --team "<squadra>" \
  --give <id> --receive <id> \
  --mode mantra
```

Checklist risposta:

- [ ] `valid` / `verdict` presenti
- [ ] `give[].ptv` e `confidence` popolati
- [ ] Con tabella voti vuota: `seasonNotice` valorizzato, confidence bassa/assente
- [ ] Classic con ruoli incrociati → `valid=false` + `validationErrors`

## 4. Deploy frontend

File:

- `frontend/src/app/core/models/my-team.models.ts`
- `frontend/src/app/core/services/my-team.service.ts`
- `frontend/src/app/features/my-team/trade-evaluator/trade-evaluator.component.ts`
- `frontend/src/app/features/my-team/my-team.component.ts`

QA manuale in **La Mia Squadra → Scambi**:

- [ ] Comparsa blocco “Valuta scambio”
- [ ] Toggle Classic/Mantra
- [ ] Selezione Cedo/Ricevo → Verdetto semaforico
- [ ] Badge confidence / flag visibili
- [ ] Warning copertura modulo sopra il verdetto (Mantra)
- [ ] Tolleranza default 10%

## 5. Workflow incrementale

1. Imposta `CURRENT_SEASON` (es. `2025-26`) e `CURRENT_GIORNATA` (ultima disputata).
2. Abilita `.github/workflows/voti-refresh.yml`.
3. Prova manuale: *Actions → voti-refresh → Run workflow*.
4. Dopo il job: `SELECT MAX(giornata) FROM player_matchday_votes WHERE season_start = 2025;`

## 6. Monitoraggio post-go-live

- Log API su `/trades/evaluate` (latenza, 4xx su giocatori non risolti).
- A fine stagione: confrontare verdetti vs rendimento reale (stesso spirito di `backtest_fairness.py`, ma con `FP_Corr` hybrid).
- Non alzare i pesi forma oltre 0.25 finché non c’è un backtest con hybrid reale (vedi `BACKTEST_NOTES.md`).

## 7. Rollback

| Layer | Azione |
|-------|--------|
| Frontend | Rimuovere `<app-trade-evaluator>` / revert component |
| API | Revert `trades.py`; l’endpoint sparisce, dashboard/execute restano |
| DB | `DROP TABLE IF EXISTS player_matchday_votes;` (solo se necessario) |
| Workflow | Disabilitare `voti-refresh` |

Nessuna scrittura sulla rosa: `/evaluate` è read-only; rollback sicuro.

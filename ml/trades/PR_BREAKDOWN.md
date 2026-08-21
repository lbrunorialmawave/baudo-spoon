# PR / Issue breakdown — Motore Valutazione Scambi

Suggerimento di spezzatura in PR ordinate (ogni PR mergeabile da sola).

---

### PR1 — Schema + scraper incrementale + loader
**Scope:** solo pipeline dati forma recente.

| Path | Azione |
|------|--------|
| `db/migrations/030_add_player_matchday_votes.sql` | add |
| `voti/scraper.js` | mod (CLI `--giornata`, `--output=stdout`) |
| `ml/data/voti_matchday_loader.py` | add |
| `scripts/backfill_matchday_votes.py` | add |
| `.github/workflows/voti-refresh.yml` | add |

**Test:** unit parsing su fixture JSON piccola (opzionale); dry-run loader su staging dopo migration.

---

### PR2 — Fairness engine puro (no API)
**Scope:** logica Classic/Mantra + segnali + test.

| Path | Azione |
|------|--------|
| `ml/trades/fairness.py` | add |
| `ml/trades/signals.py` | add |
| `ml/trades/enrichment.py` | add |
| `ml/trades/tests/test_fairness.py` | add |
| `ml/trades/tests/test_signals.py` | add |
| `ml/trades/__init__.py` | mod (export) |
| `ml/trades/backtest_fairness.py` | add |
| `ml/trades/BACKTEST_NOTES.md` | add |

**Test:** `pytest ml/trades/tests/`

---

### PR3 — Endpoint `POST /trades/evaluate`
**Scope:** wiring API + query segnali.

| Path | Azione |
|------|--------|
| `api/src/routers/trades.py` | mod |
| `scripts/smoke_trades_evaluate.py` | add |

**Dipende da:** PR1 (tabella, anche vuota) + PR2.  
**Test:** smoke su staging con context roster reale.

---

### PR4 — Frontend trade-evaluator
**Scope:** UI nel tab Scambi.

| Path | Azione |
|------|--------|
| `frontend/.../my-team.models.ts` | mod |
| `frontend/.../my-team.service.ts` | mod |
| `frontend/.../trade-evaluator/trade-evaluator.component.ts` | add |
| `frontend/.../my-team.component.ts` | mod |

**Dipende da:** PR3.  
**Test:** QA manuale checklist in `ROLLOUT.md` §4.

---

### Docs (con PR1 o PR finale)
- `ml/trades/ROLLOUT.md`
- `ml/trades/PR_BREAKDOWN.md` (questo file)

---

## Defaults calibrati (2025-26 backtest)

| Parametro | Valore |
|-----------|--------|
| Pesi PTV base / forma / titolarità | 0.55 / 0.25 / 0.20 |
| `tolerance_percent` | **10** |
| EWMA λ | 0.65 |
| Finestra forma | 5 giornate |
| Titolarità | 0.6×prob_matchday + 0.4×esperti×10 |

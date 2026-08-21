# Backtest Fairness Engine — stagione 2025-26

Script: `ml/trades/backtest_fairness.py`  
Dati: `voti/voti_fantacalcio-2025-26.json` (38 giornate, ~586 giocatori con almeno 1 voto)

## Protocollo

1. A ogni cut-point C ∈ {10, 15, 19, 25} si costruisce uno snapshot per giocatore con:
   - **base proxy** = media fantavoto pre-C (z-score → 0–100 sul pool ruolo)
   - **forma** = EWMA λ=0.65 delle ultime ≤5 pagelle pre-C
   - **titolarità** fissa a 55 (nessun dato matchday_status offline)
2. Si campionano coppie 1-per-1 stesso ruolo (≥3 partite pre e post).
3. Il verdetto PTV viene confrontato con il delta % della media fantavoto **post-C**.

> **Limite importante:** la base strutturale reale (`FP_Corr` hybrid multi-stagione) non è disponibile offline. Qui la base è la media pre-cut della *stessa* serie usata per la forma, quindi il segnale forma è meno indipendente di quanto sarà in produzione. I numeri sotto sono un lower-bound di utilità, non una stima del motore completo.

## Risultati (weights 0.55 / 0.25 / 0.20, 500 coppie/cut)

| Cut | tol=8% hit | tol=8% sign | tol=10% hit | tol=12% hit |
|-----|-----------|-------------|-------------|-------------|
| g10 | 38.6%     | 47.8%       | 40.8%       | 42.4%       |
| g15 | 58.4%     | 65.6%       | 61.8%       | 64.0%       |
| g19 | 55.4%     | 59.6%       | 56.8%       | 62.4%       |
| g25 | 64.0%     | 69.2%       | 62.6%       | 64.4%       |

- Prima della giornata 15 il segnale è rumoroso → coerente col cold-start progressivo già implementato.
- Dalla giornata 15 in poi **sign-agree ≈ 60–70%** a tol 8–10%.

## Sensitivity pesi (tol=10%, cut 15/19/25, media)

| base / forma / tit | hit-rate medio | sign-agree |
|--------------------|----------------|------------|
| 0.70 / 0.10 / 0.20 | **63.1%**      | **68.8%**  |
| 0.55 / 0.25 / 0.20 | 60.8%          | 67.2%      |
| 0.45 / 0.35 / 0.20 | 60.8%          | 67.2%      |
| 0.40 / 0.40 / 0.20 | 60.3%          | 66.8%      |

Il leggero vantaggio di `0.70/0.10/0.20` è atteso nel proxy offline (base e forma non sono ortogonali). **In produzione si mantiene il default del piano `0.55/0.25/0.20`** perché `FP_Corr` porta informazione multi-stagione distinta dalla forma recente.

## Raccomandazioni operative

1. **`tolerance_percent` default: 10%** (invece di 8%) — migliore bilanciamento hit/sign dalla g15 in poi, senza collassare tutto su “equilibrato”.
2. **Pesi PTV: lasciare 0.55 / 0.25 / 0.20** fino a un backtest con `FP_Corr` reale da hybrid artifact.
3. **UI:** mostrare confidenza forma e `seasonNotice`; sotto g15 il motore deve restare prudente (già coperto dal ramp `games/5`).
4. Rieseguire il backtest dopo il primo mese di 2026-27 con voti live + hybrid per validare i pesi.

## Comando

```bash
python ml/trades/backtest_fairness.py \
  --voti voti/voti_fantacalcio-2025-26.json \
  --cuts 10,15,19,25 \
  --pairs 500 --seed 42 --sweep-tol \
  --weights 0.55,0.25,0.20
```

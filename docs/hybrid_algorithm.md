# Algoritmo Ibrido MANTRA+ML — Documentazione Completa

## Architettura

```
┌─────────────────────────────────────────────────────────────────┐
│                    ml/mantra_ibrido/runner.py                   │
│  orchestrazione: carica mantra_results → merge → score → label │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐    ┌──────────────────────────────────┐
│   merger.py          │    │        scoring.py                 │
│  merge_datasets()    │───▶│  compute_hybrid_scores()         │
│  Allinea MANTRA e ML │    │  Calcola fpIbrido, ml_norm,      │
│  per player_name     │    │  confidence, mlBoost, fpGap, EV  │
└──────────────────────┘    └──────────────┬───────────────────┘
                                           ▼
                               ┌──────────────────────────┐
                               │   classifications.py      │
                               │  compute_hybrid_classif() │
                               │  Assegna le 8 label       │
                               └──────────────────────────┘
```

---

## 1. Pipeline MANTRA (`ml/mantra/`)

### 1.1 Pilastro 1 — Voto Storico (`pilastro1.py`)
Peso nell'FP: **25%**

Calcola la media ponderata del fantavoto storico del giocatore.

```
P1 = media_ponderata(voti_storici, minuti)
```

- Usa i voti `Fantacalcio` storici (stagioni precedenti)
- Peso proporzionale ai minuti giocati
- **Fallback**: neo-arrivi → 2000 minuti fittizi; mancanti → mediana del ruolo → 0

### 1.2 Pilastro 2 — Trend Recente (`pilastro2.py`)
Peso nell'FP: **25%**

Misura la **tendenza** del rendimento recente rispetto alla carriera.

```
P2 = 50 + z_score(media_recente - media_storica) * 10
```

- Confronta gli ultimi N voti con la media storica
- Pools: per ruolo allargato (es. Dc+Dd+Ds per "Difensori centrali")
- **Fallback**: pool < 2 giocatori → P2 = 50 (neutro)

### 1.3 Pilastro 3 — Qualità Squadra (`pilastro3.py`)
Peso nell'FP: **30%**

Stima l'impatto del **contesto squadra** sul rendimento del giocatore.

```
P3 = team_skill * coefficiente_ruolo
```

- `team_skill`: media P1+P2 dei compagni di squadra (normalizzata 0-100)
- `coefficiente_ruolo`: quanto il ruolo beneficia della qualità squadra
  - Attaccanti (A/Pc): coeff ≈ 0.010 (dipendono molto dai compagni)
  - Difensori/Portieri: coeff ≈ 0.001 (meno dipendenti)
- **Fallback**: team_skill → 50 (neutro), coeff → 0.003

### 1.4 Pilastro 4 — Continuità (`pilastro4.py`)
Peso nell'FP: **20%**

Premia la **regolarità di rendimento** e penalizza l'incostanza.

```
P4 = clip(cp / cp_max_ruolo * 100, 0, 100)
```

- `CP` (Continuità Ponderata): `P1*0.25 + P2*0.25 + P3*0.30 + Pz1*0.10 + Pz2*0.10`
  - `Pz1`: partite giocate su totale disponibili
  - `Pz2`: % partite con voto ≥ 6
- `cp_max_pool`: massimo CP nel ruolo → normalizza a 100
- **Fallback**: Pz/Pz2 nulli → 0; cp_max_pool vuoto → 1.0

### 1.5 Scoring MANTRA (`ml/mantra/scoring.py`)

```
FP      = P1 * 0.25 + P2 * 0.25 + P3 * 0.30 + P4 * 0.20
CP      = P1 * 0.25 + P2 * 0.25 + P3 * 0.30 + Pz1 * 0.10 + Pz2 * 0.10

FP_Corr = clip(50 + 50 * tanh(z_score(FP, ruolo) * k / 10), 0, 100)
CP_Corr = clip(50 + z_score(CP, ruolo) * 10, 5, 100)

FP_Mantra = clip(FP_Corr * flessibilità, 0, 100)
  dove flessibilità = 1.00 (1 ruolo), 1.05 (2 ruoli), 1.08 (3+ ruoli)

Fattore_Eroe = clip(1 + (1 - CP / CP_medio_tutti) * 0.5, 0.6, 1.6)

VR = clip(FP_Mantra * Fattore_Eroe / CP_Corr * 100, 0, 300)

Prezzo_Massimo = max(Pz1, 1)   # fallback: media Pz1 del pool esteso di ruolo se mai quotato (Pz1<=0)
Percentile_Ruolo = rank_percentile(FP_Mantra, pool esteso di ruolo)   # 0=peggiore, 1=migliore
```

**Nota**: `z_score` con mean/std calcolati sul pool esteso del ruolo.
`k` = clip(1 / %(giocatori con |z|>1.5), 1, CAP_K=6). Più il ruolo è eterogeneo, più k è alto → i valori vengono "spalmati".

**Nota su Prezzo_Massimo**: ancorato al prezzo reale del giocatore stesso
(`Pz1`), non a VR/CP né alla media di ruolo. Un'ancora basata sulla media
del pool di ruolo (versione precedente) collassa gli outlier di ruolo — es.
un centrocampista offensivo taggato con un ruolo MANTRA difensivo economico
— al prezzo medio dei suoi compagni di ruolo, molto più economici. La media
di ruolo resta solo un fallback per i giocatori mai quotati (Pz1≤0, neo
arrivi). `Percentile_Ruolo` (rank di `FP_Mantra` nel pool esteso) è esposto
per uso downstream ma non modifica `Prezzo_Massimo` in pipeline.

**Stima d'asta (opzionale, calcolata a runtime, non in pipeline)**: l'API
`GET /mantra/players` accetta i parametri opzionali `stima_asta` e
`num_partecipanti` per sovrascrivere `Prezzo_Massimo` con una proiezione
che aumenta il prezzo sopra al listino in base al `Percentile_Ruolo` del
giocatore e al numero di partecipanti alla lega, riusando
`ml.optimizer.inflation.inflation_multiplier` — lo stesso modello già usato
dall'ottimizzatore rose e dal tracker d'asta live (`ml/auction/`). Senza
questi parametri, `Prezzo_Massimo` resta la quotazione reale.

---

## 2. Pipeline Ibrida (`ml/mantra_ibrido/`)

### 2.1 Merger (`merger.py`)

Allinea i due dataset su `player_name`:
1. Carica `mantra_results_{season}.json` (≈ 450 giocatori)
2. Carica `results_latest.json` (ML predictions, ≈ 500+ giocatori)
3. **Left join** su `player_name`: tutti i giocatori MANTRA vengono mantenuti
4. Chi ha match ML → `has_ml_data = True`
5. Chi non ha match ML → `has_ml_data = False` (solo dati MANTRA)

### 2.2 Scoring Ibrido (`scoring.py`)

#### ML_score_norm
Normalizza la prediction ML (scala 4-9) in 0-100:
```
ml_norm = clip((predicted - 4.0) / 5.0 * 100, 0, 100)
```

#### FP_Ibrido
Media ponderata tra voto MANTRA storico e prediction ML:
```
fpIbrido = FP_Corr * PESO_MANTRA + ml_norm * PESO_ML
```
Default: `PESO_MANTRA = 0.5`, `PESO_ML = 0.5` (configurabile)

#### Confidence Score (0-100)
Quanto è affidabile la stima:
```
std_term = 1 / (1 + prediction_std)
min_term = min(expected_minutes / 2700, 1)

confidence = (std_term * 0.6 + min_term * 0.4) * 100
```
- `std_term`: basso se prediction_std è alto (modello insicuro). Nota: `prediction_std` è la deviazione standard tra le predizioni di tutti i modelli ML (cross-model std), tipicamente ~0.03-0.1
- `min_term`: basso se il giocatore gioca pochi minuti. `expected_minutes` deriva da `mins_played` nei dati di test del trainer
- **Valori osservati**: P50=58.3, range [50-60] (con expected_minutes=0). Con expected_minutes popolato, range previsto ~[30-100]

#### ML_Boost (0-100)
Quanto la prediction ML è sopra la media del ruolo (z-score):
```
z = (predicted - media_ruolo) / max(std_ruolo, 0.01)
mlBoost = clip(50 + z * 15, 0, 100)
```
- **50** = prediction nella media del ruolo
- **65** = ~1 deviazione standard sopra la media
- **80** = ~2 deviazioni standard sopra la media
- **Valori osservati**: P50=46.9, P90=68.7, max=100

#### FP_Gap
Differenza tra MANTRA e ML (entrambi in scala 0-100):
```
fpGap = FP_Corr - ml_norm
```
- **Positivo** = MANTRA > ML (giocatore con voti alti ma prediction cauta)
- **Negativo** = ML > MANTRA (ML vede potenziale che MANTRA non coglie)
- **Valori osservati**: P50=+7.8, range [-42, +48]

#### Expected_Value (Punti Stagione Attesi)
```
fpIbrido_voto = 4.0 + (fpIbrido / 100) * 6.0    # 0-100 → 4-10
expectedValue = fpIbrido_voto * (expected_min / 90) * EV_SCALE_FACTOR
```
Stima dei punti fantacalcio attesi per l'intera stagione.

### 2.3 Classificazioni (`classifications.py`) — Le 8 Label

| Label | Condizione | Significato | Count (n=351) |
|---|---|---|---|
| **ML_Confirmed** | pred>6.5 AND conf≥57 AND min>1500 | Prediction affidabile, giocatore sicuro | ~0 |
| **ML_Risky** | conf < 50 | Prediction poco affidabile | ~0 |
| **ML_Top** | pred≥6.7 AND mlBoost≥65 | Giocatore top riconosciuto dal ML | ~TBD |
| **ML_Boosted** | mlBoost>70 AND FP_Corr<60 | Sorpresa nascosta: ML alto ma MANTRA medio | ~TBD |
| **Contradiction** | \|fpGap\| > 30 | Disaccordo forte MANTRA vs ML | ~TBD |
| **Minutes_Risk** | expected_min < 900 | Rischio minutaggio | ~0 |
| **Best_Value** | VR≥110 AND fpIbrido≥50 | Ottimo rapporto qualità/prezzo | ~TBD |
| **Sleeper** | FP_Corr<30 AND ml_norm>45 | Sottovalutato dal MANTRA | ~TBD |

**Importante**: le label **non sono mutuamente esclusive**. Un giocatore può avere più label (es. ML_Top + Best_Value).

### 2.4 Config (`config.py`)

Parametri configurabili via admin panel (`/predictions/hybrid/config`):

| Parametro | Default | Descrizione |
|---|---|---|
| PESO_MANTRA | 0.5 | Peso MANTRA in FP_Ibrido |
| PESO_ML | 0.5 | Peso ML in FP_Ibrido |
| W_PREDICTION_STD | 0.6 | Peso prediction_std in Confidence |
| W_MINUTES | 0.4 | Peso expected_minutes in Confidence |
| EV_SCALE_FACTOR | 1.0 | Moltiplicatore Expected_Value |
| CONFIDENZA_SOGLIA | 57.0 | Min confidence per ML_Confirmed |
| ML_BOOST_SOGLIA | 70.0 | Min mlBoost per ML_Boosted |
| ML_BOOST_FP_CORR_MAX | 60.0 | Max FP_Corr per ML_Boosted |
| ML_TOP_PRED_MIN | 6.7 | Min predicted per ML_Top |
| ML_TOP_BOOST_MIN | 65.0 | Min mlBoost per ML_Top |
| SOGLIA_GAP_ALERT | 30.0 | Min |gap| per Contradiction |
| SLEEPER_FP_CORR_MAX | 30.0 | Max FP_Corr per Sleeper |
| SLEEPER_ML_NORM_MIN | 45.0 | Min ml_norm per Sleeper |
| BEST_VALUE_VR_MIN | 110.0 | Min VR per Best_Value |
| BEST_VALUE_FP_IBRIDO_MIN | 50.0 | Min fpIbrido per Best_Value |
| MINUTES_RISK_MAX | 900.0 | Max expected_min per Minutes_Risk |

---

## 3. Flusso completo dei dati

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATI DI INPUT                                 │
├──────────────┬───────────────────────────┬───────────────────────────┤
│ MANTRA       │ ML Pipeline               │ DB PostgreSQL             │
│ (scraper)    │ (trainer.py)              │ (storico)                 │
│              │                           │                           │
│ Voti storici │ predicted_fantavoto       │ player_season_stats       │
│ Minuti gioc. │ prediction_std            │ player_profiles           │
│ Squadra      │ expected_minutes          │ player_season_roles       │
│ Ruolo MANTRA │ role_mean/std             │ player_quotations         │
│              │                           │ player_id_map             │
└──────┬───────┴───────────┬───────────────┴─────────────┬─────────────┘
       │                   │                             │
       ▼                   ▼                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ml/mantra/runner.py                               │
│  Calcola: P1, P2, P3, P4, CP, FP, FP_Corr, CP_Corr, VR              │
│  Output: mantra_results_{season}.json                                │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│               ml/mantra_ibrido/runner.py                             │
│                                                                      │
│  1. merger.merge_datasets(mantra_results, results_latest)            │
│     → left join su player_name                                      │
│     → arricchisce con predicted, pred_std, expected_min              │
│                                                                      │
│  2. scoring.compute_hybrid_scores(merged, config)                    │
│     → calcola ml_norm, fpIbrido, confidence, mlBoost, fpGap, EV     │
│     → per chi non ha ML: fpIbrido = FP_Corr, resto None             │
│                                                                      │
│  3. classifications.compute_hybrid_classifications(players, config)  │
│     → assegna le 8 label                                             │
│     → solo per giocatori con has_ml_data=True                        │
│                                                                      │
│  Output: mantra_ibrido_results_{season}.json                         │
└──────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    API FastAPI (/predictions/hybrid)                 │
│  Legge il JSON, trasforma chiavi in camelCase, serve al frontend    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Formule riassuntive

```
ml_norm     = clip((predicted - 4.0) / 5.0, 0, 1) * 100
fpIbrido    = FP_Corr * PESO_MANTRA + ml_norm * PESO_ML
confidence  = (1/(1+pred_std)*0.6 + min(expected_min/2700,1)*0.4) * 100
mlBoost     = clip(50 + (predicted - μ_ruolo)/σ_ruolo * 15, 0, 100)
fpGap       = FP_Corr - ml_norm
fpIbridoVoto= 4.0 + fpIbrido/100 * 6.0
expectedVal = fpIbridoVoto * expected_min/90 * EV_SCALE_FACTOR
```

---

## 5. Dataset (2026-07-28)

| Metrica | N | P10 | P25 | P50 | P75 | P90 | Min | Max |
|---|---|---|---|---|---|---|---|---|
| FP_Corr | 532 | 21.5 | 29.0 | 45.3 | 63.3 | 84.6 | 7.0 | 99.3 |
| ml_norm | 351 | 39.2 | 41.7 | 46.5 | 53.1 | 59.6 | 35.6 | 92.0 |
| confidence | 351 | 56.5 | 57.5 | 58.3 | 58.9 | 59.3 | 49.8 | 60.0 |
| predicted | 351 | 6.0 | 6.1 | 6.3 | 6.7 | 7.0 | 5.8 | 8.6 |
| mlBoost | 351 | 34.4 | 39.3 | 46.9 | 59.6 | 68.7 | 12.8 | 100.0 |
| fpGap | 351 | -16.9 | -6.9 | 7.8 | 25.6 | 36.1 | -42.1 | 47.9 |
| VR | 532 | 60.2 | 72.2 | 93.7 | 111.6 | 126.6 | 33.7 | 161.6 |

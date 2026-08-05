# Plan: Sostituire `/predictions` con Pagina Ibrida MANTRA+ML (v3.0)

## TL;DR

Sostituire la pagina `/predictions` (che oggi mostra solo `predicted_fantavoto` ML con due tab) con una **nuova pagina ibrida** che unisce MANTRA (FP, VR, pilastri) + ML (predicted_fantavoto, expected_minutes, confidence, VAR) in un'unica vista. Pesi 50/50 di default ma **regolabili in tempo reale via interfaccia admin**, con anteprima isolata che non inquina i dati pubblici. Tre tab: Panoramica Ibrida (nuovo) + ML Next Season + ML Pipeline Info (con pannello di configurazione ibrida).

---

## Cosa cambia concretamente

| Aspetto | Ora | Dopo |
|---|---|---|
| **Route `/predictions`** | Due tab: Current Season (solo predicted ML) + Next Season | Tre tab: **Ibrido** (nuovo, default) + Next Season (invariato) + **Pipeline Info** (ora include pannello admin) |
| **Current Season tab** | Lista giocatori con predicted, avg, role filter | **SOSTITUITO** dal tab Ibrido |
| **Dati mostrati** | Solo ML predictions | MANTRA (FP_Corr, VR, P1-P4, ruolo_mantra) + ML (predicted, expected_minutes, confidence) + score ibridi (FP_Ibrido, Punti Stagione Attesi, Confidence, Gap, label) |
| **Endpoint principale** | `GET /predictions/players` | `GET /predictions/hybrid` — serve JSON pre-calcolato, pesi da configurazione persistente |
| **Configurabilità** | — | Pannello admin con slider doppio MANTRA/ML, pesi confidence, soglie, salvataggio permanente o anteprima effimera |
| **Isolamento test** | — | Anteprima (`persist:false`) scrive su file separato, endpoint dedicato; nessun dato sperimentale mostrato agli utenti normali |

---

## Decisioni tecniche

### D1. Confidence_Score formula
`1/(1+prediction_std)` invece di espressioni che possono diventare negative. Sempre positiva.
`prediction_std=0 → 1.0`, `std=1 → 0.5`, `std=3 → 0.25`.

### D2. reliability_weight rimosso
Il campo non esiste nell'output serializzato ML (`results_latest.json`) — esiste solo nei domain models interni (`PlayerV2`, `Player`), non serializzato. La formula usa solo due componenti:
- `W_PREDICTION_STD = 0.6` (peso della deviazione standard)
- `W_MINUTES = 0.4` (peso dei minuti attesi)

Entrambi validati con somma = 1.0.

### D3. FP_Gap su scala omogenea 0-100
`FP_Gap = FP_Corr - ML_score_norm`. Entrambi sono valori 0-100, quindi il gap è immediatamente interpretabile. `ML_Boost` (z-score trasformato) rimane solo per la label "ML_Boosted".

### D4. Expected_Value denormalizzato — "Punti Stagione Attesi"
Non più un prodotto arbitrario (che con FP_Ibrido 0-100 produceva numeri come 1950, non interpretabili), ma **punti totali fantacalcio attesi in stagione**. Si riporta `FP_Ibrido` (0-100) in scala voto reale 4-10:

```
FP_Ibrido_voto = 4 + (FP_Ibrido / 100) × 6
Partite_Attese = expected_minutes / 90
Expected_Value = FP_Ibrido_voto × Partite_Attese × EV_SCALE_FACTOR
```

**Esempio**: `FP_Corr=80`, `predicted=6.5` → `ML_score_norm=50` → `FP_Ibrido=65` → `FP_Ibrido_voto=7.9`. Con `expected_minutes=2700` → `Partite_Attese=30` → `Expected_Value = 7.9 × 30 × 1.0 = 237`.

`EV_SCALE_FACTOR` (default 1.0) permette future calibrazioni (es. correggere sovrastime sistematiche) senza cambiare la formula base.

In UI: colonna **"Punti Stagione Attesi"** (non "Expected Value", per chiarezza verso l'utente finale).

### D5. Merge per `player_fotmob_id`
Il runner MANTRA fa già join su `player_id_map` nella query SQL, ma non esporta `player_fotmob_id` nell'output JSON. **Aggiunto** come campo (già presente nel DataFrame). Il merge primario avviene su questo ID; fallback su nome normalizzato (lowercase, trim, rimuovi spazi doppi).

Se `results_latest.json` non esiste, tutti i giocatori avranno `has_ml_data = false` — nessun crash.

### D6. Pre-calcolo dell'output ibrido
Il file `mantra_ibrido_results_{season}.json` viene generato una volta (manualmente o via endpoint admin) e servito staticamente, come già avviene per `/mantra/players`. **Nessun ricalcolo a ogni richiesta GET**, salvo lazy init se il file manca del tutto.

### D7. Filtro ruolo semplificato
Solo filtro per ruolo MANTRA (12 ruoli: Por, Dc, Dd, Ds, B, E, M, C, T, W, A, Pc). Il ruolo ML canonico (GK/DEF/MID/FWD) non ha filtro separato — causerebbe solo insiemi vuoti. Restano i filtri: search, confidenceMin, label ibrida.

### D8. Configurazione regolabile con anteprima isolata
I pesi **non sono hardcoded**. La configurazione viene letta/scritta su `config/mantra_ibrido_config.json` tramite `config_store.py`. L'interfaccia admin offre due modalità:

- **Salva e Rigenera** (`persist:true`): aggiorna la configurazione persistente e rigenera il file pubblico.
- **Prova** (`persist:false`): applica i pesi solo in memoria, scrive su un file di anteprima separato `mantra_ibrido_preview_{season}.json`, **mai sul file pubblico**.

L'endpoint `GET /predictions/hybrid` legge solo il file di produzione; l'anteprima è servita da un endpoint dedicato, accessibile solo all'admin. Questo isolamento evita che un test sperimentale dell'admin sia visibile, anche temporaneamente, agli utenti normali.

### D9. Sleeper label
`FP_Corr < 50 AND ML_score_norm > 40` (entrambi scala 0-100).

---

## Classificazioni (Step 1.7)

Etichette non mutuamente esclusive — un giocatore può averne più di una:

| Label | Condizione | Colore suggerito |
|---|---|---|
| **ML_Confirmed** | predicted > 6.5 AND Confidence_Score ≥ 60 AND expected_minutes > 1500 | Verde scuro |
| **ML_Risky** | Confidence_Score < 30 | Rosso / arancione |
| **ML_Boosted** | ML_Boost > 65 | Viola |
| **Contradiction** | \|FP_Gap\| > 25 | Giallo / ambra |
| **Minutes_Risk** | expected_minutes < 900 | Rosso chiaro |
| **Best_Value** | VR > 140 AND expected_minutes > 1500 | Verde chiaro |
| **Sleeper** | FP_Corr < 50 AND ML_score_norm > 40 | Blu |

---

## Fasi implementative

### Fase 1 — Backend ML: modulo `ml/mantra_ibrido/`

#### 1.1 — `ml/mantra_ibrido/__init__.py`
Esporta `run_hybrid_computation`, `MantraIbridoConfig`, `load_config`, `update_config`.

#### 1.2 — `ml/mantra_ibrido/config.py`
Dataclass **non frozen** `MantraIbridoConfig` con i campi:

| Parametro | Default | Descrizione |
|---|---|---|
| `PESO_MANTRA` | 0.5 | Peso FP_Corr nel calcolo FP_Ibrido |
| `PESO_ML` | 0.5 | Peso predicted_fantavoto normalizzato |
| `W_PREDICTION_STD` | 0.6 | Peso prediction_std in Confidence_Score |
| `W_MINUTES` | 0.4 | Peso expected_minutes |
| `EV_SCALE_FACTOR` | 1.0 | Fattore correttivo per Expected_Value |
| `SOGLIA_CONFIDENZA_MIN` | 0.3 | Sotto questa soglia → flag "dato ML insufficiente" |
| `SOGLIA_GAP_ALERT` | 25.0 | Gap FP_Corr - ML_score_norm per flag "contraddizione" |

Nessun valore hardcoded se non come fallback iniziale: la config viene sempre caricata da file tramite `config_store`.

#### 1.3 — `ml/mantra_ibrido/config_store.py` (gestione persistenza e validazione)

```python
import json, os, tempfile
from dataclasses import asdict
from pathlib import Path
from .config import MantraIbridoConfig

DEFAULT_CONFIG_PATH = Path("config/mantra_ibrido_config.json")

DEFAULTS = {
    "PESO_MANTRA": 0.5,
    "PESO_ML": 0.5,
    "W_PREDICTION_STD": 0.6,
    "W_MINUTES": 0.4,
    "EV_SCALE_FACTOR": 1.0,
    "SOGLIA_CONFIDENZA_MIN": 0.3,
    "SOGLIA_GAP_ALERT": 25.0
}

def validate_config(cfg: dict) -> None:
    if not (0.999 <= cfg["PESO_MANTRA"] + cfg["PESO_ML"] <= 1.001):
        raise ValueError("PESO_MANTRA + PESO_ML deve essere 1.0")
    if not (0.999 <= cfg["W_PREDICTION_STD"] + cfg["W_MINUTES"] <= 1.001):
        raise ValueError("W_PREDICTION_STD + W_MINUTES deve essere 1.0")
    for k in ["PESO_MANTRA", "PESO_ML", "W_PREDICTION_STD", "W_MINUTES"]:
        if not 0.0 <= cfg[k] <= 1.0:
            raise ValueError(f"{k} deve essere in [0,1]")
    for k in ["SOGLIA_CONFIDENZA_MIN", "SOGLIA_GAP_ALERT", "EV_SCALE_FACTOR"]:
        if cfg[k] <= 0:
            raise ValueError(f"{k} deve essere > 0")

def load_config(path: Path = DEFAULT_CONFIG_PATH) -> MantraIbridoConfig:
    """Legge la config persistita; se il file non esiste, usa DEFAULTS."""
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        full = {**DEFAULTS, **data}   # riempie eventuali chiavi mancanti
    else:
        full = dict(DEFAULTS)
    validate_config(full)
    return MantraIbridoConfig(**full)

def update_config(partial: dict, path: Path = DEFAULT_CONFIG_PATH) -> MantraIbridoConfig:
    """Merge parziale contro la config CORRENTE, non contro i default.
    Evita che campi già personalizzati (es. pesi) vengano azzerati
    quando si aggiorna solo un sottoinsieme di parametri."""
    current = asdict(load_config(path))
    merged = {**current, **partial}
    validate_config(merged)
    new_config = MantraIbridoConfig(**merged)
    save_config(new_config, path)
    return new_config

def save_config(config: MantraIbridoConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    data = asdict(config)
    validate_config(data)
    os.makedirs(path.parent, exist_ok=True)
    # Scrittura atomica: evita file corrotti in caso di crash a metà scrittura
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)
```

**Nota critica**: `update_config` fa merge contro la configurazione **già personalizzata** (letta con `load_config`), non contro `DEFAULTS`. In una versione precedente del piano, un `PUT /config` che aggiornava un solo campo (es. `EV_SCALE_FACTOR`) avrebbe silenziosamente azzerato pesi già personalizzati dall'admin (es. `PESO_MANTRA=0.7` sarebbe tornato a `0.5`). Questo fix elimina quel rischio.

**Vincolo consigliato lato API**: i campi `PESO_MANTRA` e `PESO_ML` (così come `W_PREDICTION_STD`/`W_MINUTES`) andrebbero sempre aggiornati **in coppia** nel body di `PUT /config`, mai singolarmente, per evitare stati intermedi che falliscono la validazione della somma.

#### 1.4 — Modifica preliminare: `ml/mantra/runner.py`
Aggiungere il campo `player_fotmob_id` nell'output JSON di ogni giocatore (il dato è già presente nel DataFrame, `pim.player_fotmob_id` nella query SQL).

```json
{
  "fantacalcio_id": 12345,
  "player_fotmob_id": 67890,
  "player_name": "...",
  ...
}
```

#### 1.5 — `ml/mantra_ibrido/merger.py`

Funzione `merge_datasets(mantra_path: Path, ml_path: Path) -> dict`:
1. Carica `mantra_results_{season}.json` (lista `players`)
2. Carica `results_latest.json` (liste `predictions`, `var_results`, `next_season_predictions`)
3. Costruisce mappa `{player_fotmob_id → ml_data}` dalle predictions ML
4. Arricchisce ogni giocatore MANTRA con: `predicted_fantavoto`, `prediction_std`, `expected_minutes` (se match), `var_score`, `esv`, `next_season_predicted`
5. Match fallback per nome normalizzato se `player_fotmob_id` mancante
6. Flag `has_ml_data: bool`
7. Se `results_latest.json` non esiste → tutti i giocatori senza ML data, nessun errore
8. Log: numero di match per fotmob_id, per nome, e giocatori senza ML
9. Restituisce `{ "players": [...], "meta": {...}, "mantra_classifications": {...} }`

#### 1.6 — `ml/mantra_ibrido/scoring.py`

Funzione `compute_hybrid_scores(players_arricchiti: list, config: MantraIbridoConfig) -> list`:

Per ogni giocatore **con** ML data:

```python
ml_norm = clip((predicted_fantavoto - 4) / 5 * 100, 0, 100)
fp_ibrido = FP_Corr * config.PESO_MANTRA + ml_norm * config.PESO_ML

std_term = 1 / (1 + prediction_std)
min_term = min(expected_minutes / 2700, 1)
confidence_score = (std_term * config.W_PREDICTION_STD + min_term * config.W_MINUTES) * 100

ml_boost = compute_zscore_boost(predicted_fantavoto, pool_by_role)  # 50 + z*15, clip [0,100]

fp_gap = FP_Corr - ml_norm

fp_ibrido_voto = 4 + (fp_ibrido / 100) * 6
partite_attese = expected_minutes / 90
expected_value = fp_ibrido_voto * partite_attese * config.EV_SCALE_FACTOR
```

Per giocatori **senza** ML data:
- `fp_ibrido = FP_Corr` (solo MANTRA)
- `ml_score_norm`, `confidence_score`, `ml_boost`, `fp_gap`, `expected_value` = `null`
- `ml_unavailable: true`

#### 1.7 — `ml/mantra_ibrido/classifications.py`
Funzione `compute_hybrid_classifications(players_ibrido: list) -> dict` applicando la tabella label sopra. Output: `{ "ML_Confirmed": [...nomi], "ML_Risky": [...], ... }`.

#### 1.8 — `ml/mantra_ibrido/runner.py`

```python
import tempfile, os
from pathlib import Path
from dataclasses import asdict

def run_hybrid_computation(
    mantra_path: Path,
    ml_path: Path,
    output_dir: Path,
    config: MantraIbridoConfig | None = None,
    output_filename: str | None = None,   # permette di scrivere su preview invece che su produzione
) -> dict:
    if config is None:
        config = load_config()

    merged = merge_datasets(mantra_path, ml_path)

    # Stagione derivata dal meta del file MANTRA (non una variabile esterna non definita)
    season = merged["meta"].get("seasonStart")
    if season is None:
        raise ValueError("Impossibile determinare la stagione: meta.seasonStart mancante nel file MANTRA")

    players_ibridi = compute_hybrid_scores(merged["players"], config)
    classifications = compute_hybrid_classifications(players_ibridi)

    result = {
        "meta": {**merged["meta"], "config": asdict(config)},
        "players": players_ibridi,
        "classifications": classifications,
    }

    filename = output_filename or f"mantra_ibrido_results_{season}.json"
    final_path = output_dir / filename

    # Scrittura atomica: evita che GET /hybrid legga un file troncato durante il run
    fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
    with os.fdopen(fd, 'w') as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, final_path)

    return result
```

Se `output_filename` viene passato (es. per l'anteprima), si scrive su quel file **senza mai toccare** il file di produzione.

#### 1.9 — Test automatici (`ml/mantra_ibrido/tests/`)

- `test_merger.py`: match per fotmob_id, fallback per nome, giocatore senza ML, ML senza MANTRA
- `test_scoring.py`: FP_Ibrido 50/50, Confidence_Score con std=0 e std alto, Expected_Value con minuti zero, campi null per giocatori senza ML, verifica esempio `FP_Corr=80 / predicted=6.5 → Expected_Value=237`
- `test_classifications.py`: ogni label con casi limite (soglie appena sopra/sotto)
- `test_config_store.py`:
  - `load_config()` su file assente → ritorna `DEFAULTS`
  - `load_config()` su file parziale → integra i default mancanti
  - `update_config({"EV_SCALE_FACTOR": 1.2})` con pesi già personalizzati (es. `0.7/0.3`) → **i pesi non vengono azzerati** (verifica esplicita del bug fixato)
  - Validazione: somma pesi ≠ 1 → `ValueError`
  - Scrittura atomica: crash simulato a metà scrittura → file originale rimane intatto
- `test_runner.py`: scrittura su `output_filename` personalizzato senza toccare il file standard; derivazione corretta della stagione; errore esplicito se `seasonStart` assente

---

### Fase 2 — Backend API

#### 2.1 — Nuovi endpoint in `api/src/routers/intelligence.py`

Tutti gli endpoint di scrittura/modifica richiedono autenticazione admin tramite il sistema di auth già presente nell'app (`Depends(require_admin)` o dependency equivalente già in uso altrove nel router).

**`GET /predictions/hybrid`**
- Legge **sempre e solo** `mantra_ibrido_results_{season}.json` (file di produzione), con fallback stagioni 2026→2025→2024
- Se assente → lazy init con `run_hybrid_computation()`
- Filtri: `ruolo` (MANTRA), `search`, `confidenceMin`, `label`, `sortBy` (validato contro enum, vedi sotto), `sortDir`
- Paginazione: `page`, `size` (max 200)

**`GET /predictions/hybrid/stats`**
- Statistiche aggregate sullo stesso file: `totalPlayers`, `pctWithMl`, `avgFpIbrido`, `avgConfidenceScore`, `avgFpGap`, `classificationCounts`

**`GET /predictions/hybrid/config`**
- Restituisce la configurazione corrente (`load_config()`)

**`PUT /predictions/hybrid/config`**
- Riceve un body JSON parziale, chiama `update_config(partial)` (merge contro la config corrente, non contro i default)
- Non rigenera i risultati — serve poi chiamare `POST /run`

**`POST /predictions/hybrid/run`**

Corpo:
```json
{
  "overrides": { "PESO_MANTRA": 0.6, "PESO_ML": 0.4 },
  "persist": false
}
```

Logica:
```python
@router.post("/predictions/hybrid/run")
def run_hybrid(body: RunHybridRequest, user=Depends(require_admin)):
    if body.persist:
        # Salva permanentemente (se ci sono overrides) e rigenera il file PUBBLICO
        config = update_config(body.overrides) if body.overrides else load_config()
        result = run_hybrid_computation(
            mantra_path, ml_path, output_dir,
            config=config,
            output_filename=None,  # nome standard di produzione
        )
    else:
        # Configurazione effimera: solo in memoria, mai persistita
        base = asdict(load_config())
        effective = MantraIbridoConfig(**{**base, **(body.overrides or {})})
        validate_config(asdict(effective))
        # Scrive SOLO sul file di preview, mai su quello servito da /hybrid
        result = run_hybrid_computation(
            mantra_path, ml_path, output_dir,
            config=effective,
            output_filename=f"mantra_ibrido_preview_{season}.json",
        )
    return {
        "season": season,
        "nPlayers": len(result["players"]),
        "generatedAt": result["meta"]["generatedAt"],
        "persisted": body.persist,
    }
```

Punti chiave:
- `persist:false` **non scrive mai** sul file letto da `GET /predictions/hybrid` → nessun utente normale vede dati sperimentali, nemmeno temporaneamente
- `persist:true` è l'unica via per aggiornare sia la config sia i risultati pubblici

**`GET /predictions/hybrid/preview`** (solo admin)
- Se `mantra_ibrido_preview_{season}.json` esiste, lo restituisce; altrimenti 404
- Usato esclusivamente dal pannello admin per mostrare l'anteprima non pubblicata

**Validazione `sortBy` — enum esplicito:**
```python
class HybridSortField(str, Enum):
    fp_ibrido = "fpIbrido"
    confidence_score = "confidenceScore"
    expected_value = "expectedValue"
    fp_gap = "fpGap"
    predicted_fantavoto = "predictedFantavoto"
    fp_corr = "FP_Corr"
    vr = "VR"
```
`GET /hybrid?sortBy=...` valida contro questo enum → 422 se il valore non è riconosciuto.

#### 2.2 — `api/src/data_repository.py`
Aggiungere `get_hybrid_predictions()`:
- Carica `mantra_ibrido_results_{season}.json` (con fallback R2)
- Se non esiste → lazy init via merger/scoring
- Cache opzionale Redis

---

### Fase 3 — Frontend

#### 3.1 — `frontend/src/app/core/models/api.models.ts`

```typescript
export interface HybridPlayerPrediction {
  playerName: string;
  team: string | null;
  canonicalRole: string | null;
  ruoloPrimario: string | null;
  ruoliMantra: string[];
  P1: number | null;
  P2: number | null;
  P3: number | null;
  P4: number | null;
  FP_Corr: number | null;
  VR: number | null;
  predictedFantavoto: number | null;
  expectedMinutes: number | null;
  confidenceScore: number | null;
  mlBoost: number | null;
  fpGap: number | null;
  fpIbrido: number | null;
  expectedValue: number | null;    // "Punti Stagione Attesi"
  prezzoMassimo: number | null;
  hybridLabels: string[];
  hasMlData: boolean;
}

export interface HybridPredictionsResponse {
  total: number;
  page: number;
  size: number;
  items: HybridPlayerPrediction[];
  meta: {
    seasonStart: number;
    generatedAt: string;
    config: Record<string, number>;
    nPlayersWithMl: number;
    nPlayersWithoutMl: number;
  };
}

export interface HybridStatsResponse {
  totalPlayers: number;
  pctWithMl: number;
  avgFpIbrido: number;
  avgConfidenceScore: number;
  avgFpGap: number;
  classificationCounts: Record<string, number>;
}

export interface HybridConfig {
  PESO_MANTRA: number;
  PESO_ML: number;
  W_PREDICTION_STD: number;
  W_MINUTES: number;
  EV_SCALE_FACTOR: number;
  SOGLIA_CONFIDENZA_MIN: number;
  SOGLIA_GAP_ALERT: number;
}
```

#### 3.2 — `frontend/src/app/core/services/prediction.service.ts`

```typescript
getHybridPredictions(params?: {
  page?: number; size?: number; ruolo?: string;
  search?: string; confidenceMin?: number; label?: string;
  sortBy?: string; sortDir?: string
}): Observable<HybridPredictionsResponse>

getHybridStats(): Observable<HybridStatsResponse>

getHybridConfig(): Observable<HybridConfig>

updateHybridConfig(config: Partial<HybridConfig>): Observable<HybridConfig>

runHybrid(overrides?: Partial<HybridConfig>, persist?: boolean): Observable<{
  season: number; nPlayers: number; generatedAt: string; persisted: boolean;
}>

getHybridPreview(): Observable<HybridPredictionsResponse>  // solo admin
```

#### 3.3 — Riscrittura `predictions.component.ts`

**Tab 1 — Panoramica Ibrida** (default)
- Header: "Hybrid Predictions" + badge pesi correnti (letti dalla config, es. "MANTRA 50% / ML 50%") + badge "X% con dati ML"
- Stats strip: avg FP_Ibrido, avg Confidence, avg FP_Gap, n° classificazioni attive
- Filtri: search, ruolo MANTRA (dropdown 12 ruoli), confidence (preset: All / ≥70 / ≥50 / <30), label (pill colorate per ogni classificazione)
- Tabella ordinabile (click su header): `#, Player, Team, Ruolo M, P1-P4 (tooltip), FP_Corr, Predicted, FP_Ibrido (barra), VR, Conf (barra colorata), Punti Stagione Attesi, Gap (freccia colorata), Labels (badge)`
  - FP_Ibrido: barra di progresso 0-100 + valore
  - Confidence_Score: barra verde ≥70, gialla 40-69, rossa <40
  - FP_Gap: verde se positivo (MANTRA > ML), arancione se negativo (ML > MANTRA), con freccia
- Stati: loading (skeleton), errore (error-boundary), vuoto ("No hybrid predictions available")
- Paginazione: 50 items, Prev/Next
- Click riga → drawer dettaglio giocatore

**Tab 2 — Next Season**
- Identico all'attuale: lista `predictedNextFantavoto` con bar

**Tab 3 — Pipeline Info + Admin Panel**
- Sezione esistente invariata: model comparison, run ID, feature importance
- **Nuova sezione "Configurazione Ibrida"**, visibile solo ad admin:
  - Slider singolo per `PESO_MANTRA` (0-100%); `PESO_ML = 100 - PESO_MANTRA`, entrambe le etichette aggiornate in tempo reale
  - Campi numerici per `W_PREDICTION_STD` / `W_MINUTES` (validazione somma = 1)
  - Campi per `EV_SCALE_FACTOR`, `SOGLIA_CONFIDENZA_MIN`, `SOGLIA_GAP_ALERT`
  - Bottone **"Prova"** (`persist:false`): chiama `runHybrid(overrides, false)`, poi `getHybridPreview()`, mostra i risultati in una sotto-sezione separata "Anteprima" con badge arancione "Anteprima — non pubblicata, non visibile agli utenti". **Non modifica la tabella del Tab 1.**
  - Bottone **"Salva e Rigenera"** (`persist:true`): chiama `runHybrid(overrides, true)`, poi ricarica config e Tab 1 con i dati pubblici aggiornati
  - Timestamp "Ultima rigenerazione: {generatedAt}" sempre visibile
  - Validazione client-side: bottoni disabilitati se le somme dei pesi non tornano

#### 3.4 — Markup slider doppio (logico)

```html
<div class="dual-slider-container">
  <label>Peso MANTRA: <span id="mantra-value">50%</span></label>
  <label class="right">Peso ML: <span id="ml-value">50%</span></label>
  <div class="slider-track">
    <input type="range" id="slider-mantra" min="0" max="100" value="50"
           (input)="onMantraSliderChange($event)">
  </div>
</div>
```

```typescript
onMantraSliderChange(event: Event) {
  const value = parseInt((event.target as HTMLInputElement).value, 10);
  this.config.PESO_MANTRA = value / 100;
  this.config.PESO_ML = (100 - value) / 100;
}
```

Un unico slider fisico rappresenta `PESO_MANTRA`; `PESO_ML` è sempre derivato. Se l'admin modifica il campo numerico direttamente, l'altro si adatta di conseguenza.

#### 3.5 — Shell e Route
Nessuna modifica: `shell.component.ts` e `app.routes.ts` restano invariati.

---

## Riepilogo file coinvolti

### Nuovi file
- `ml/mantra_ibrido/__init__.py`
- `ml/mantra_ibrido/config.py`
- `ml/mantra_ibrido/config_store.py`
- `ml/mantra_ibrido/merger.py`
- `ml/mantra_ibrido/scoring.py`
- `ml/mantra_ibrido/classifications.py`
- `ml/mantra_ibrido/runner.py`
- `ml/mantra_ibrido/tests/__init__.py`
- `ml/mantra_ibrido/tests/test_merger.py`
- `ml/mantra_ibrido/tests/test_scoring.py`
- `ml/mantra_ibrido/tests/test_classifications.py`
- `ml/mantra_ibrido/tests/test_config_store.py`
- `ml/mantra_ibrido/tests/test_runner.py`
- `config/mantra_ibrido_config.json` (creato automaticamente al primo salvataggio)

### File da modificare
- `ml/mantra/runner.py` — aggiungere `player_fotmob_id` all'output JSON
- `api/src/routers/intelligence.py` — +6 endpoint: `GET /predictions/hybrid`, `GET /predictions/hybrid/stats`, `GET /predictions/hybrid/config`, `PUT /predictions/hybrid/config`, `POST /predictions/hybrid/run`, `GET /predictions/hybrid/preview`
- `api/src/data_repository.py` — +1 metodo: `get_hybrid_predictions()`
- `frontend/src/app/core/models/api.models.ts` — +4 interfacce ibride
- `frontend/src/app/core/services/prediction.service.ts` — +6 metodi HTTP
- `frontend/src/app/features/predictions/predictions.component.ts` — riscritto con 3 tab + admin panel
- `frontend/src/app/features/predictions/predictions.component.html` — template rivisto
- `frontend/src/app/features/predictions/predictions.component.scss` — stili aggiuntivi

### File da NON modificare
- `frontend/src/app/app.routes.ts` — route `/predictions` invariata
- `frontend/src/app/shared/components/shell/shell.component.ts` — voce menu "Predictions" invariata
- `api/src/main.py` — nessun nuovo router da registrare

---

## Dipendenze tra step

```
Fase 1.4 (modifica runner MANTRA)
        │
        ▼
Fase 1.2 (config) ──┐
Fase 1.3 (config_store) ──┐
Fase 1.5 (merger) ──┼──→ Fase 1.6 (scoring) ──→ Fase 1.7 (classifications)
                     │                                      │
                     └──→ Fase 1.8 (runner) ←───────────────┘
                                           │
                                    Fase 1.9 (tests)
                                           │
                                           ▼
                                      Fase 2 (API)
                                           │
                                           ▼
                                      Fase 3 (Frontend)
```

---

## Ordine di implementazione consigliato

1. **Fase 1.1–1.3** — modulo base + `config_store.py` (merge corretto, scrittura atomica) + test dedicati
2. **Fase 1.4** — modifica MANTRA runner (`player_fotmob_id`)
3. **Fase 1.5–1.7** — merger, scoring (Expected_Value corretto), classifications + test
4. **Fase 1.8** — runner con `output_filename` parametrizzato e derivazione corretta di `season`
5. **Fase 1.9** — suite test completa
6. **Fase 2** — endpoint API, con guard admin sul meccanismo di auth già esistente nell'app
7. **Fase 3** — frontend: modelli, service, componente con 3 tab e pannello admin

---

## Verifica finale

### Config Store
- `load_config()` su file assente → ritorna `DEFAULTS`
- `load_config()` su file parziale → integra i default mancanti
- `update_config({"EV_SCALE_FACTOR": 1.2})` con pesi personalizzati (`0.7/0.3`) → i pesi **non** vengono azzerati
- `PUT /config` con somma pesi ≠ 1 → rifiutato con errore di validazione
- Scrittura atomica: crash simulato a metà scrittura → file originale intatto

### Merge dati
- Match per `player_fotmob_id` (es. due JSON con lo stesso ID → merge OK)
- Match fallback per nome normalizzato
- Giocatore solo in MANTRA → `has_ml_data=false`, nessun crash
- `results_latest.json` assente → tutti i giocatori senza ML data, nessun errore

### Score ibridi
- `FP_Corr=80`, `predicted=6.5` → `ML_score_norm=50` → `FP_Ibrido=65`
- `FP_Ibrido=65`, `expected_minutes=2700` → `FP_Ibrido_voto=7.9` → `Expected_Value=237`
- `prediction_std=0` → `Confidence_Score=100`
- `prediction_std=3` → `Confidence_Score=55`
- `expected_minutes=0` → `Expected_Value=0`
- `FP_Corr=45`, `ML_score_norm=50` → label Sleeper attiva
- `FP_Gap=30` → label Contradiction attiva

### API
- `GET /predictions/hybrid?page=1&size=50` → risposta paginata con tutti i campi ibridi
- `GET /predictions/hybrid/stats` → statistiche aggregate coerenti
- `GET /predictions/hybrid?ruolo=Dc` → solo difensori centrali
- `GET /predictions/hybrid?sortBy=xyz` (valore non valido) → 422
- `PUT /predictions/hybrid/config` con somma pesi ≠ 1 → rifiutato
- `POST /predictions/hybrid/run` con `persist:false` → rigenera solo il file di preview, il file pubblico e la config restano invariati
- `POST /predictions/hybrid/run` con `persist:true` → aggiorna config e file pubblico
- Endpoint di scrittura richiedono autenticazione admin (già presente nel sistema)

### Frontend
- `/predictions` carica con tab Ibrido come default
- Tabella mostra tutte le colonne previste, incluso "Punti Stagione Attesi"
- FP_Ibrido con barra di progresso, Confidence_Score con barra colorata, FP_Gap con colore condizionale
- Filtri e ordinamento funzionanti
- Admin panel: slider collegato, bottone "Prova" mostra anteprima separata senza toccare il Tab 1, bottone "Salva e Rigenera" aggiorna sia config sia dati pubblici
- Tab Next Season e Pipeline Info accessibili e invariati nella parte esistente

---

## Note tecniche riassuntive

- **reliability_weight rimosso**: non presente in `results_latest.json`, solo nei domain models interni
- **Pre-calcolo**: l'API non ricalcola a ogni richiesta; il JSON ibrido è generato una volta e servito staticamente
- **Isolamento preview/produzione**: `persist:false` scrive solo su file di anteprima, mai su quello pubblico — nessun utente normale vede dati sperimentali
- **Merge config parziale**: sempre contro la configurazione corrente, mai contro i default, per non perdere personalizzazioni già salvate
- **Scrittura atomica**: sia per `config_store.py` che per `runner.py`, tramite file temporaneo + `os.replace()`
- **CamelCase nei campi JSON**: `fpIbrido`, `confidenceScore`, `mlBoost`, `fpGap`, `expectedValue`, `hybridLabels`, `hasMlData` — consistente con altri endpoint API
- **Nomi MANTRA in PascalCase** preservati: `FP_Corr`, `P1`, `P2`, `P3`, `P4`, `VR`, `Prezzo_Massimo`
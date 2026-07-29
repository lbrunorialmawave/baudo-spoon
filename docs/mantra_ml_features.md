# PIANO OPERATIVO: Integrazione Dati MANTRA come Feature ML

## 1. Diagnosi e Obiettivi

### 1.1 Il Problema

Il sistema attuale prevede il fantavoto medio utilizzando:
- **Modello ML** → ~45 feature da statistiche FotMob (trend, contesto squadra, prestazioni)
- **Pipeline MANTRA** → voti storici, prezzi d'asta, expected stats, minuti medi, esperienza

Le due pipeline sono **disgiunte**: l'ibrido le media 50/50, ma il ML **non impara mai** da ciò che il MANTRA già sa. Questo lascia sul tavolo informazioni cruciali:

- Quanto ha **votato** in passato (V, DV)
- Quanto è stato **costante** (Pr, Min_annuo)
- Quanto **expected** produceva (xG90, xA90)
- Quanto **esperienza** ha in Serie A (Stagioni_IT)
- Quanto **valeva all'asta** (Pz1, Pz2, Pz3)

### 1.2 Obiettivo del Progetto

**Integrare le feature MANTRA nel training del modello ML**, in modo che il sistema impari a pesare automaticamente queste informazioni, superando il semplice compromesso 50/50.

**Vincoli**:
- **Zero leakage temporale** — condizione non negoziabile
- **Validazione robusta** su almeno 2-3 stagioni out-of-sample
- **Gestione della collinearità** tra feature (sia nuove che tra fonti)
- **Metriche di successo chiare** e monitoraggio in produzione

---

## 2. Feature Candidate — Priorità e Ordine di Impatto

Le feature sono già disponibili in PostgreSQL. La priorità indicata è basata su ipotesi di impatto, da validare empiricamente.

### Fase 1 — Low-Hanging Fruit (Impatto atteso: Alto)

| Feature | Descrizione | Ruolo/i target |
|---------|-------------|----------------|
| **V** | Voto medio storico | Tutti |
| **DV** | Deviazione standard del voto | Tutti |
| **Pr** | Tasso di presenza (%) | Tutti |
| **Min_annuo** | Minuti giocati medi a stagione | Tutti |
| **xG90** | Expected goals per 90' | A (attaccanti) |
| **xA90** | Expected assist per 90' | C/A (centrocampisti/attaccanti) |
| **Stagioni_IT** | Stagioni in Serie A | Tutti (soprattutto giovani) |

### Fase 2 — Feature Derivate (Impatto atteso: Medio-Alto)

| Feature | Formula | Razionale |
|---------|---------|-----------|
| **voto_trend** | `V / media_ruolo_V` | Sovra-performance rispetto al ruolo |
| **consistency_score** | `V / (DV + epsilon)` | Stabilità: alto se voto alto e deviazione bassa |
| **expected_ratio** | `xG90 / (G90 + 0.01)` | Efficienza realizzativa (cap a 5) |
| **esperienza_flag** | `1 if Stagioni_IT <= 2 else 0` | Giovane vs esperto (binaria) |
| **clean_sheet_ratio** | `clean_sheets / presenze` | Solo portieri, se non già presente |

### Fase 3 — Ensemble (Impatto atteso: Basso-Medio)

Pesi ottimali per ruolo tra ML e MANTRA, da stimare:
- per ruolo (P/D/C/A)
- con validazione leave-one-season-out per evitare overfitting

---

## 3. Fase 0 — Pre-requisiti e Controlli (Prima di ogni training)

### 3.1 Ispezione della View `player_season_aggregates`

**Obiettivo**: verificare che `vote_avg`, `presence_rate`, ecc. siano versionate per stagione e NON includano dati della stagione target.

**Azione**:
```sql
-- Verifica struttura
\d+ player_season_aggregates

-- Controllo campione: per un giocatore, vedere se voti di una stagione
-- sono stati calcolati usando dati di stagioni successive
SELECT player_id, season, vote_avg, 
       LAG(vote_avg) OVER (PARTITION BY player_id ORDER BY season) as prev_vote
FROM player_season_aggregates
WHERE player_id = 'esempio'
ORDER BY season;
```

**Test automatico** (invariato dalla v2.1, confermato corretto in tutte le revisioni successive):

```python
def test_no_leakage(df: pd.DataFrame, meta_col: str = 'V_computed_up_to_season') -> None:
    """
    Verifica che per ogni riga, la stagione fino a cui è stato calcolato il valore
    della feature (ad esempio V) sia strettamente minore della stagione target.
    """
    if meta_col not in df.columns:
        raise ValueError(
            f"Colonna '{meta_col}' non trovata. "
            "Aggiungere la colonna nel data loader per abilitare il test di leakage."
        )
    mask_leak = df['season'] <= df[meta_col]
    if mask_leak.any():
        n_leak = mask_leak.sum()
        examples = df[mask_leak][['player_id', 'season', meta_col]].head(5)
        raise AssertionError(
            f"Leakage rilevato in {n_leak} righe. Esempi:\n{examples.to_string()}"
        )
    print("✅ Test di leakage superato: nessuna feature include dati della stagione target.")
```

Il test va eseguito **prima** di qualsiasi split train/test, sul dataset completo.

### 3.2 Verifica di Correlazione tra Fonti (FotMob vs MANTRA)

Per feature ridondanti (es. `saves_per90`, `clean_sheet_rate`):

```python
corr_matrix = df[['saves_per90_mantra', 'saves_per90_fotmob']].corr()

if corr_matrix.iloc[0, 1] > 0.9:
    drop_cols.append('saves_per90_mantra')
else:
    logger.warning(f"Correlazione saves_per90 = {corr:.2f} — verificare definizioni")
    df['saves_per90_hybrid'] = (df['saves_per90_mantra'] + df['saves_per90_fotmob']) / 2
```

### 3.3 Analisi Preliminare: Correlazione Lag-1

Prima della Fase 1, calcola la **correlazione tra ogni feature candidata e il target futuro**, stratificata per ruolo:

```python
def correlation_lag_analysis(df, target_col='fantavoto_next_season'):
    results = {}
    for ruolo in df['ruolo'].unique():
        df_ruolo = df[df['ruolo'] == ruolo]
        for col in feature_candidates:
            corr = df_ruolo[col].corr(df_ruolo[target_col])
            results[(ruolo, col)] = corr
    return pd.Series(results).sort_values(ascending=False)
```

---

## 4. Fase 1 — Integrazione Base + Validazione A/B

### 4.1 Modifiche al Data Pipeline

```sql
SELECT
    fm.*,
    pss.vote_avg          AS "V",
    pss.vote_std          AS "DV",
    pss.presence_rate     AS "Pr",
    pss.minutes_avg       AS "min_annuo",
    pss.xg_per90          AS "xG90",
    pss.xa_per90          AS "xA90",
    pss.seasons_in_italy  AS "stagioni_it"
FROM fotmob_features fm
LEFT JOIN player_season_aggregates pss
    ON fm.player_id = pss.player_id
    AND fm.season = pss.season
WHERE fm.season < :current_season
```

### 4.2 Controllo di Collinearità Preliminare

Prima del RFE: matrice di correlazione e **VIF (Variance Inflation Factor)**:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[['V', 'DV', 'Pr', 'min_annuo', 'xG90', 'xA90', 'stagioni_it']].dropna()
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

Se VIF > 5 per una o più feature → intervenire con feature derivate o PCA.

### 4.3 Piano di Validazione A/B (Blind) — Criterio Go/No-Go a Due Livelli

**Protocollo**: split temporale (training 2 stagioni, test 1, ripetuto su 3 combinazioni),
metriche RMSE/MAE/MAPE per ruolo, test di Diebold-Mariano tra gli errori dei due modelli,
breakdown per ruolo, feature importance via SHAP/permutation importance.

**Criterio Go/No-Go** *(v2.3: da criterio rigido a due livelli, per non scartare modelli
validi per rumore di un singolo split)*:

| Livello | Condizione | Decisione |
|---------|------------|-----------|
| **Go netto** | RMSE ridotto ≥3% su **tutte** le stagioni testate | Procedi alla Fase 2/Deploy |
| **Go condizionato** | RMSE ridotto ≥3% su almeno 2 stagioni e <1% di peggioramento sulla terza | Analisi della stagione outlier; se spiegabile, procedi con un **canarino** in produzione |
| **No-Go** | Peggioramento >1% in più di una stagione, o >3% in una | Non procedere, rivedere feature o modello |

**Nota v2.3 sulla soglia del 3%**: non è un valore normativo. Va **calibrata empiricamente
in Fase 0**, calcolando la deviazione storica del RMSE tra le stagioni disponibili: se il
RMSE del modello attuale varia già del ±2-3% da una stagione all'altra per pura variabilità,
una soglia inferiore a quella variabilità storica non distingue un vero miglioramento dal
rumore. La soglia definitiva va documentata insieme al calcolo che l'ha generata.

### 4.4 Imputazione Train/Test-Safe

*(Sezione riscritta integralmente in v2.2-v2.5: la versione v2.1 originale calcolava
l'imputazione con `groupby().transform()` su tutto il dataframe, che causa leakage se
applicato indistintamente a train e test. Sostituita con un transformer sklearn dedicato.)*

**Requisito operativo, non negoziabile** *(v2.3)*: `MantraImputer` va sempre incapsulato in
una `sklearn.pipeline.Pipeline`, e il `.fit()`/`.fit_transform()` va chiamato **esclusivamente**
sul training fold di ogni split. Mai chiamare `.fit()` sul dataframe completo prima dello
split, nemmeno "per controllare che funzioni" — è la causa più comune di leakage silenzioso.

```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class MantraImputer(BaseEstimator, TransformerMixin):
    """
    Imputa le feature MANTRA mancanti con fallback gerarchico:
    media (season, ruolo) -> media (ruolo) -> media globale.
    Tutte le medie sono apprese ESCLUSIVAMENTE in fit(), sul training fold.

    IMPORTANTE: va usato solo dentro una sklearn.pipeline.Pipeline, con fit()
    chiamato esclusivamente sul training fold di ogni split. Vedi test
    'test_pipeline_split_safety' per la verifica automatica di questo vincolo
    (quel test verifica che le statistiche apprese dipendano esclusivamente
    dai dati passati a fit() — non è una garanzia assoluta contro ogni forma
    di leakage, es. stato globale condiviso o cache esterne; quelle categorie
    di bug richiedono disciplina di codice e code review, non solo test).
    """

    def __init__(self, feature_cols, role_col='ruolo', season_col='season', drop_original=True):
        self.feature_cols = feature_cols
        self.role_col = role_col
        self.season_col = season_col
        self.drop_original = drop_original

    @property
    def imputed_cols(self):
        return [f"{c}_imputed" for c in self.feature_cols]

    @property
    def missing_flags(self):
        return [f"{c}_missing" for c in self.feature_cols]

    def fit(self, X, y=None):
        self.group_means_ = {}   # (season, ruolo) -> media, per colonna
        self.role_means_ = {}    # ruolo -> media, per colonna (fallback intermedio)
        self.global_means_ = {}  # media globale, per colonna (fallback finale)

        for col in self.feature_cols:
            self.group_means_[col] = X.groupby([self.season_col, self.role_col])[col].mean()
            self.role_means_[col] = X.groupby(self.role_col)[col].mean()
            self.global_means_[col] = X[col].mean()
        return self

    def transform(self, X):
        X_trans = X.copy()

        for col in self.feature_cols:
            imputed_col = f"{col}_imputed"
            flag_col = f"{col}_missing"
            X_trans[flag_col] = X_trans[col].isna().astype(int)

            # Livello 1: media (season, ruolo). Reso deterministico rispetto all'ordine
            # delle righe con reset_index/reindex esplicito (v2.4), invece di affidarsi
            # implicitamente al comportamento di un left join.
            group_df = self.group_means_[col].rename('g1_mean').reset_index()
            merged = (
                X_trans[[self.season_col, self.role_col]]
                .reset_index()
                .merge(group_df, on=[self.season_col, self.role_col], how='left')
                .set_index('index')
                .reindex(X_trans.index)
            )
            level1 = merged['g1_mean'].values

            # Livello 2: media per ruolo, dove il livello 1 manca
            role_df = self.role_means_[col].rename('g2_mean').reset_index()
            merged_role = (
                X_trans[[self.role_col]]
                .reset_index()
                .merge(role_df, on=self.role_col, how='left')
                .set_index('index')
                .reindex(X_trans.index)
            )
            level2 = merged_role['g2_mean'].values

            # Livello 3: media globale, dove anche il livello 2 manca (ruolo mai visto)
            level3 = self.global_means_[col]

            fallback_chain = np.where(
                ~pd.isna(level1), level1,
                np.where(~pd.isna(level2), level2, level3)
            )

            X_trans[imputed_col] = X_trans[col].where(~X_trans[col].isna(), fallback_chain)

        if self.drop_original:
            X_trans = X_trans.drop(columns=self.feature_cols)

        return X_trans
```

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('imputer', MantraImputer(feature_cols=['V', 'DV', 'Pr', 'min_annuo', 'xG90', 'xA90'])),
    ('scaler', StandardScaler()),
    ('model', ...)
])

# CORRETTO: fit solo sul training fold
pipeline.fit(X_train, y_train)
X_test_transformed = pipeline.transform(X_test)  # mai pipeline.fit(X_test)
```

**Nota v2.4 — colonna interamente NaN nel training.** Se una feature è interamente mancante
per tutto il training fold, il fallback gerarchico arriva fino al livello globale, che sarà
anch'esso NaN, e questo si propaga fino al modello a valle (errore per molti stimatori
sklearn). **Regola operativa**: verificare che nessuna colonna sia interamente NaN sul
training fold prima di passarla a `feature_cols`; se lo è, va esclusa a monte, non gestita
silenziosamente dall'imputer.

**Nota v2.4/v2.5 — shrinkage come possibile evoluzione futura (non implementata).** Il
fallback a gradini sceglie sempre il livello più specifico disponibile, anche con poche
osservazioni (es. 2 giocatori). Un'alternativa più robusta sarebbe una media pesata
(shrinkage/James-Stein) `μ = w1·mean(season,ruolo) + w2·mean(ruolo) + w3·mean(globale)`, con
pesi proporzionali alla numerosità di ciascun gruppo. Da valutare se, dopo il deploy, si
osservano previsioni instabili per i gruppi più piccoli (tipicamente portieri).

**Nota v2.5 — possibile ottimizzazione del merge (non applicata).** Poiché `group_means_[col]`
è già una `Series` con `MultiIndex`, si potrebbe ottenere lo stesso risultato con un singolo
`reindex` diretto (`self.group_means_[col].reindex(pd.MultiIndex.from_arrays(...))`), evitando
il doppio passaggio di merge. Non applicata di default in questa versione — richiede test
propri — da valutare se il profiling in produzione mostra che il merge è un collo di bottiglia.

### 4.5 Analisi dei Valori Mancanti

Strategia: imputazione con media di gruppo per giocatori con poche presenze, imputazione
train-safe come sopra. **Nota v2.3**: la soglia "< 5 presenze" della v2.1 originale era
arbitraria — va sostituita con un criterio basato sul **10° percentile della distribuzione
delle presenze per ruolo nella stagione di train**, calibrato empiricamente in Fase 0, non
fissato a priori.

---

## 5. Fase 2 — Feature Derivate (condizionale alla Fase 1)

**Da implementare solo se la Fase 1 ha superato il criterio Go.**

```sql
SELECT
    ...,
    (pss.vote_avg / NULLIF(ruolo_avg.vote_avg, 0)) AS voto_trend,
    (pss.vote_avg / NULLIF(pss.vote_std + 0.1, 0)) AS consistency_score,
    (pss.xg_per90 / NULLIF(fm.goals_per90 + 0.01, 0)) AS expected_ratio,
    CASE WHEN pss.seasons_in_italy <= 2 THEN 1 ELSE 0 END AS esperienza_flag
FROM ...
LEFT JOIN ruolo_avg ON ...
```

**Criterio**: se la Fase 2 non migliora significativamente la Fase 1 (RMSE ridotto < 1%),
mantenere la Fase 1 per semplicità.

---

## 6. Fase 3 — Ensemble (condizionale e prudente)

### 6.1 Step Intermedio: Pesi Fissi per Ruolo

```python
errors = df.groupby('ruolo').agg({'ml_error': 'mean', 'mantra_error': 'mean'})
errors['peso_ml'] = 1 / (errors['ml_error']**2)
errors['peso_mantra'] = 1 / (errors['mantra_error']**2)
errors['peso_ml_norm'] = errors['peso_ml'] / (errors['peso_ml'] + errors['peso_mantra'])
errors['peso_mantra_norm'] = 1 - errors['peso_ml_norm']
```

### 6.2 Meta-Modello — Validazione con Bootstrap a Blocchi

*(v2.2: il bootstrap originale campionava righe singole, ignorando che le osservazioni
della stessa stagione condividono contesto — correlazione within-season che un bootstrap
iid sottostima, producendo un intervallo di confidenza artificialmente stretto. Corretto a
blocchi, ricampionando stagioni intere.)*

```python
def bootstrap_meta_vs_fixed(df, season_col='season', n_bootstrap=1000, random_state=None):
    """
    Bootstrap a blocchi (per stagione) per stimare la variabilità della differenza
    di RMSE tra meta-modello e pesi fissi, rispettando la correlazione within-season.
    """
    rng = np.random.default_rng(random_state)
    seasons = df[season_col].unique()
    n_seasons = len(seasons)

    diffs = []
    for _ in range(n_bootstrap):
        sampled_seasons = rng.choice(seasons, size=n_seasons, replace=True)
        sample = pd.concat([df[df[season_col] == s] for s in sampled_seasons], ignore_index=True)
        rmse_meta = compute_rmse(sample, 'meta_pred')
        rmse_fixed = compute_rmse(sample, 'fixed_pred')
        diffs.append(rmse_fixed - rmse_meta)

    diffs = np.array(diffs)
    return np.percentile(diffs, 5), np.median(diffs), np.percentile(diffs, 95)
```

**Nota v2.4/v2.5 sul numero di stagioni disponibili**: con solo 3 stagioni, esistono al
massimo 3³ = 27 **configurazioni distinte** di ricampionamento (sequenze ordinate con
reinserimento) — non 27 campioni bootstrap in senso stretto. Questo **non invalida il
metodo** (il bootstrap a blocchi resta corretto nel non sottostimare la varianza rispetto a
un bootstrap iid), ma **limita la risoluzione** dell'intervallo di confidenza ottenibile: con
più stagioni disponibili in futuro, lo stesso metodo produrrà stime più precise.

**Ruolo del bootstrap ridimensionato** *(v2.3)*: con sole 3 stagioni, il bootstrap da solo
non è un criterio decisionale affidabile. Va usato come **supporto**, non primario. Ordine
di valutazione consigliato:

1. **Valutazione stagione per stagione** — il meta-modello migliora su ogni singola stagione?
2. **Media del RMSE** su tutte le stagioni — quanto è il miglioramento medio?
3. **Test di Diebold-Mariano** — il miglioramento medio è statisticamente significativo?
4. **Bootstrap a blocchi** — solo come controllo di coerenza aggiuntivo.

**Nota metodologica v2.5 su Diebold-Mariano**: il test assume una struttura di dipendenza
specifica sugli errori; con sole 3 stagioni la sua potenza statistica può essere limitata. Il
punto 3 non va trattato come un cancello binario isolato (solo p-value) — la decisione finale
deve bilanciare evidenza statistica, coerenza tra stagioni (punto 1) e rilevanza pratica
dell'effetto (punto 2).

**Decisione finale**: giustificato solo se il meta-modello migliora su ogni stagione, con
significatività Diebold-Mariano, e il bootstrap non contraddice (`lower > 0`). Anche se
giustificato, procedere con un periodo di **canarino in produzione** prima di sostituire i
pesi fissi, mai un deploy diretto.

**Condizioni necessarie per il meta-modello**: almeno 30-40 giocatori per ruolo nel set di
validazione (per i portieri, spesso insufficiente → usare solo pesi fissi).

---

## 7. Metriche di Successo Finali

| Metrica | Soglia Minima | Note |
|---------|---------------|------|
| **RMSE riduzione** | ≥ 3% (da calibrare, vedi Sezione 4.3) su ogni stagione | Rispetto al baseline |
| **MAE riduzione** | ≥ 3% su ogni stagione | Più robusto per portieri |
| **Significatività** | p-value < 0.05 (Diebold-Mariano) | Non da usare isolatamente — vedi Sezione 6.2 |
| **Calibrazione** | Miglioramento del Brier score | Se trasformato in classi |
| **Stabilità** | Varianza delle previsioni ridotta | Tra diversi split temporali |
| **Performance inizio stagione** | Miglioramento > 5% | Prime 5-10 giornate |

---

## 8. Piano di Monitoraggio in Produzione — Runbook

1. **Performance rolling**: finestra di 4 giornate, confronto RMSE/MAE tra modello aggiornato
   e baseline storico.
2. **Deriva delle feature**: monitorare la distribuzione delle feature MANTRA.
3. **Dashboard**: RMSE per ruolo (settimanale), feature importance (mensile), grafico di
   affidabilità (mensile).

### Runbook di Risposta agli Allarmi

*(v2.2: aggiunto criterio quantitativo esplicito al posto del giudizio soggettivo; v2.3:
percorso verso soglia adattiva; v2.4: gestione eventi eccezionali nella finestra di calibrazione)*

| Step | Azione | Tempistica | Responsabile |
|------|--------|------------|--------------|
| 1 | Notifica automatica a Slack/email del team | Immediata | Sistema di monitoraggio |
| 2 | Check rapido: confrontare il degrado del nuovo modello con quello del baseline sulla stessa finestra. **Criterio quantitativo (provvisorio)**: se il baseline degrada di una quantità comparabile (entro 1 punto percentuale) → probabile causa sistemica (dati, mercato, infortuni); se il nuovo modello degrada e il baseline no → problema specifico del nuovo modello | <1 ora | Data Scientist on-call |
| 3 | Revert automatico al modello precedente se il degrado è >8% o confermato specifico del nuovo modello | <2 ore | MLOps (script di rollback pronto) |
| 4 | Analisi post-mortem: isolare le feature responsabili (confronto distribuzioni, SHAP) | 1-2 giorni | Data Scientist |
| 5 | Rideploy solo dopo fix e nuova validazione su dati recenti | Dopo approvazione | MLOps |

**Nota v2.3 — verso una soglia adattiva.** La soglia fissa dell'1pp al punto 2 è ragionevole
ma arbitraria. Andrebbe sostituita da un criterio adattivo basato sulla distribuzione storica
del ΔRMSE (es. `ΔRMSE > media_storica + 2σ`, o 95° percentile storico) — ma questo richiede
una serie storica che appena dopo il deploy non esiste (*cold-start*). Percorso:

1. **Prime 8-10 giornate**: soglia fissa dell'1pp come criterio provvisorio.
2. **Dopo 8-10 giornate**: calcolare media e deviazione standard di ΔRMSE, passare al
   criterio adattivo.
3. **Da lì in poi**: aggiornare la distribuzione in modo incrementale (rolling window).

**Nota v2.4 — eventi eccezionali durante la calibrazione.** Se nelle prime 8-10 giornate si
verifica un evento anomalo (cambio regolamento, mercato di riparazione, infortuni insoliti),
la baseline calcolata su quella finestra sarà distorta. La finestra va **ricostruita** dopo
che l'evento è passato — chi è on-call deve poter segnalare manualmente "finestra da
ricalcolare" senza attendere il ciclo automatico.

**Pre-requisito**: il rollback va testato in staging prima del deploy.

---

## 9. Test Unitari

*(v2.2: 3 test di base; v2.3: estesi a 8; v2.4: `test_pipeline_split_safety` riscritto come
test discriminante vero, non solo smoke test; v2.5: docstring riformulato per precisione)*

```python
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def test_fit_transform_equals_fit_then_transform(sample_train_df, feature_cols):
    """fit_transform() deve produrre lo stesso risultato di fit() + transform()."""
    imp1 = MantraImputer(feature_cols=feature_cols)
    out1 = imp1.fit_transform(sample_train_df)
    imp2 = MantraImputer(feature_cols=feature_cols)
    imp2.fit(sample_train_df)
    out2 = imp2.transform(sample_train_df)
    pd.testing.assert_frame_equal(out1, out2)


def test_joblib_serializable(sample_train_df, feature_cols, tmp_path):
    """Il transformer fittato deve essere serializzabile con joblib (requisito deploy)."""
    imp = MantraImputer(feature_cols=feature_cols).fit(sample_train_df)
    path = tmp_path / "imputer.joblib"
    joblib.dump(imp, path)
    imp_loaded = joblib.load(path)
    pd.testing.assert_frame_equal(
        imp.transform(sample_train_df), imp_loaded.transform(sample_train_df)
    )


def test_fully_missing_column(sample_train_df, feature_cols):
    """Se una colonna è interamente NaN, il fallback deve ricadere sulla media globale
    (anch'essa NaN in questo caso) — comportamento atteso ed esplicito, non un crash
    silenzioso. Regola operativa: va comunque esclusa a monte, vedi Sezione 4.4."""
    df = sample_train_df.copy()
    df[feature_cols[0]] = np.nan
    imp = MantraImputer(feature_cols=feature_cols).fit(df)
    out = imp.transform(df)
    assert out[f"{feature_cols[0]}_imputed"].isna().all()


def test_column_without_nan_unchanged(sample_train_df, feature_cols):
    """Una colonna senza valori mancanti deve restare invariata dopo l'imputazione."""
    df = sample_train_df.copy()
    df[feature_cols[0]] = df[feature_cols[0]].fillna(df[feature_cols[0]].mean())
    imp = MantraImputer(feature_cols=feature_cols).fit(df)
    out = imp.transform(df)
    pd.testing.assert_series_equal(
        out[f"{feature_cols[0]}_imputed"], df[feature_cols[0]], check_names=False
    )


def test_pipeline_split_safety(feature_cols):
    """
    Test discriminante: verifica che le statistiche apprese dipendano
    esclusivamente dai dati passati a fit().

    Nota: dimostra che fit() usa soltanto il DataFrame che riceve — non
    dimostra l'immunità da OGNI possibile forma di leakage (es. stato globale
    condiviso, cache esterne, singleton). Quelle categorie di bug richiedono
    disciplina di codice e code review, non solo test unitari.
    """
    df_block_a = pd.DataFrame({
        'season': [2022] * 10, 'ruolo': ['ATT'] * 10, 'V': [6.0] * 10,
    })
    df_block_b = pd.DataFrame({
        'season': [2022] * 10, 'ruolo': ['ATT'] * 10, 'V': [9.0] * 10,
    })
    df_full = pd.concat([df_block_a, df_block_b], ignore_index=True)

    imp_fold = MantraImputer(feature_cols=['V']).fit(df_block_a)
    imp_full = MantraImputer(feature_cols=['V']).fit(df_full)

    mean_fold = imp_fold.group_means_['V'].loc[(2022, 'ATT')]
    mean_full = imp_full.group_means_['V'].loc[(2022, 'ATT')]

    assert np.isclose(mean_fold, 6.0), (
        f"Leakage sospetto: la media appresa sul fold ({mean_fold}) non corrisponde "
        f"alla media attesa del solo blocco A (6.0)."
    )
    assert not np.isclose(mean_fold, mean_full), (
        "Il test non è discriminante: le medie di fold e dataset completo coincidono."
    )


def test_pipeline_integration_smoke(sample_full_df, feature_cols):
    """Smoke test complementare: verifica solo che la Pipeline giri correttamente
    dentro cross_val_score. NON è un test di leakage — vedi test sopra per quello."""
    pipeline = Pipeline([
        ('imputer', MantraImputer(feature_cols=feature_cols)),
        ('model', SomeRegressor()),
    ])
    scores = cross_val_score(
        pipeline, sample_full_df, y, cv=3, scoring='neg_root_mean_squared_error'
    )
    assert len(scores) == 3
```

---

## 10. Piano d'Azione e Tempistiche

| Fase | Attività | Durata stimata | Responsabile |
|------|----------|----------------|--------------|
| **Fase 0** | Ispezione DB, test leakage, correlazione lag-1, calibrazione soglia 3% e soglia presenze | 2-3 giorni | Data Engineer |
| **Fase 1** | Implementazione query, feature engineering, `MantraImputer`, validazione A/B | 1 settimana | ML Engineer + Data Scientist |
| **Decisione** | Go/No-Go a due livelli basato su risultati Fase 1 | 1 giorno | Team Lead |
| **Fase 2** | Feature derivate (se Go) | 3-4 giorni | Data Scientist |
| **Fase 3** | Ensemble: pesi fissi, poi meta-modello se giustificato | 1 settimana | ML Engineer |
| **Deploy** | Integrazione in produzione, monitoraggio, runbook testato in staging | 2-3 giorni | MLOps |
| **Post-deploy** | Monitoraggio rolling, transizione a soglia adattiva dopo 8-10 giornate | Continuo | Data Scientist |

---

## 11. Rischi e Mitigazioni

| Rischio | Probabilità | Impatto | Mitigazione |
|---------|-------------|---------|-------------|
| Leakage nascosto | Media | Alto | Test automatico Sezione 3.1 + `test_pipeline_split_safety` + audit manuale delle view |
| Overfitting del meta-modello | Alta (per portieri) | Medio | Pesi fissi per portieri, LOSO per gli altri, bootstrap solo come supporto |
| Degrado performance in produzione | Bassa | Alto | Monitoraggio rolling + rollback automatico testato in staging |
| Valori mancanti sistematici | Media | Medio | Fallback gerarchico train-safe, colonne interamente NaN escluse a monte |
| Cambio di definizione feature MANTRA | Bassa | Medio | Documentazione + test di consistenza |
| Soglie arbitrarie (3%, 1pp) | Media | Medio | Calibrazione empirica in Fase 0, transizione a criterio adattivo dopo cold-start |

---

## 12. Conclusioni e Raccomandazioni

### Stato del Documento

Questo piano ha attraversato cinque cicli di revisione tecnica (v2.1 → v2.5). Non sono
emerse ulteriori criticità strutturali nella revisione più recente — un'indicazione di
robustezza, non una dimostrazione di correttezza assoluta: eventuali criticità residue
emergeranno più probabilmente dall'esperienza di implementazione e monitoraggio in
produzione che da ulteriore revisione statica del documento.

### Le tre priorità concrete prima del deploy

1. **Implementare `test_pipeline_split_safety` nella sua forma discriminante** (Sezione 9) e
   includerlo nella CI — protegge dall'errore più insidioso (fit dell'imputer fuori dal
   training fold), pur restando mirato a quella specifica classe di leakage.
2. **Calibrare la soglia del 3%** (Sezione 4.3) sulla variabilità storica del RMSE tra le
   stagioni disponibili, documentando il calcolo.
3. **Documentare nel runbook** (Sezione 8) la transizione dalla soglia fissa dell'1pp a
   quella adattiva dopo 8-10 giornate, inclusa la procedura per eventi eccezionali.

### Raccomandazioni Generali

1. **Non anticipare le fasi** — ogni fase deve superare i criteri prima di procedere.
2. **Documenta tutto** — specialmente le decisioni su collinearità, imputazione e soglie.
3. **Mantieni il modello baseline** in produzione come canarino per confronto continuo.
4. **Prepara un rollback** testato in staging, reversibile in < 30 minuti.

---

*Documento redatto da: Data Science Team*
*Data: 2026-07-29*
*Versione: Finale unificata (v2.1 + correzioni v2.2, v2.3, v2.4, v2.5)*
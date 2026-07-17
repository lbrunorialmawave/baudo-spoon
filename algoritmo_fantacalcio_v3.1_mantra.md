# ALGORITMO FANTACALCIO — v3.1 (Sistema MANTRA)

Versione definitiva: integra tutte le correzioni matematiche validate, l'architettura a 12 ruoli, e una Fase 8 di classificazioni pronte per l'asta.

**Changelog v3.0 → v3.1:**
- Reintegrati 4 dettagli implementativi che la sintesi discorsiva precedente aveva compresso (vedi note evidenziate sotto)
- Aggiunta Fase 8: Top per Ruolo, Certezze, Low Cost, Low Cost Titolari, Scommesse Multi-ruolo, Watchlist Giovani, Rischio Contestuale
- Segnalato l'uso reale di ogni input raccolto in Fase 1 (Età, Stabilità Allenatore, Cambio Squadra erano raccolti ma mai utilizzati — ora impiegati nella Fase 8)

---

## FASE 1 — INPUT DATI E PREPARAZIONE

**1. Ruolo e Polivalenza (Mantra):**
- `Ruoli_Mantra`: i ruoli ufficiali assegnati (es. "Dd;E")
- `Ruolo_Primario`: **il ruolo più arretrato (profondità minima) tra quelli posseduti** — non più "il primo elencato" (v3.1 → v3.2). Gerarchia di profondità tattica: `Por (0) → Dc,B,Dd,Ds (1) → E,M (2) → C (3) → T,W (4) → A,Pc (5)`. A parità di profondità, vince l'ordine di apparizione nella lista originale.
- `Num_Ruoli`: conteggio dei ruoli posseduti (1, 2, 3+)

**Perché il ruolo più arretrato e non il primo della lista:** un giocatore polivalente va schierato, quando possibile, nello slot più arretrato tra quelli disponibili — questo libera gli slot avanzati per chi non può giocare più indietro, massimizzando la flessibilità di modulo complessiva della rosa (es. un A/T conviene schierarlo come T: libera il proprio slot A per un attaccante puro che non potrebbe altrimenti giocare da trequartista; stessa logica per un C/T, meglio schierarlo come C). Lo stesso criterio si applica quindi alla classificazione statistica: il Ruolo_Primario riflette lo slot in cui il giocatore ha più valore tattico reale, non un artefatto dell'ordine con cui è elencato nel file.

**Nota su un'assunzione precedente, ora superata:** nella v3.1 avevo usato "il primo ruolo della lista" come convenzione, segnalando che l'ordine nei listoni ufficiali potrebbe non riflettere una vera gerarchia. La regola di profondità tattica risolve il problema alla radice: non dipende più dall'ordine arbitrario del file. **Disambiguazione residua:** se nel dataset reale è disponibile anche il `Ruolo_Classic` (P/D/C/A) abbinato, può essere usato come ulteriore verifica nei casi limite (es. un giocatore con più ruoli alla stessa profondità).

**2. Statistiche storiche (storico dei voti):**
- `Stagioni_IT`, `Min_tot` → `Min_annuo`
- `V` (media voto), `DV` (deviazione standard voto), `Pr` (% presenze, frazione 0-1) — **questo è lo storico dei voti**, alimenta il Pilastro 1

**Regole neo-arrivi (`Stagioni_IT == 0`):**
- `Min_annuo`: forfettario — 2000' per titolare designato, 500' per riserva/scommessa
- `V`, `DV`: mediana del Ruolo_Primario (non media, per non farsi influenzare dagli outlier di ruolo)
- `Pr`: **due rami distinti**, non un valore unico — `0,75` se titolare designato sulla carta, `0,40` se riserva/scommessa. *(Punto reintegrato — la sintesi precedente riportava solo "0,75", perdendo la distinzione più utile: senza il ramo a 0,40 un neo-acquisto di panchina verrebbe sistematicamente sopravvalutato in Pr.)*

**3. Statistiche avanzate (per-90', allineate temporalmente):**
- `xG90, xA90, G90, A90`, calcolate sulle **ultime 38 partite effettivamente giocate** (tetto di recency: non oltre 24 mesi)

**4. Contesto:**
- `PS_corretto` (Peso Squadra 0-100, aggiustato post-mercato)
- `Età`, `Stabilità Allenatore` ∈ {Alta, Media, Bassa}, `Cambio Squadra` ∈ {Sì, No} — **raccolti ma non usati in nessun Pilastro**; impiegati solo nella Fase 8 (Watchlist Giovani e Rischio Contestuale) per non introdurre nuovi bias nei punteggi numerici core

**5. Mercato (storico dei valori):**
- `Pz1, Pz2, Pz3`: prezzi d'asta ultimi 3 anni (`Pz1 = 0` se mai a listone) — **questo è lo storico dei valori**, alimenta interamente il Pilastro 4

---

## FASE 2 — I 4 PILASTRI (scala 0-100)

### Pilastro 1 — Solidità (peso 30%)
```
P1 = min(Min_annuo/2700, 1)*30 + (V/10)*25 + (1/(1+DV))*20 + Pr*25
```

### Pilastro 2 — Potenziale (peso 30%)

**Pool Esteso — tabella completa di fusione** *(punto reintegrato — la sintesi precedente citava solo "B si fonde con Dc,Dd,Ds" come esempio, omettendo le altre coppie):*

| Ruolo piccolo | Si fonde con |
|---|---|
| B | Dc, Dd, Ds |
| Dd | Dc, Ds, B |
| Ds | Dc, Dd, B |
| E | M |
| M | E |
| Pc | A |
| A | Pc |
| T | W |
| W | T |

Regola: se il pool del Ruolo_Primario ha meno di 20 giocatori, media e devstd si calcolano sul gruppo affine (il giocatore resta comunque etichettato col proprio ruolo).

**Filtro sulla baseline, non solo sul singolo giocatore** *(punto reintegrato — il più importante dei quattro: la sintesi diceva solo "se Min_annuo < 450, z-score forzato a 0", ma non specificava che anche il calcolo di media/devstd del pool deve escludere chi sta sotto quella soglia. Senza questa esclusione, i campioni piccoli/rumorosi (es. un gol fortunato in 10 minuti) inquinano comunque la baseline usata per tutti gli altri, vanificando la protezione che la soglia doveva garantire):*

```
Pool statistico = SOLO giocatori con Min_annuo >= 450 (nel gruppo di ruolo/fuso)

SE Min_annuo < 450 (per il singolo giocatore):
    z_qualita = 0, z_output = 0
ALTRIMENTI:
    z_qualita = zscore(xG90+xA90, pool_statistico)
    z_output  = zscore(G90+A90, pool_statistico)

P2 = clip(50 + (z_qualita*0,60 + z_output*0,40)*15, 0, 100)
```

**Pilastro 2bis — Portieri:** stessa logica, basata su gol evitati attesi (`z_parate`), clean sheet (`z_clean`), uscite (`z_uscite`), pesi 0,50/0,35/0,15.

### Pilastro 3 — Peso Squadra (peso 20%)

| Ruolo | Coeff_Base |
|---|---|
| Por | 0,0025 |
| Dc, B, Dd, Ds | 0,003 |
| E, M | 0,0035 |
| C | 0,0038 |
| T, W | 0,0042 |
| A, Pc | 0,004 |

```
Moltiplicatore = 1 + max(0, (PS_corretto-50)*Coeff_Base)
Max_Moltiplicatore = 1 + (100-50)*Coeff_Base
P3 = clip(PS_corretto * Moltiplicatore / Max_Moltiplicatore, 0, 100)
```

### Pilastro 4 — Mercato Storico (peso 20%)

```
CP = P1*0,2 + P2*0,3 + P3*0,5    // Costo Potenziale, derivato dai pilastri (non un dato di mercato indipendente)

Picco = MAX(Pz1,Pz2,Pz3)
Trend = SE Pz1==0 ALLORA 0 ALTRIMENTI (Pz3-Pz1)/(Pz1+5)

Livello = (CP / CP_max_ruolo) * 30      // max 30
Picco_c = clip(Picco/max(CP,1), 0, 2) * 25   // max 50
Trend_c = clip(Trend*20+50, 0, 100) * 0,2    // max 20

P4 = clip(Livello + Picco_c + Trend_c, 0, 100)
```

**`CP_max_ruolo` usa lo stesso Pool Esteso del Pilastro 2** *(punto reintegrato — non era esplicitato che la fusione pool si applica anche qui e in Fase 4/5, non solo nel Pilastro 2. Senza questa precisazione, i ruoli rari tornerebbero a soffrire di pool troppo piccoli proprio nei calcoli che più contano per il prezzo finale.)*

---

## FASE 3 — FANTAPUNTO GREZZO
```
FP = P1*0,30 + P2*0,30 + P3*0,20 + P4*0,20
```

## FASE 4 — STANDARDIZZAZIONE FP (per ruolo, con Pool Esteso)
```
FP_std = (FP - media_FP_pool) / devstd_FP_pool     // pool = stesso Pool Esteso del Pilastro 2/4
k = clip(1/%(FP_std>1,5 nel pool), 1, 6)
FP_Corr = clip(50 + 50*tanh(FP_std*k/10), 0, 100)
```

## FASE 5 — STANDARDIZZAZIONE CP (per ruolo, con Pool Esteso)
```
CP_std = (CP - media_CP_pool) / devstd_CP_pool
CP_Corr = clip(50 + CP_std*10, 5, 100)
```

## FASE 6 — FLESSIBILITÀ, VALORE REALE, PREZZO MASSIMO
```
Fattore_Flessibilità: 1 ruolo→1,00 | 2 ruoli→1,05 | 3+ ruoli→1,08
FP_Mantra = clip(FP_Corr * Fattore_Flessibilità, 0, 100)

Fattore_Eroe = clip(1 + (1 - CP/CP_medio_tutti)*0,5, 0,6, 1,6)

VR = clip((FP_Mantra * Fattore_Eroe / CP_Corr)*100, 0, 300)
Prezzo_Massimo = max(CP*(VR/100), 1)
```

## FASE 7 — REGOLE DECISIONALI (mutuamente esclusive, in ordine)
```
1. 🏆 TOP           → FP > 80
2. 💎 AFFARE        → FP > 60 E VR > 140
3. 🔄 SCOMMESSA     → FP < 50 E VR > 130
4. ⚠️ SOPRAVALUTATO → VR < 80
5. ⚖️ GIUSTO        → 90 ≤ VR ≤ 110
6. (nessuna)        → tutti gli altri
```

---

## FASE 8 — CLASSIFICAZIONI E REPORT D'ASTA (nuova)

Queste categorie sono **filtri e ordinamenti sulle colonne già calcolate** — non introducono nuovi pilastri né toccano i punteggi numerici, per non alterare quanto già validato.

### A. Top per Ruolo (Primario)
Per ciascuno dei 12 ruoli Mantra, i migliori N (es. 10-15) tra i giocatori il cui **Ruolo_Primario** è quello — ordinati per `FP_Corr_MANTRA` decrescente. Riflette la classificazione statistica "ufficiale" del giocatore (quella usata anche per pool, PS e budget).

### A2. Per Ruolo Mantra (multi-eleggibilità) — nuovo
Stessa logica del punto A, ma un giocatore compare **in ogni ruolo che può ricoprire**, non solo nel suo Ruolo_Primario: un A/T compare sia tra gli A sia tra i T. Utile in asta quando ti serve riempire uno slot specifico e vuoi vedere anche chi lo copre come ruolo secondario, non solo chi lo ha come ruolo principale. *(I punteggi mostrati — FP_Corr_MANTRA, VR, ecc. — restano quelli calcolati sul Ruolo_Primario del giocatore: qui cambia solo il criterio di raggruppamento/visualizzazione, non il calcolo sottostante.)*

### B. Certezze (basso rischio)
```
Filtro: Stagioni_IT >= 2  E  Pr >= 0,70  E  DV <= mediana_ruolo(DV)  E  P1 >= 70
Ordina per: P1 decrescente
```
Giocatori con track record consolidato, alta continuità, bassa variabilità di rendimento — il profilo "sicuro" da schierare senza sorprese.

### C. Low Cost (occasioni economiche)
```
Filtro: Prezzo_Massimo <= soglia_budget (parametrizzabile, default 15 crediti)  E  VR > 110
Ordina per: VR decrescente
```
Sovrappone parzialmente ad AFFARE/SCOMMESSA di Fase 7, ma qui il criterio guida è il **budget assoluto**, non solo il rapporto qualità/prezzo — utile per l'ultima fase d'asta con pochi crediti rimasti.

### D. Low Cost Titolari
```
Sottoinsieme di C, con l'aggiunta: Pr >= 0,65
```
Distingue i low cost che **giocano sicuro** dai low cost "lotteria" (jolly panchinari con VR alto ma minutaggio incerto) — la differenza pratica tra un vero titolare a basso prezzo e una scommessa che potrebbe restare in panchina.

### E. Scommesse Multi-ruolo
```
Tra i 🔄 SCOMMESSA di Fase 7, priorità a Num_Ruoli >= 2
```
Le scommesse a basso costo che, oltre al potenziale, coprono più caselle tattiche — minimizzano il rischio anche se il giocatore non esplode, perché restano comunque utili come jolly.

### F. Watchlist Giovani (usa finalmente `Età`)
```
Filtro: Età <= 23  E  Trend_Inizio > 0
```
Giovani il cui prezzo di mercato è in crescita anno su anno, indipendentemente dall'FP attuale — segnala potenziale futuro anche quando i pilastri di rendimento immediato non sono ancora al top (tipico di un giovane in rampa di lancio con minutaggio ancora limitato).

### G. Rischio Contestuale (annotazione, non ranking)
```
Flag testuale (non modifica alcun punteggio):
  "⚠️ Cambio Squadra" se Cambio_Squadra == Sì
  "⚠️ Allenatore instabile" se Stabilità_Allenatore == Bassa
```
Da affiancare a qualunque altra etichetta come avviso qualitativo — un giocatore può essere 🏆 TOP e avere comunque il flag "cambio squadra", segnalando che il suo rendimento storico potrebbe essere meno predittivo del solito.

---

## EXTRA — BUDGET (500 crediti / 8 squadre / 6 blocchi) — invariato

| Blocco | Ruoli | Crediti |
|---|---|---|
| Portieri | Por | 30 |
| Difesa Pura | Dc, B, Dd, Ds | 70 |
| Ibridi Difensivi | E, M | 60 |
| Centro Nevralgico | C | 90 |
| Linea Fantasia | T, W | 90 |
| Attacco | A, Pc | 160 |
| **Totale** | | **500** ✓ |

---

## NOTA CALIBRAZIONE (invariata dalla v3.0)
Coefficienti da validare sui dati reali prima dell'asta: 15 nel Pilastro 2, cap k a 6, 0,6-1,6 sul Fattore Eroe, soglie 2700'/450', soglia pool 20, Coeff_Base per ruolo (Pilastro 3), valori Fattore Flessibilità. Nuovi da calibrare in Fase 8: soglia budget "Low Cost" (default 15 crediti), soglia età "giovane" (default 23).

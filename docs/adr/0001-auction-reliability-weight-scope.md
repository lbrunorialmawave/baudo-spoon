# ADR 0001 — Scope del reliability weight decisionale in Auction

## Contesto
plan-limited-cohort-hardening.md WS3 poneva la scelta tra:
- Opzione A ("shrink-once"): solo il projected_score già shrinkato in output è la
  fonte di sconto, sia per Optimizer sia per Auction.
- Opzione B ("shrink + decision-weight ovunque"): entrambi i moduli applicano anche
  reliability_weight (e risk_aversion via prediction_std) sopra al projected_score
  shrinkato.

## Decisione
Opzione B. Lo shrink di output è calibrato per la *presentazione* (evitare numeri
assurdi a schermo); il reliability_weight decisionale serve a penalizzare la
*fiducia nella decisione automatica* (selezione rosa, ranking d'asta). Sono due
scopi distinti anche se derivano dagli stessi minuti — è corretto applicarli
entrambi quando la decisione è automatica.

## Conseguenza
`VarEngine.apply_reliability_weight` passa da default `False` a default `True`
in `AuctionConfigSchema` e `ml.auction.models.AuctionConfig`. `risk_aversion`
resta opt-in (default `0.0`) finché non è calibrato su dati reali.

### Calibrazione risk_aversion (piano)
1. Raccogliere prediction_std e ranking VAR su una stagione completa in SHADOW.
2. Backtest: confrontare hit-rate top-N con risk_aversion ∈ {0.0, 0.25, 0.5, 1.0}.
3. Solo dopo backtest, promuovere un default non-zero nei preset d'asta.

## Alternative scartate
Opzione A: più semplice, ma lascia il caso Adzic-in-Auction irrisolto in modo
sistematico (senza questo cambio, l'Auction non applica alcuno sconto aggiuntivo
oltre allo shrink di display, a differenza dell'Optimizer).

## Riferimenti
- plan-limited-cohort-hardening.md WS3
- plan-limited-cohort-patches.md G1/G2

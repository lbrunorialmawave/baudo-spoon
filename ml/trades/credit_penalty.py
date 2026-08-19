"""Credit penalty on transfers — formalised from the league rule.

Ad ogni trasferimento il valore è ridotto del ``decay_percent``% del prezzo
di acquisto originario (arrotondamento half-up), fino a un pavimento pari al
``floor_percent``% del prezzo originario; raggiunto il pavimento il valore
resta invariato.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def round_half_up(value: float) -> int:
    """≥0.50 → up; <0.50 → down. Uses ROUND_HALF_UP, not banker's rounding."""
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def recompute_value_on_transfer(
    original_purchase_price: int,
    current_value: int,
    decay_step_percent: float = 25.0,
    floor_percent: float = 25.0,
) -> int:
    """Ricalcola il valore di un giocatore al momento di un trasferimento.

    Applicato SOLO se credit_penalty_enabled=True per la fantasy_league.
    Idempotente: chiamare due volte con current_value già al floor non lo
    abbassa oltre.

    Parameters
    ----------
    original_purchase_price:
        Prezzo di acquisto originario (asta).
    current_value:
        Valore corrente prima di questo trasferimento.
    decay_step_percent:
        Percentuale del prezzo originario da sottrarre ad ogni trasferimento.
    floor_percent:
        Pavimento come % del prezzo originario (mai sotto 1 credito).
    """
    if original_purchase_price < 0:
        raise ValueError("original_purchase_price must be >= 0")
    if current_value < 0:
        raise ValueError("current_value must be >= 0")

    step = round_half_up(original_purchase_price * decay_step_percent / 100.0)
    floor = max(round_half_up(original_purchase_price * floor_percent / 100.0), 1)
    # If original was 0, floor stays 1 only when there was a positive price;
    # for zero-price keep 0.
    if original_purchase_price == 0:
        return 0

    new_value = current_value - step
    return max(new_value, floor)

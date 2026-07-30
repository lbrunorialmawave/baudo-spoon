"""Regression test: SEASON_VALUE mode fixes the appearances-blind ranking bias."""

from ml.auction.models import ValuationMode
from ml.auction.var import DemandCurve, VarEngine


def _pool():
    """Pool of midfielders designed to trigger the ranking-flip.

    Player flash: high per-match rating (8.0), few appearances → low season_value (4.0)
    Player workhorse: lower per-match rating (6.5), many appearances → high season_value (9.0)

    PER_MATCH_RATING should rank flash above workhorse (8.0 > 6.5).
    SEASON_VALUE should rank workhorse above flash (9.0 > 4.0).

    Season values are scaled to the same range as per-match ratings to avoid
    demand-curve distortion in the test.
    """
    base = [
        {
            "player_id": "flash",
            "role": "C",
            "projected_score": 8.0,
            "season_value": 4.0,  # great per-match but rarely plays
        },
        {
            "player_id": "workhorse",
            "role": "C",
            "projected_score": 6.5,
            "season_value": 9.0,  # solid per-match and always plays
        },
    ]
    for i in range(10):
        base.append(
            {
                "player_id": f"filler_{i}",
                "role": "C",
                "projected_score": 5.0 + i * 0.2,
                "season_value": 5.0 + i * 0.3,
            }
        )
    return base


def _linear_curve():
    """Near-linear demand curve to isolate VAR ranking from pricing convexity."""
    return DemandCurve(base_price=1.0, scale=0.5, exponent=1.0, calibrated=True)


def test_per_match_rating_ranks_flash_higher():
    engine = VarEngine(
        valuation_mode=ValuationMode.PER_MATCH_RATING,
        demand_curve=_linear_curve(),
    )
    results = engine.evaluate(_pool())
    ranked_ids = [r.player_id for r in results]
    assert ranked_ids.index("flash") < ranked_ids.index("workhorse")


def test_season_value_ranks_workhorse_higher():
    engine = VarEngine(
        valuation_mode=ValuationMode.SEASON_VALUE,
        demand_curve=_linear_curve(),
    )
    results = engine.evaluate(_pool())
    ranked_ids = [r.player_id for r in results]
    assert ranked_ids.index("workhorse") < ranked_ids.index("flash")


def test_season_value_falls_back_to_projected_score_when_missing():
    """When season_value is None, SEASON_VALUE mode uses projected_score with no crash."""
    pool = [
        {"player_id": "p1", "role": "D", "projected_score": 7.0, "season_value": None},
        {"player_id": "p2", "role": "D", "projected_score": 6.0, "season_value": 200.0},
    ]
    engine = VarEngine(valuation_mode=ValuationMode.SEASON_VALUE)
    results = engine.evaluate(pool)
    assert len(results) == 2


def test_default_mode_is_per_match_rating():
    engine = VarEngine()
    assert engine.valuation_mode == ValuationMode.PER_MATCH_RATING

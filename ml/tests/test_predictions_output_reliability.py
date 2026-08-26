"""PR9: PlayerPredictionSchema maps artifact keys and exposes output-reliability fields."""

from __future__ import annotations

from api.src.schemas import NextSeasonPredictionSchema, PlayerPredictionSchema


def test_player_prediction_schema_maps_artifact_keys_and_pr9_fields() -> None:
    """Router builds PlayerPredictionSchema from raw artifact rows.

    Artifact keys are season_start / predicted_fantavoto /
    predicted_fantavoto_display; the public schema surfaces season /
    predicted / predicted_display (plus sample_cohort / ml_values_noisy).
    """
    raw = {
        "player_name": "Test Player",
        "player_fotmob_id": 42,
        "team_name": "Test FC",
        "canonical_role": "A",
        "season_start": 2025,
        "fantavoto_medio": 6.5,
        "predicted_fantavoto": 7.8,
        "confidence": 0.7,
        "prediction_interval_low": 6.9,
        "prediction_interval_high": 8.7,
        "expected_minutes": 450.0,
        "sample_cohort": "LIMITED",
        "ml_values_noisy": True,
        "predicted_fantavoto_display": 6.9,
    }

    # Mirror the mapping performed in list_predictions (intelligence.py).
    item = PlayerPredictionSchema(
        player_name=raw["player_name"],
        player_fotmob_id=raw.get("player_fotmob_id"),
        team_name=raw.get("team_name"),
        canonical_role=raw.get("canonical_role"),
        # season_start in the artifact is a JSON int; PlayerPredictionSchema.season
        # is str, and Pydantic v2 doesn't coerce int -> str.
        season=str(raw["season_start"]) if raw.get("season_start") is not None else raw.get("season"),
        fantavoto_medio=raw.get("fantavoto_medio"),
        predicted=float(
            raw["predicted_fantavoto"]
            if raw.get("predicted_fantavoto") is not None
            else raw.get("predicted", 0.0)
        ),
        confidence=raw.get("confidence"),
        prediction_interval_low=raw.get("prediction_interval_low"),
        prediction_interval_high=raw.get("prediction_interval_high"),
        expected_minutes=raw.get("expected_minutes"),
        sample_cohort=raw.get("sample_cohort"),
        ml_values_noisy=raw.get("ml_values_noisy"),
        predicted_display=(
            raw.get("predicted_fantavoto_display")
            if raw.get("predicted_fantavoto_display") is not None
            else raw.get("predicted_display")
        ),
    )

    dumped = item.model_dump(by_alias=True)
    assert dumped["playerName"] == "Test Player"
    assert dumped["season"] == "2025"
    assert dumped["predicted"] == 7.8
    assert dumped["sampleCohort"] == "LIMITED"
    assert dumped["mlValuesNoisy"] is True
    assert dumped["predictedDisplay"] == 6.9
    # Raw predicted must remain the undamped value for metrics consumers.
    assert item.predicted == 7.8
    assert item.predicted_display == 6.9


def test_player_prediction_schema_fallback_for_legacy_artifact_keys() -> None:
    """Older artifacts may still use season / predicted without _display."""
    item = PlayerPredictionSchema(
        player_name="Legacy",
        season="2024",
        predicted=6.2,
        sample_cohort=None,
        ml_values_noisy=None,
        predicted_display=None,
    )
    dumped = item.model_dump(by_alias=True)
    assert dumped["predicted"] == 6.2
    assert dumped["sampleCohort"] is None
    assert dumped["mlValuesNoisy"] is None
    assert dumped["predictedDisplay"] is None


def test_next_season_prediction_schema_accepts_pr9_fields() -> None:
    raw = {
        "player_name": "Next Star",
        "player_fotmob_id": 99,
        "predicted_next_fantavoto": 8.1,
        "sample_cohort": "STANDARD",
        "ml_values_noisy": False,
        "predicted_next_fantavoto_display": 8.1,
    }
    item = NextSeasonPredictionSchema(**raw)
    dumped = item.model_dump(by_alias=True)
    assert dumped["predictedNextFantavoto"] == 8.1
    assert dumped["sampleCohort"] == "STANDARD"
    assert dumped["mlValuesNoisy"] is False
    assert dumped["predictedNextFantavotoDisplay"] == 8.1

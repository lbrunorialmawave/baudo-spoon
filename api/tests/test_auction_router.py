"""Integration tests for the auction router.

Usa un'app FastAPI minimale che monta solo ``auction.router`` per evitare
il lifespan globale (Redis, DataRepository).  Il test copre tutti gli
8 endpoint del router + il caso 404 di sessione inesistente.
"""

from __future__ import annotations

from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import auction as auction_router

# ---------------------------------------------------------------------------
# App minimale: solo router auction
# ---------------------------------------------------------------------------

app = FastAPI()
app.include_router(auction_router.router)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Pool helpers
# ---------------------------------------------------------------------------


def _player(pid: str, role: str, cost: int, score: float) -> dict[str, object]:
    return {
        "playerId": pid,
        "name": pid.upper(),
        "role": role,
        "realTeam": "TEST",
        "cost": cost,
        "projectedScore": score,
    }


def _participant(pid: str, budget: int = 500) -> dict[str, object]:
    return {
        "participantId": pid,
        "displayName": pid,
        "budgetInitial": budget,
    }


@pytest.fixture
def default_pool() -> list[dict[str, object]]:
    return [
        _player("p1", "P", 30, 8.0),
        _player("p2", "P", 25, 7.5),
        _player("d1", "D", 22, 7.0),
        _player("c1", "C", 30, 7.5),
        _player("a1", "A", 40, 8.0),
    ]


@pytest.fixture
def init_payload(default_pool: list[dict[str, object]]) -> dict[str, object]:
    return {
        "seasonStart": 2025,
        "participants": [
            _participant("u1"),
            _participant("u2"),
        ],
        "config": {
            "numParticipants": 2,
        },
        "playerPool": default_pool,
    }


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


class TestInit:
    def test_init_ok(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        resp = client.post("/auction/init", json=init_payload)
        assert resp.status_code == 201
        body = resp.json()
        assert "sessionId" in body
        assert isinstance(body["sessionId"], str)
        assert len(body["sessionId"]) > 0

    def test_init_partecipanti_duplicati(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        bad = cast(dict[str, object], init_payload.copy())
        bad["participants"] = [_participant("u1"), _participant("u1")]
        resp = client.post("/auction/init", json=bad)
        # initialize_auction raises ValueError -> 500 (unhandled exception handler)
        # oppure 422 se Pydantic valida prima.  Accettiamo entrambi per ora.
        assert resp.status_code in (422, 500)

    def test_init_num_participants_mismatch(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        bad = cast(dict[str, object], init_payload.copy())
        bad["config"] = {"numParticipants": 5}
        resp = client.post("/auction/init", json=bad)
        assert resp.status_code in (422, 500)

    def test_init_con_budget_custom_ok(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        """referenceBudget=300 e budgetInitial=500 devono passare."""
        cfg = cast(dict[str, object], init_payload["config"])
        cfg["referenceBudget"] = 300
        cfg["budgetInitial"] = 500
        resp = client.post("/auction/init", json=init_payload)
        assert resp.status_code == 201
        assert resp.json().get("sessionId")

    def test_init_defaults_budget_300(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        """Senza i campi nel payload, i default 300/300 devono essere accettati."""
        cfg = cast(dict[str, object], init_payload["config"])
        # Assicuriamoci che i campi siano assenti
        cfg.pop("referenceBudget", None)
        cfg.pop("budgetInitial", None)
        resp = client.post("/auction/init", json=init_payload)
        assert resp.status_code == 201

    def test_init_reference_budget_zero_rifiutato(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        """referenceBudget <= 0 deve essere rifiutato da Pydantic (422)."""
        cfg = cast(dict[str, object], init_payload["config"])
        cfg["referenceBudget"] = 0
        resp = client.post("/auction/init", json=init_payload)
        assert resp.status_code == 422

    def test_init_budget_initial_negativo_rifiutato(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        """budgetInitial <= 0 deve essere rifiutato da Pydantic (422)."""
        cfg = cast(dict[str, object], init_payload["config"])
        cfg["budgetInitial"] = -10
        resp = client.post("/auction/init", json=init_payload)
        assert resp.status_code == 422

    def test_init_serialize_round_trip_preserva_budget(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        """referenceBudget e budgetInitial sopravvivono al round-trip."""
        cfg = cast(dict[str, object], init_payload["config"])
        cfg["referenceBudget"] = 300
        cfg["budgetInitial"] = 500

        sid = _init_session(client, init_payload)
        ser_resp = client.get(f"/auction/{sid}/serialize")
        payload = ser_resp.json()["payload"]

        # Verifica che i campi siano nel payload serializzato
        assert payload["config"]["reference_budget"] == 300
        assert payload["config"]["budget_initial"] == 500

        # Verifica che il round-trip via /deserialize preservi i campi
        deser_resp = client.post("/auction/deserialize", json=payload)
        assert deser_resp.status_code == 201
        new_sid = deser_resp.json()["sessionId"]
        new_ser = client.get(f"/auction/{new_sid}/serialize").json()["payload"]
        assert new_ser["config"]["reference_budget"] == 300
        assert new_ser["config"]["budget_initial"] == 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_session(
    client: TestClient, payload: dict[str, object]
) -> str:
    resp = client.post("/auction/init", json=payload)
    assert resp.status_code == 201
    return cast(str, resp.json()["sessionId"])


# ---------------------------------------------------------------------------
# record + 4-step validation
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_ok(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.post(
            f"/auction/{sid}/record",
            json={
                "playerId": "p1",
                "winnerParticipantId": "u1",
                "finalPrice": 20,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["sequenceNumber"] == 1
        assert "priceIndexAfter" in body
        assert body["rejectionCode"] is None

    def test_record_unknown_player(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.post(
            f"/auction/{sid}/record",
            json={
                "playerId": "ghost",
                "winnerParticipantId": "u1",
                "finalPrice": 1,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["rejectionCode"] == "unknown_player"

    def test_record_unknown_participant(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.post(
            f"/auction/{sid}/record",
            json={
                "playerId": "p1",
                "winnerParticipantId": "ghost",
                "finalPrice": 1,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["rejectionCode"] == "unknown_participant"

    def test_record_credit_reserve_violation(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        # u1 budget 500, 25 slot totali → max_allowed = 500 - 24 = 476
        resp = client.post(
            f"/auction/{sid}/record",
            json={
                "playerId": "p1",
                "winnerParticipantId": "u1",
                "finalPrice": 477,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["rejectionCode"] == "credit_reserve_violation"


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------


class TestUndo:
    def test_undo_ok(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        client.post(
            f"/auction/{sid}/record",
            json={
                "playerId": "p1",
                "winnerParticipantId": "u1",
                "finalPrice": 20,
            },
        )
        resp = client.post(f"/auction/{sid}/undo")
        assert resp.status_code == 200
        body = resp.json()
        assert body["assignments"] == []

    def test_undo_su_sessione_vuota(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.post(f"/auction/{sid}/undo")
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------


class TestProjection:
    def test_projection_ok(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.get(f"/auction/{sid}/projection/p1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["playerId"] == "p1"
        assert body["tier"] in ("LOW", "MID", "TOP")
        assert isinstance(body["expectedPrice"], (int, float))

    def test_projection_player_non_disponibile(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        # Assegna p1 a u1
        client.post(
            f"/auction/{sid}/record",
            json={
                "playerId": "p1",
                "winnerParticipantId": "u1",
                "finalPrice": 1,
            },
        )
        # p1 ora non è più in available_pool → 404
        resp = client.get(f"/auction/{sid}/projection/p1")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# alternatives
# ---------------------------------------------------------------------------


class TestAlternatives:
    def test_alternatives_ok(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.get(f"/auction/{sid}/alternatives/p1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["targetPlayerId"] == "p1"

    def test_alternatives_target_non_esiste(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.get(f"/auction/{sid}/alternatives/ghost")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# pool (ricerca per nome)
# ---------------------------------------------------------------------------


class TestPool:
    def test_pool_full(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.get(f"/auction/{sid}/pool")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)
        assert len(body) == 5  # default_pool ha 5 player
        # Ogni elemento ha la shape di AuctionPlayerSummarySchema
        for item in body:
            assert set(item.keys()) == {
                "playerId",
                "name",
                "realTeam",
                "role",
                "cost",
                "projectedScore",
            }

    def test_pool_filter_substring_case_insensitive(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        # di default ogni player.name = playerId.upper() (es. "P1", "D1")
        sid = _init_session(client, init_payload)
        resp = client.get(f"/auction/{sid}/pool?q=p")
        assert resp.status_code == 200
        body = resp.json()
        # "p" è contenuto in "P1" e "P2" (case-insensitive)
        assert {item["playerId"] for item in body} == {"p1", "p2"}

    def test_pool_filter_no_match(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.get(f"/auction/{sid}/pool?q=zzz")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_pool_filter_assegnato_non_piu_disponibile(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        # Assegna p1 → ora p1 esce da available_pool
        client.post(
            f"/auction/{sid}/record",
            json={
                "playerId": "p1",
                "winnerParticipantId": "u1",
                "finalPrice": 10,
            },
        )
        resp = client.get(f"/auction/{sid}/pool")
        assert resp.status_code == 200
        body = resp.json()
        assert {item["playerId"] for item in body} == {"p2", "d1", "c1", "a1"}

    def test_pool_sessione_inesistente(self, client: TestClient) -> None:
        resp = client.get("/auction/ghost/pool")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_ok(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.get(f"/auction/{sid}/summary")
        assert resp.status_code == 200
        body = resp.json()
        assert "participants" in body
        assert "assignments" in body
        assert "priceIndex" in body
        assert len(body["participants"]) == 2


# ---------------------------------------------------------------------------
# serialize / deserialize
# ---------------------------------------------------------------------------


class TestSerialize:
    def test_serialize_ok(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        client.post(
            f"/auction/{sid}/record",
            json={
                "playerId": "p1",
                "winnerParticipantId": "u1",
                "finalPrice": 10,
            },
        )
        resp = client.get(f"/auction/{sid}/serialize")
        assert resp.status_code == 200
        body = resp.json()
        assert "payload" in body
        assert "assignments" in body["payload"]

    def test_deserialize_round_trip(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        client.post(
            f"/auction/{sid}/record",
            json={
                "playerId": "p1",
                "winnerParticipantId": "u1",
                "finalPrice": 10,
            },
        )
        ser_resp = client.get(f"/auction/{sid}/serialize")
        payload = ser_resp.json()["payload"]

        deser_resp = client.post("/auction/deserialize", json=payload)
        assert deser_resp.status_code == 201
        new_sid = deser_resp.json()["sessionId"]
        assert new_sid != sid

        # Verifica summary: deve avere 1 assignment
        summary_resp = client.get(f"/auction/{new_sid}/summary")
        assert summary_resp.status_code == 200
        assert len(summary_resp.json()["assignments"]) == 1

    def test_deserialize_payload_invalido(self, client: TestClient) -> None:
        resp = client.post("/auction/deserialize", json={"foo": "bar"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# discard (DELETE)
# ---------------------------------------------------------------------------


class TestDiscard:
    def test_discard_ok(
        self, client: TestClient, init_payload: dict[str, object]
    ) -> None:
        sid = _init_session(client, init_payload)
        resp = client.delete(f"/auction/{sid}")
        assert resp.status_code == 204
        # Summary dopo discard deve dare 404
        summary_resp = client.get(f"/auction/{sid}/summary")
        assert summary_resp.status_code == 404

    def test_discard_sessione_inesistente(self, client: TestClient) -> None:
        resp = client.delete("/auction/ghost_session")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Sessione inesistente
# ---------------------------------------------------------------------------


class TestSessionNotFound:
    def test_summary_sessione_inesistente(self, client: TestClient) -> None:
        resp = client.get("/auction/ghost/summary")
        assert resp.status_code == 404

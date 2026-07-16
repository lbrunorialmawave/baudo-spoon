"""Tests for the stateful orchestrator.

Covers (per spec §4, §8):
* ``initialize_auction`` validation (num_participants, unique ids,
  inflation config coherence).
* ``record_assignment`` 4-step validation in order, with explicit
  ``rejection_code`` for each case.
* ``undo_last_assignment`` determinism (budget, roster, price_index,
  available_pool all rolled back).
* ``get_auction_summary`` snapshot.
* ``serialize_state`` / ``deserialize_state`` round-trip.
* ``AuctionSession`` façade.
"""

from __future__ import annotations

import copy
from typing import cast

import pytest

from ml.auction.models import (
    AlternativesConfig,
    AuctionConfig,
    MarketDriftConfig,
    ParticipantSetup,
)
from ml.auction.orchestrator import (
    AuctionSession,
    deserialize_state,
    get_auction_summary,
    initialize_auction,
    record_assignment,
    serialize_state,
    undo_last_assignment,
)
from ml.optimizer.inflation import InflationConfig
from ml.optimizer.models import Player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _participants(n: int = 4) -> list[ParticipantSetup]:
    return [
        ParticipantSetup(
            participant_id=f"u{i}", display_name=f"User {i}", budget_initial=500
        )
        for i in range(1, n + 1)
    ]


def _mk(
    pid: str,
    name: str,
    role: str,
    cost: int,
    score: float,
    team: str = "TEST",
) -> Player:
    return Player(
        player_id=pid,
        name=name,
        real_team=team,
        role=role,  # type: ignore[arg-type]
        cost=cost,
        projected_score=score,
    )


@pytest.fixture
def mid_pool() -> list[Player]:
    """Pool misto per test di orchestrazione (4 GK, 6 D, 4 C, 4 A)."""
    return [
        _mk("p1", "GK1", "P", 30, 8.0, "A"),
        _mk("p2", "GK2", "P", 25, 7.5, "B"),
        _mk("p3", "GK3", "P", 20, 7.0, "C"),
        _mk("p4", "GK4", "P", 15, 6.5, "D"),
        _mk("d1", "DF1", "D", 22, 7.0, "A"),
        _mk("d2", "DF2", "D", 18, 6.5, "B"),
        _mk("d3", "DF3", "D", 15, 6.0, "C"),
        _mk("d4", "DF4", "D", 10, 5.5, "D"),
        _mk("c1", "MF1", "C", 30, 7.5, "A"),
        _mk("c2", "MF2", "C", 25, 7.0, "B"),
        _mk("a1", "FW1", "A", 40, 8.0, "A"),
        _mk("a2", "FW2", "A", 35, 7.5, "B"),
    ]


# ---------------------------------------------------------------------------
# initialize_auction
# ---------------------------------------------------------------------------


class TestInitializeAuction:
    def test_setup_valido(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(_participants(4), AuctionConfig(num_participants=4), mid_pool)
        assert len(state.participants) == 4
        for ps in state.participants.values():
            assert ps.budget_residual == 500
            assert ps.squad == []
            assert ps.role_breakdown == {"P": 0, "D": 0, "C": 0, "A": 0}
        assert state.assignments == []
        # Price index inizializzato a 1.0 per tutte le combinazioni
        for role in ("P", "D", "C", "A"):
            for tier in ("LOW", "MID", "TOP"):
                assert state.price_index[role][tier] == 1.0
        # Pool iniziale = pool completo
        assert {p.player_id for p in state.available_pool} == {
            p.player_id for p in mid_pool
        }
        # Mappa percentile popolata
        assert len(state.role_percentile_map) == len(mid_pool)

    def test_numero_partecipanti_non_corrisponde(
        self, mid_pool: list[Player]
    ) -> None:
        with pytest.raises(ValueError, match="num_participants"):
            initialize_auction(
                _participants(4), AuctionConfig(num_participants=5), mid_pool
            )
        with pytest.raises(ValueError, match="num_participants"):
            initialize_auction(
                _participants(3), AuctionConfig(num_participants=4), mid_pool
            )

    def test_partecipanti_duplicati(self, mid_pool: list[Player]) -> None:
        ps = _participants(4)
        ps[1] = ParticipantSetup(
            participant_id=ps[0].participant_id,
            display_name="dup",
            budget_initial=500,
        )
        with pytest.raises(ValueError, match="duplicate participant_id"):
            initialize_auction(ps, AuctionConfig(num_participants=4), mid_pool)

    def test_inflation_baseline_abilitata_richiede_config(
        self, mid_pool: list[Player]
    ) -> None:
        with pytest.raises(ValueError, match="inflation_config is required"):
            AuctionConfig(num_participants=4, use_inflation_baseline=True)

    def test_inflation_baseline_con_config_invalido(
        self, mid_pool: list[Player]
    ) -> None:
        cfg = AuctionConfig(
            num_participants=4,
            use_inflation_baseline=True,
            inflation_config=cast_to_object("not_an_inflation_config"),
        )
        with pytest.raises(TypeError, match="InflationConfig"):
            initialize_auction(_participants(4), cfg, mid_pool)

    def test_inflation_baseline_con_config_valido(
        self, mid_pool: list[Player]
    ) -> None:
        cfg = AuctionConfig(
            num_participants=4,
            use_inflation_baseline=True,
            inflation_config=InflationConfig(),
        )
        state = initialize_auction(_participants(4), cfg, mid_pool)
        assert state is not None


def cast_to_object(s: str) -> object:
    """Helper: cast di una stringa a ``object`` per test di tipo non valido."""
    return s


# ---------------------------------------------------------------------------
# record_assignment — 4-step validation
# ---------------------------------------------------------------------------


class TestRecordAssignmentValidation:
    def test_step1_unknown_player(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        result = record_assignment(state, "ghost", "u1", 10)
        assert result.success is False
        assert result.rejection_code == "unknown_player"
        assert "ghost" in (result.rejection_reason or "")
        assert result.updated_state is None

    def test_step1_player_already_assigned(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        assert record_assignment(state, "p1", "u1", 10).success
        result = record_assignment(state, "p1", "u2", 10)
        assert result.success is False
        assert result.rejection_code == "player_already_assigned"
        assert result.updated_state is None

    def test_step2_unknown_winner(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        result = record_assignment(state, "p1", "ghost_user", 10)
        assert result.success is False
        assert result.rejection_code == "unknown_participant"

    def test_step3_role_full(self, mid_pool: list[Player]) -> None:
        """Compila 3 portieri su u1, poi tenta di assegnare un 4° portiere."""
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        for pid in ("p1", "p2", "p3"):
            assert record_assignment(state, pid, "u1", 1).success
        result = record_assignment(state, "p4", "u1", 1)
        assert result.success is False
        assert result.rejection_code == "role_full"
        assert "P" in (result.rejection_reason or "")

    def test_step4_credit_reserve_violation(self, mid_pool: list[Player]) -> None:
        """Regola della riserva crediti: u1 ha 500 crediti e 25 slot, dopo
        aver preso 24 giocatori a 1 credito, non può permettersi 5 crediti
        per il 25° giocatore (perché gli resterebbero 0 crediti, e ogni
        slot vuoto richiede >= 1 credito)."""
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        # 25 - 3 (P) = 22 giocatori rimanenti da comprare: usiamo una
        # simulazione alternativa. Compiliamo tutti i 25 slot su u1
        # lasciando i prezzi al minimo (1), poi verifichiamo che il
        # 25-esimo con prezzo > 1 sia rifiutato.
        # 25 giocatori totali su u1:
        # Assegna 25 giocatori (P, D, C, A totali) a 1 credito ciascuno.
        # 25 assegnazioni con budget 500 = OK (500 - 25 = 475).
        # Il 25° a 1: budget_residual = 500 - 24 = 476, max_allowed = 476.
        # Proviamo a 477 -> violazione.
        # Servono 25 giocatori: abbiamo solo 12 nel pool, quindi non
        # possiamo riempire la rosa.  Invece, testiamo direttamente
        # la regola al momento dell'assegnazione: se il budget è tale
        # che il prezzo proposto eccede (budget_residual - slot_rimasti
        # - 1), viene rifiutato.
        #
        # Caso semplice: u1 ha 500, slot rimasti = 25, max_allowed = 500-24=476.
        result = record_assignment(state, "p1", "u1", 477)
        assert result.success is False
        assert result.rejection_code == "credit_reserve_violation"
        assert "477" in (result.rejection_reason or "")

    def test_step4_credit_reserve_al_limite_ok(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        # max_allowed = 500 - 24 = 476 -> 476 deve passare
        result = record_assignment(state, "p1", "u1", 476)
        assert result.success is True

    def test_negative_price(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        result = record_assignment(state, "p1", "u1", -1)
        assert result.success is False
        assert result.rejection_code == "negative_price"

    def test_validazione_in_ordine_4_step(self, mid_pool: list[Player]) -> None:
        """Più violazioni contemporanee: viene catturata SOLO la prima
        (ordine 1 -> 2 -> 3 -> 4)."""
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        # Player inesistente + winner inesistente -> la 1 cattura
        result = record_assignment(state, "ghost", "ghost_user", 10)
        assert result.rejection_code == "unknown_player"


# ---------------------------------------------------------------------------
# record_assignment — happy path e mutazioni
# ---------------------------------------------------------------------------


class TestRecordAssignmentMutations:
    def test_assegnazione_valida_scala_budget(
        self, mid_pool: list[Player]
    ) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        result = record_assignment(state, "p1", "u1", 25)
        assert result.success
        u1 = state.participants["u1"]
        assert u1.budget_residual == 475
        assert len(u1.squad) == 1
        assert u1.squad[0].player_id == "p1"
        assert u1.role_breakdown["P"] == 1

    def test_assegnazione_valida_aggiorna_price_index(
        self, mid_pool: list[Player]
    ) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        # p1 (TOP, costo 30) pagato 20 -> ratio = 20/30 = 0.667
        # new_index = 0.7 * 1.0 + 0.3 * 0.667 = 0.7 + 0.2 = 0.9
        result = record_assignment(state, "p1", "u1", 20)
        assert result.success
        # Verifica che il record sia stato creato
        rec = state.assignments[0]
        assert rec.player.player_id == "p1"
        assert rec.tier == "TOP"
        assert rec.price_index_before == pytest.approx(1.0)
        assert rec.price_index_after == pytest.approx(0.9, rel=1e-3)

    def test_assegnazione_rimuove_dal_pool(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        n_before = len(state.available_pool)
        record_assignment(state, "p1", "u1", 10)
        assert len(state.available_pool) == n_before - 1
        assert not any(
            p.player_id == "p1" for p in state.available_pool
        )

    def test_assegnazioni_multiple_sequenza_corretta(
        self, mid_pool: list[Player]
    ) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        record_assignment(state, "p1", "u1", 10)
        record_assignment(state, "p2", "u2", 20)
        record_assignment(state, "d1", "u1", 5)
        assert [r.sequence_number for r in state.assignments] == [1, 2, 3]
        assert state.assignments[2].player.player_id == "d1"


# ---------------------------------------------------------------------------
# undo_last_assignment
# ---------------------------------------------------------------------------


class TestUndoLastAssignment:
    def test_undo_ripristina_budget(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        record_assignment(state, "p1", "u1", 25)
        undo_last_assignment(state)
        u1 = state.participants["u1"]
        assert u1.budget_residual == 500
        assert u1.squad == []
        assert u1.role_breakdown["P"] == 0

    def test_undo_ripristina_pool(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        n_before = len(state.available_pool)
        record_assignment(state, "p1", "u1", 10)
        assert len(state.available_pool) == n_before - 1
        undo_last_assignment(state)
        assert len(state.available_pool) == n_before
        assert any(p.player_id == "p1" for p in state.available_pool)

    def test_undo_ripristina_price_index_da_snapshot(
        self, mid_pool: list[Player]
    ) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        index_before = copy.deepcopy(state.price_index)
        record_assignment(state, "p1", "u1", 15)
        # L'indice P:TOP è cambiato
        assert state.price_index["P"]["TOP"] != pytest.approx(1.0)
        undo_last_assignment(state)
        # Dopo undo, l'indice deve essere identico allo stato iniziale
        for role in ("P", "D", "C", "A"):
            for tier in ("LOW", "MID", "TOP"):
                assert state.price_index[role][tier] == pytest.approx(
                    index_before[role][tier]
                )

    def test_undo_multiplo_sequenza(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        record_assignment(state, "p1", "u1", 10)
        record_assignment(state, "p2", "u2", 20)
        record_assignment(state, "d1", "u1", 5)
        assert len(state.assignments) == 3
        undo_last_assignment(state)
        undo_last_assignment(state)
        undo_last_assignment(state)
        assert len(state.assignments) == 0
        for ps in state.participants.values():
            assert ps.squad == []
            assert ps.budget_residual == 500

    def test_undo_su_stato_vuoto_solleva_errore(
        self, mid_pool: list[Player]
    ) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        with pytest.raises(IndexError, match="nessuna assegnazione"):
            undo_last_assignment(state)


# ---------------------------------------------------------------------------
# get_auction_summary
# ---------------------------------------------------------------------------


class TestGetAuctionSummary:
    def test_summary_immuntable_e_completo(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        record_assignment(state, "p1", "u1", 10)
        summary = get_auction_summary(state)
        assert len(summary.participants) == 4
        assert len(summary.assignments) == 1
        # Deep copy del price_index (non riferimento diretto)
        summary.price_index["P"]["TOP"] = 99.0
        assert state.price_index["P"]["TOP"] != 99.0


# ---------------------------------------------------------------------------
# serialize / deserialize
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip_semplice(self, mid_pool: list[Player]) -> None:
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        record_assignment(state, "p1", "u1", 15)
        record_assignment(state, "d1", "u2", 8)

        payload = serialize_state(state)
        restored = deserialize_state(payload)

        # Stato uguale (per quanto riguarda i campi serializzati)
        assert set(restored.participants.keys()) == set(state.participants.keys())
        for pid, ps in restored.participants.items():
            orig = state.participants[pid]
            assert ps.budget_residual == orig.budget_residual
            assert [p.player_id for p in ps.squad] == [
                p.player_id for p in orig.squad
            ]
            assert ps.role_breakdown == orig.role_breakdown
        assert len(restored.assignments) == len(state.assignments)
        for r_restored, r_orig in zip(
            restored.assignments, state.assignments
        ):
            assert r_restored.player.player_id == r_orig.player.player_id
            assert r_restored.final_price == r_orig.final_price
            assert r_restored.tier == r_orig.tier
            assert r_restored.role == r_orig.role
            assert r_restored.price_index_before == pytest.approx(
                r_orig.price_index_before
            )
            assert r_restored.price_index_after == pytest.approx(
                r_orig.price_index_after
            )

    def test_round_trip_preserva_reference_e_budget(
        self, mid_pool: list[Player]
    ) -> None:
        """reference_budget e budget_initial sopravvivono al round-trip."""
        cfg = AuctionConfig(
            num_participants=4,
            reference_budget=300,
            budget_initial=500,
        )
        state = initialize_auction(_participants(4), cfg, mid_pool)
        restored = deserialize_state(serialize_state(state))
        assert restored.config.reference_budget == 300
        assert restored.config.budget_initial == 500

    def test_round_trip_undo_dopo_deserialize(self, mid_pool: list[Player]) -> None:
        """Lo stato deserializzato deve supportare ancora l'undo."""
        state = initialize_auction(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        record_assignment(state, "p1", "u1", 15)
        record_assignment(state, "d1", "u2", 8)
        restored = deserialize_state(serialize_state(state))
        # Undo della seconda assegnazione
        undo_last_assignment(restored)
        assert len(restored.assignments) == 1
        assert restored.assignments[0].player.player_id == "p1"
        # u2 deve avere budget e roster azzerati
        u2 = restored.participants["u2"]
        assert u2.budget_residual == 500
        assert u2.squad == []


# ---------------------------------------------------------------------------
# AuctionSession façade
# ---------------------------------------------------------------------------


class TestAuctionSessionFacade:
    def test_record_via_sessione(self, mid_pool: list[Player]) -> None:
        sess = AuctionSession(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        result = sess.record("p1", "u1", 10)
        assert result.success
        assert sess.state.participants["u1"].budget_residual == 490

    def test_undo_via_sessione(self, mid_pool: list[Player]) -> None:
        sess = AuctionSession(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        sess.record("p1", "u1", 10)
        sess.undo()
        assert sess.state.participants["u1"].budget_residual == 500

    def test_projection_via_sessione(self, mid_pool: list[Player]) -> None:
        sess = AuctionSession(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        # p1: TOP GK (percentile 1.0), costo 30 -> expected_price = 30 * 1.0 = 30
        assert sess.projection("p1") == pytest.approx(30.0)
        sess.record("p1", "u1", 20)
        # p1 pagato 20, expected pre-update = 30 -> ratio = 20/30 = 0.6667
        # P:TOP_main = 0.7 * 1.0 + 0.3 * 0.6667 = 0.9
        # P:MID (adiacente a TOP) riceve spillover:
        #   new = 1.0 * (1 + 0.25 * (0.6667 - 1)) = 0.9167
        # p2: MID GK (percentile 0.6667), costo 25
        # expected for p2 = 25 * 0.9167 ≈ 22.9167
        assert sess.projection("p2") == pytest.approx(25.0 * 0.9167, rel=1e-4)

    def test_summary_via_sessione(self, mid_pool: list[Player]) -> None:
        sess = AuctionSession(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        sess.record("p1", "u1", 10)
        summary = sess.summary()
        assert len(summary.assignments) == 1
        assert summary.assignments[0].player.player_id == "p1"

    def test_serialize_via_sessione(self, mid_pool: list[Player]) -> None:
        sess = AuctionSession(
            _participants(4), AuctionConfig(num_participants=4), mid_pool
        )
        sess.record("p1", "u1", 10)
        payload = sess.serialize()
        participants_payload = cast(dict[str, object], payload["participants"])
        assert "u1" in participants_payload
        # Round-trip
        restored = deserialize_state(payload)
        assert restored.participants["u1"].budget_residual == 490

import pytest

from src.portfolio.conflict import (
    ConflictDecision,
    EntryRequest,
    PortfolioConflictEngine,
)


def make_engine(
    max_concurrent_positions: int = 2,
) -> PortfolioConflictEngine:
    return PortfolioConflictEngine(
        max_concurrent_positions=max_concurrent_positions,
    )


def test_single_entry_is_approved():
    engine = make_engine()

    result = engine.evaluate(
        EntryRequest(
            strategy_name="MRL1",
            side="long",
            quantity=2,
        )
    )

    assert result.decision is ConflictDecision.APPROVED
    assert result.approved is True
    assert engine.open_position_count == 0


def test_register_entry_creates_portfolio_position():
    engine = make_engine()

    position = engine.register_entry(
        EntryRequest(
            strategy_name="MRL1",
            side="long",
            quantity=2,
        )
    )

    assert position.strategy_name == "MRL1"
    assert position.side == "long"
    assert position.quantity == 2
    assert engine.open_position_count == 1
    assert engine.has_open_position("MRL1")


def test_same_strategy_second_entry_is_rejected():
    engine = make_engine()

    engine.register_entry(
        EntryRequest(
            strategy_name="MRL1",
            side="long",
        )
    )

    result = engine.evaluate(
        EntryRequest(
            strategy_name="MRL1",
            side="long",
        )
    )

    assert result.decision is ConflictDecision.REJECTED
    assert "already has an open" in result.reason


def test_different_strategies_can_coexist():
    engine = make_engine(max_concurrent_positions=2)

    engine.register_entry(
        EntryRequest(
            strategy_name="MRL1",
            side="long",
        )
    )

    result = engine.evaluate(
        EntryRequest(
            strategy_name="MRS2",
            side="short",
        )
    )

    assert result.approved is True


def test_max_concurrent_positions_is_enforced():
    engine = make_engine(max_concurrent_positions=2)

    engine.register_entry(
        EntryRequest(
            strategy_name="MRL1",
            side="long",
        )
    )

    engine.register_entry(
        EntryRequest(
            strategy_name="MRS2",
            side="short",
        )
    )

    result = engine.evaluate(
        EntryRequest(
            strategy_name="S2R",
            side="short",
        )
    )

    assert result.decision is ConflictDecision.REJECTED
    assert "maximum concurrent" in result.reason


def test_exit_releases_portfolio_slot():
    engine = make_engine(max_concurrent_positions=1)

    engine.register_entry(
        EntryRequest(
            strategy_name="MRL1",
            side="long",
        )
    )

    closed = engine.register_exit("MRL1")

    assert closed.strategy_name == "MRL1"
    assert engine.open_position_count == 0

    result = engine.evaluate(
        EntryRequest(
            strategy_name="S2R",
            side="short",
        )
    )

    assert result.approved is True


def test_exit_without_position_is_rejected():
    engine = make_engine()

    with pytest.raises(
        RuntimeError,
        match="has no open portfolio position",
    ):
        engine.register_exit("MRL1")


def test_batch_entries_are_deterministic():
    engine = make_engine(max_concurrent_positions=2)

    requests_a = [
        EntryRequest("MRS2", "short"),
        EntryRequest("MRL1", "long"),
    ]

    requests_b = [
        EntryRequest("MRL1", "long"),
        EntryRequest("MRS2", "short"),
    ]

    results_a = engine.evaluate_batch(requests_a)
    results_b = engine.evaluate_batch(requests_b)

    normalized_a = [(r.strategy_name, r.decision, r.reason) for r in results_a]

    normalized_b = [(r.strategy_name, r.decision, r.reason) for r in results_b]

    assert normalized_a == normalized_b


def test_duplicate_same_strategy_in_batch_only_one_is_approved():
    engine = make_engine(max_concurrent_positions=2)

    requests = [
        EntryRequest("MRL1", "long"),
        EntryRequest("MRL1", "short"),
    ]

    results = engine.evaluate_batch(requests)

    approved = [result for result in results if result.approved]

    rejected = [result for result in results if not result.approved]

    assert len(approved) == 1
    assert len(rejected) == 1
    assert "multiple simultaneous entries" in rejected[0].reason


def test_batch_respects_existing_positions():
    engine = make_engine(max_concurrent_positions=3)

    engine.register_entry(
        EntryRequest(
            strategy_name="MRL1",
            side="long",
        )
    )

    results = engine.evaluate_batch(
        [
            EntryRequest("MRL1", "long"),
            EntryRequest("MRS2", "short"),
            EntryRequest("S2R", "short"),
        ]
    )

    result_map = {result.strategy_name: result for result in results}

    assert result_map["MRL1"].approved is False
    assert result_map["MRS2"].approved is True
    assert result_map["S2R"].approved is True


def test_batch_does_not_partially_register_when_rejected():
    engine = make_engine(max_concurrent_positions=1)

    requests = [
        EntryRequest("MRL1", "long"),
        EntryRequest("MRS2", "short"),
    ]

    with pytest.raises(
        RuntimeError,
        match="portfolio batch rejected",
    ):
        engine.register_batch(requests)

    assert engine.open_position_count == 0
    assert engine.open_strategies == ()


def test_successful_batch_registers_all_positions():
    engine = make_engine(max_concurrent_positions=3)

    positions = engine.register_batch(
        [
            EntryRequest("MRS2", "short", quantity=3),
            EntryRequest("MRL1", "long", quantity=2),
        ]
    )

    assert len(positions) == 2
    assert engine.open_position_count == 2

    assert engine.has_open_position("MRL1")
    assert engine.has_open_position("MRS2")


def test_invalid_side_is_rejected():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="side must be either",
    ):
        engine.evaluate(
            EntryRequest(
                strategy_name="MRL1",
                side="flat",
            )
        )


def test_invalid_quantity_is_rejected():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="quantity must be greater than zero",
    ):
        engine.evaluate(
            EntryRequest(
                strategy_name="MRL1",
                side="long",
                quantity=0,
            )
        )


def test_invalid_max_concurrent_positions_is_rejected():
    with pytest.raises(
        ValueError,
        match="max_concurrent_positions",
    ):
        PortfolioConflictEngine(
            max_concurrent_positions=0,
        )


def test_reset_clears_all_portfolio_positions():
    engine = make_engine(max_concurrent_positions=3)

    engine.register_entry(EntryRequest("MRL1", "long"))

    engine.register_entry(EntryRequest("MRS2", "short"))

    assert engine.open_position_count == 2

    engine.reset()

    assert engine.open_position_count == 0
    assert engine.open_strategies == ()

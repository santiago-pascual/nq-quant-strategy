from src.paper.ibkr_observation_ledger import bar_value_hash, visible_bar_version


def test_causal_as_of_consumer_cannot_see_revision_before_observed():
    contract_id = 815824267
    bar_start = 1791504480  # 2026-10-09 00:08 UTC; exchange bar end is 00:09
    original = {"open": 30989.5, "high": 30989.5, "low": 30987.25,
                "close": 30989.5, "volume": 587.0}
    revised = {**original, "low": 30975.0, "close": 30975.75, "volume": 1571.0}
    original_hash = bar_value_hash(original)
    revised_hash = bar_value_hash(revised)
    records = [
        {"contract_id": contract_id, "bar_start_epoch_utc": bar_start,
         "observation_sequence": 1, "observed_at_utc": "2026-10-09T00:18:35.781000+00:00",
         "exchange_bar_end_utc": "2026-10-09T00:09:00+00:00", "bar_complete_when_observed": True,
         "ohlcv": original, "bar_value_sha256": original_hash, "value_version": 1, "is_revision": False},
        {"contract_id": contract_id, "bar_start_epoch_utc": bar_start,
         "observation_sequence": 2, "observed_at_utc": "2026-10-09T00:19:06.126000+00:00",
         "exchange_bar_end_utc": "2026-10-09T00:09:00+00:00", "bar_complete_when_observed": True,
         "ohlcv": revised, "bar_value_sha256": revised_hash, "value_version": 2, "is_revision": True},
    ]

    before_first = visible_bar_version(records, contract_id=contract_id,
                                       bar_start_epoch_utc=bar_start,
                                       as_of="2026-10-09T00:18:00+00:00")
    between_observations = visible_bar_version(records, contract_id=contract_id,
                                               bar_start_epoch_utc=bar_start,
                                               as_of="2026-10-09T00:19:00+00:00")
    after_revision = visible_bar_version(records, contract_id=contract_id,
                                          bar_start_epoch_utc=bar_start,
                                          as_of="2026-10-09T00:20:00+00:00")

    assert before_first is None
    assert between_observations["value_version"] == 1
    assert between_observations["ohlcv"]["close"] == 30989.5
    assert after_revision["value_version"] == 2
    assert after_revision["ohlcv"]["close"] == 30975.75
    assert original_hash != revised_hash


def test_as_of_selection_rejects_incomplete_bar_even_if_observed():
    record = {"contract_id": 1, "bar_start_epoch_utc": 1791504480,
              "observation_sequence": 1, "observed_at_utc": "2026-10-09T00:08:30+00:00",
              "bar_complete_when_observed": False, "ohlcv": {"close": 1}}
    assert visible_bar_version([record], contract_id=1, bar_start_epoch_utc=1791504480,
                               as_of="2026-10-09T00:10:00+00:00") is None

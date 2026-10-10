from src.paper.ibkr_delayed_diagnostic import run_diagnostic, validate_historical_bars


def test_valid_historical_bars_are_usable_and_utc_ordered():
    result = validate_historical_bars([
        {"timestamp": "1791378000", "open": 20000, "high": 20002, "low": 19999, "close": 20001, "volume": 12},
        {"timestamp": "1791378060", "open": 20001, "high": 20003, "low": 20000, "close": 20002, "volume": 8},
    ])
    assert result["usable"] is True
    assert result["valid_count"] == 2
    assert result["duplicate_timestamps"] == 0
    assert result["strictly_ordered"] is True
    assert result["first_timestamp_utc"].endswith("+00:00")


def test_duplicate_or_invalid_ohlcv_is_not_usable():
    rows = [
        {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"timestamp": 2, "open": 100, "high": 99, "low": 98, "close": 100, "volume": -1},
    ]
    result = validate_historical_bars(rows)
    assert result["usable"] is False
    assert result["duplicate_timestamps"] == 1
    assert result["invalid_count"] == 1


def test_missing_local_tws_returns_unavailable_without_ibapi_or_orders():
    result = run_diagnostic(host="127.0.0.1", port=1, timeout=0.1, quote_wait=0)
    assert result["tws_reachable"] is False
    assert result["verdict"] == "unavailable"
    assert result["orders_submitted"] is False


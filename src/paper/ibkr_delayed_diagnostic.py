"""Read-only IBKR TWS/Gateway connectivity diagnostic for delayed MNQ data.

This is intentionally not a MarketDataSource and is not imported by the Paper
runner. It never invokes any order or account-management API.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import socket
import threading
from typing import Any, Mapping, Sequence


def validate_historical_bars(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate returned one-minute OHLCV bars without assuming market hours."""
    parsed: list[tuple[int, float, float, float, float, float]] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        try:
            stamp = int(row["timestamp"])
            o, h, low, c, volume = (float(row[k]) for k in ("open", "high", "low", "close", "volume"))
            if not all(math.isfinite(v) for v in (o, h, low, c, volume)):
                raise ValueError("non-finite OHLCV")
            if min(o, h, low, c) <= 0 or volume < 0 or h < max(o, c, low) or low > min(o, c, h):
                raise ValueError("invalid OHLCV bounds")
            parsed.append((stamp, o, h, low, c, volume))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"bar[{index}]: {exc}")
    timestamps = [item[0] for item in parsed]
    duplicates = len(timestamps) - len(set(timestamps))
    ordered = all(left < right for left, right in zip(timestamps, timestamps[1:]))
    return {
        "count": len(rows),
        "valid_count": len(parsed),
        "invalid_count": len(errors),
        "duplicate_timestamps": duplicates,
        "strictly_ordered": ordered,
        "first_timestamp_utc": datetime.fromtimestamp(min(timestamps), timezone.utc).isoformat() if timestamps else None,
        "last_timestamp_utc": datetime.fromtimestamp(max(timestamps), timezone.utc).isoformat() if timestamps else None,
        "usable": bool(parsed) and not errors and duplicates == 0 and ordered,
        "errors": errors[:20],
    }


def _probe(host: str, port: int, timeout: float) -> tuple[bool, str | None]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, None
    except OSError as exc:
        return False, str(exc)


def run_diagnostic(*, host: str = "127.0.0.1", port: int = 7497, client_id: int = 73,
                   timeout: float = 12.0, quote_wait: float = 4.0) -> dict[str, Any]:
    reachable, socket_error = _probe(host, port, min(timeout, 2.0))
    result: dict[str, Any] = {
        "provider": "Interactive Brokers TWS/IB Gateway",
        "mode_requested": "delayed (reqMarketDataType(3)); broker may return live if entitled",
        "read_only": True,
        "orders_submitted": False,
        "endpoint": f"{host}:{port}",
        "tws_reachable": reachable,
        "api_connected": False,
        "contract_resolved": False,
        "quote_received": False,
        "historical_request_completed": False,
        "bar_quality": validate_historical_bars([]),
        "errors": [],
    }
    if not reachable:
        result["errors"].append(f"TWS/IB Gateway socket unavailable: {socket_error}")
        result["verdict"] = "unavailable"
        return result

    try:
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper
    except ImportError as exc:
        result["errors"].append(f"IBKR API package is not installed: {exc}")
        result["verdict"] = "dependency_missing"
        return result

    class DiagnosticApp(EWrapper, EClient):
        def __init__(self) -> None:
            EClient.__init__(self, self)
            self.connected_event = threading.Event()
            self.contracts_done = threading.Event()
            self.historical_done = threading.Event()
            self.snapshot_done = threading.Event()
            self.contracts: list[Any] = []
            self.bars: list[dict[str, Any]] = []
            self.errors: list[dict[str, Any]] = []
            self.market_data_type: int | None = None
            self.ticks: dict[str, float] = {}

        def nextValidId(self, orderId: int) -> None:  # noqa: N802 - IB API callback
            self.connected_event.set()

        def contractDetails(self, reqId: int, contractDetails: Any) -> None:  # noqa: N802
            self.contracts.append(contractDetails)

        def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
            self.contracts_done.set()

        def marketDataType(self, reqId: int, marketDataType: int) -> None:  # noqa: N802
            self.market_data_type = int(marketDataType)

        def tickPrice(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:  # noqa: N802
            if price is not None and math.isfinite(float(price)) and float(price) > 0:
                self.ticks[f"price_tick_{tickType}"] = float(price)

        def tickSize(self, reqId: int, tickType: int, size: Any) -> None:  # noqa: N802
            try:
                self.ticks[f"size_tick_{tickType}"] = float(size)
            except (TypeError, ValueError):
                pass

        def tickSnapshotEnd(self, reqId: int) -> None:  # noqa: N802
            self.snapshot_done.set()

        def historicalData(self, reqId: int, bar: Any) -> None:  # noqa: N802
            try:
                stamp = int(bar.date)
            except (TypeError, ValueError):
                self.errors.append({"request_id": reqId, "message": f"unexpected non-epoch historical timestamp: {bar.date!r}"})
                return
            try:
                volume = float(bar.volume)
            except (TypeError, ValueError):
                volume = float("nan")
            self.bars.append({"timestamp": stamp, "open": bar.open, "high": bar.high,
                              "low": bar.low, "close": bar.close, "volume": volume})

        def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
            self.historical_done.set()

        def error(self, reqId: int, errorCode: int, errorString: str, *args: Any) -> None:
            self.errors.append({"request_id": int(reqId), "code": int(errorCode), "message": str(errorString)})
            if reqId == 3:
                self.historical_done.set()

    app = DiagnosticApp()
    reader: threading.Thread | None = None
    try:
        app.connect(host, port, client_id)
        reader = threading.Thread(target=app.run, name="ibkr-readonly-diagnostic", daemon=True)
        reader.start()
        if not app.connected_event.wait(timeout):
            result["errors"].append("IBKR API handshake timed out; verify TWS API socket access and trusted IP settings")
            result["verdict"] = "api_handshake_failed"
            return result
        result["api_connected"] = True

        query = Contract()
        query.symbol, query.secType, query.exchange, query.currency = "MNQ", "FUT", "CME", "USD"
        query.multiplier = "2"
        app.reqContractDetails(1, query)
        if not app.contracts_done.wait(timeout):
            result["errors"].append("MNQ futures contract-details request timed out")
            result["verdict"] = "contract_lookup_timeout"
            return result

        matches = []
        for details in app.contracts:
            contract = details.contract
            if (contract.symbol == "MNQ" and contract.secType == "FUT" and
                    contract.currency == "USD" and str(contract.multiplier) == "2"):
                matches.append(details)
        current_month = datetime.now(timezone.utc).strftime("%Y%m")
        matches = [d for d in matches
                   if str(d.contract.lastTradeDateOrContractMonth)[:6] >= current_month]
        matches.sort(key=lambda d: str(d.contract.lastTradeDateOrContractMonth))
        if not matches:
            result["errors"].append("No MNQ USD multiplier-2 futures contract resolved on CME")
            result["contract_matches"] = len(app.contracts)
            result["verdict"] = "contract_not_resolved"
            return result

        selected = matches[0].contract
        details = matches[0]
        result["contract_resolved"] = True
        result["contract_matches"] = len(matches)
        result["contract"] = {
            "symbol": selected.symbol, "local_symbol": selected.localSymbol,
            "con_id": selected.conId, "expiry": selected.lastTradeDateOrContractMonth,
            "exchange": selected.exchange, "currency": selected.currency,
            "multiplier": selected.multiplier,
            "time_zone_id": getattr(details, "timeZoneId", None),
            "trading_hours": getattr(details, "tradingHours", None),
            "liquid_hours": getattr(details, "liquidHours", None),
        }
        # Diagnostic only: request delayed data and never place or modify orders.
        app.reqMarketDataType(3)
        app.reqMktData(2, selected, "", True, False, [])
        app.reqHistoricalData(3, selected, "", "1 D", "1 min", "TRADES", 0, 2, False, [])
        app.historical_done.wait(timeout * 4)
        if quote_wait > 0:
            app.snapshot_done.wait(quote_wait)
        app.cancelMktData(2)
        result["market_data_type"] = app.market_data_type
        result["delayed_quote_confirmed"] = bool(app.ticks) and app.market_data_type == 3
        result["quote_received"] = bool(app.ticks)
        result["quote_ticks"] = app.ticks
        result["historical_request_completed"] = app.historical_done.is_set()
        result["bar_quality"] = validate_historical_bars(app.bars)
        result["errors"].extend(app.errors)
        if app.market_data_type is not None:
            result["observed_market_data_mode"] = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed_frozen"}.get(app.market_data_type, "unknown")
        if result["bar_quality"]["usable"]:
            if result["delayed_quote_confirmed"]:
                result["verdict"] = "delayed_quote_and_usable_historical_bars"
            elif result["quote_received"]:
                result["verdict"] = "quote_and_usable_historical_bars_mode_not_delayed"
            else:
                result["verdict"] = "usable_historical_bars_quote_unconfirmed"
        else:
            result["verdict"] = "connected_but_no_usable_bars"
        result["contract_caveat"] = "Resolved listed expiry is diagnostic only; no continuous MNQ.v.0 roll mapping is implied."
        result["timestamp_caveat"] = "Historical bars requested with epoch timestamps, interpreted as UTC bar-start times."
        result["session_caveat"] = "IBKR trading/liquid-hours metadata is reported for review; existing reviewed CME calendar remains authoritative for Paper session flags."
        result["volume_caveat"] = "TRADES-bar volume is provider-specific and is not asserted equivalent to Databento volume."
        return result
    except Exception as exc:  # diagnostic should report failures, not fabricate success
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["verdict"] = "diagnostic_error"
        return result
    finally:
        if app.isConnected():
            try:
                app.disconnect()
            except Exception:
                pass
        if reader is not None:
            reader.join(timeout=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497, help="Paper TWS default 7497; IB Gateway Paper default 4002")
    parser.add_argument("--client-id", type=int, default=73)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--quote-wait", type=float, default=4.0)
    args = parser.parse_args()
    report = run_diagnostic(host=args.host, port=args.port, client_id=args.client_id,
                            timeout=args.timeout, quote_wait=args.quote_wait)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["bar_quality"]["usable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

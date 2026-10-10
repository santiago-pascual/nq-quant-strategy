"""Bounded, read-only probe for delayed IBKR MNQ streaming availability.

This probe is deliberately separate from Paper execution. It resolves one
explicitly pinned futures contract and subscribes to market data for a bounded
interval. It never imports or invokes any broker order API.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
from typing import Any


def streaming_confirmed(*, market_data_type: int | None, valid_price_tick_count: int) -> bool:
    """Require delayed mode and an actual finite price, not merely a size event."""
    return market_data_type == 3 and valid_price_tick_count > 0


def run_stream_probe(
    *, host: str = "127.0.0.1", port: int = 7497, client_id: int = 192,
    con_id: int = 815824267, local_symbol: str = "MNQZ6",
    expiry: str = "20261218", duration_seconds: float = 25.0,
    timeout_seconds: float = 8.0,
) -> dict[str, Any]:
    """Request delayed streaming ticks for pinned MNQZ6, without order access."""
    if not 5.0 <= duration_seconds <= 120.0:
        raise ValueError("duration_seconds must be between 5 and 120")
    if not 1.0 <= timeout_seconds <= 30.0:
        raise ValueError("timeout_seconds must be between 1 and 30")
    result: dict[str, Any] = {
        "provider": "Interactive Brokers TWS/IB Gateway",
        "endpoint": f"{host}:{port}",
        "mode_requested": "delayed market data (reqMarketDataType(3))",
        "read_only": True,
        "orders_submitted": False,
        "historical_api_used": False,
        "contract_expected": {
            "symbol": "MNQ", "local_symbol": local_symbol, "con_id": con_id,
            "expiry": expiry, "exchange": "CME", "currency": "USD", "multiplier": "2",
        },
        "stream_duration_seconds": duration_seconds,
        "tws_reachable": False,
        "api_connected": False,
        "paper_account_verified": False,
        "contract_resolved": False,
        "market_data_type": None,
        "valid_price_tick_count": 0,
        "other_tick_event_count": 0,
        "tick_types": [],
        "first_tick_seconds_after_subscription": None,
        "delayed_stream_confirmed": False,
        "errors": [],
    }
    try:
        with socket.create_connection((host, port), timeout=min(timeout_seconds, 2.0)):
            result["tws_reachable"] = True
    except OSError as exc:
        result["errors"].append(f"TWS socket unavailable: {exc}")
        result["verdict"] = "unavailable"
        return result

    try:
        from ibapi.client import EClient
        from ibapi.contract import Contract
        from ibapi.wrapper import EWrapper
    except ImportError as exc:
        result["errors"].append(f"IBKR API package unavailable: {exc}")
        result["verdict"] = "dependency_missing"
        return result

    class ProbeApp(EWrapper, EClient):
        def __init__(self) -> None:
            EClient.__init__(self, self)
            self.ready = threading.Event()
            self.accounts_ready = threading.Event()
            self.contract_done = threading.Event()
            self.accounts: list[str] = []
            self.contracts: list[Any] = []
            self.market_data_type: int | None = None
            self.tick_types: set[int] = set()
            self.valid_price_tick_count = 0
            self.other_tick_count = 0
            self.first_tick_monotonic: float | None = None
            self.errors: list[dict[str, Any]] = []

        def nextValidId(self, orderId: int) -> None:  # noqa: N802
            self.ready.set()

        def managedAccounts(self, accountsList: str) -> None:  # noqa: N802
            self.accounts = [item for item in accountsList.split(",") if item]
            self.accounts_ready.set()

        def contractDetails(self, reqId: int, details: Any) -> None:  # noqa: N802
            if reqId == 1:
                self.contracts.append(details.contract)

        def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
            if reqId == 1:
                self.contract_done.set()

        def marketDataType(self, reqId: int, marketDataType: int) -> None:  # noqa: N802
            if reqId == 2:
                self.market_data_type = int(marketDataType)

        def tickPrice(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:  # noqa: N802
            if reqId == 2 and math.isfinite(float(price)) and float(price) > 0:
                self._price_tick(tickType)

        def tickSize(self, reqId: int, tickType: int, size: Any) -> None:  # noqa: N802
            try:
                valid = math.isfinite(float(size)) and float(size) >= 0
            except (TypeError, ValueError):
                valid = False
            if reqId == 2 and valid:
                self.other_tick_count += 1
                self.tick_types.add(int(tickType))

        def tickString(self, reqId: int, tickType: int, value: str) -> None:  # noqa: N802
            if reqId == 2 and value:
                self.other_tick_count += 1
                self.tick_types.add(int(tickType))

        def _price_tick(self, tick_type: int) -> None:
            if self.first_tick_monotonic is None:
                self.first_tick_monotonic = time.monotonic()
            self.valid_price_tick_count += 1
            self.tick_types.add(int(tick_type))

        def error(self, reqId: int, errorCode: int, errorString: str, *args: Any) -> None:
            self.errors.append({"request_id": int(reqId), "code": int(errorCode),
                                "message": str(errorString)})

    app = ProbeApp()
    reader: threading.Thread | None = None
    subscription_start: float | None = None
    try:
        app.connect(host, port, client_id)
        reader = threading.Thread(target=app.run, name="ibkr-readonly-stream-probe", daemon=True)
        reader.start()
        if not app.ready.wait(timeout_seconds):
            raise TimeoutError("IBKR API handshake timed out")
        result["api_connected"] = True
        app.reqManagedAccts()
        app.accounts_ready.wait(min(timeout_seconds, 3.0))
        result["managed_account_count"] = len(app.accounts)
        result["paper_account_verified"] = bool(app.accounts) and all(
            account.startswith("DU") for account in app.accounts
        )

        query = Contract()
        query.conId = int(con_id)
        query.symbol = "MNQ"
        query.secType = "FUT"
        query.exchange = "CME"
        query.currency = "USD"
        query.multiplier = "2"
        query.lastTradeDateOrContractMonth = expiry
        app.reqContractDetails(1, query)
        if not app.contract_done.wait(timeout_seconds):
            raise TimeoutError("pinned contract details request timed out")
        matching = [contract for contract in app.contracts
                    if int(contract.conId) == con_id
                    and contract.localSymbol == local_symbol
                    and str(contract.lastTradeDateOrContractMonth) == expiry
                    and contract.symbol == "MNQ" and contract.secType == "FUT"
                    and contract.exchange in {"CME", "GLOBEX"}
                    and str(contract.multiplier) == "2"]
        if len(matching) != 1:
            raise RuntimeError(f"expected one pinned MNQ contract; resolved {len(matching)}")
        result["contract_resolved"] = True

        app.reqMarketDataType(3)
        subscription_start = time.monotonic()
        app.reqMktData(2, matching[0], "", False, False, [])
        time.sleep(duration_seconds)
        app.cancelMktData(2)
        result["market_data_type"] = app.market_data_type
        result["valid_price_tick_count"] = app.valid_price_tick_count
        result["other_tick_event_count"] = app.other_tick_count
        result["tick_types"] = sorted(app.tick_types)
        if app.first_tick_monotonic is not None and subscription_start is not None:
            result["first_tick_seconds_after_subscription"] = round(
                app.first_tick_monotonic - subscription_start, 3
            )
        result["delayed_stream_confirmed"] = streaming_confirmed(
            market_data_type=app.market_data_type,
            valid_price_tick_count=app.valid_price_tick_count,
        )
        result["errors"] = app.errors[-30:]
        result["delayed_fallback_notice"] = any(
            item["code"] == 10167 for item in app.errors
        )
        if result["delayed_stream_confirmed"] and result["paper_account_verified"]:
            result["verdict"] = "paper_account_and_delayed_stream_confirmed"
        elif not result["paper_account_verified"]:
            result["verdict"] = "paper_account_not_verified"
        elif result["delayed_fallback_notice"] and not app.valid_price_tick_count:
            result["verdict"] = "delayed_fallback_notice_but_no_stream_ticks"
        elif app.market_data_type == 3:
            result["verdict"] = "delayed_mode_reported_but_no_stream_ticks"
        else:
            result["verdict"] = "stream_not_confirmed"
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["verdict"] = "diagnostic_error"
    finally:
        if app.isConnected():
            try:
                app.disconnect()
            except Exception:
                pass
        if reader is not None:
            reader.join(timeout=1.0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=192)
    parser.add_argument("--duration-seconds", type=float, default=25.0)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    args = parser.parse_args()
    result = run_stream_probe(host=args.host, port=args.port, client_id=args.client_id,
                              duration_seconds=args.duration_seconds,
                              timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("delayed_stream_confirmed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
